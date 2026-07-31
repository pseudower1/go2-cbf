"""Live MuJoCo 'digital twin' of the real Go2 — 1:1 replay, displays AND records.

Subscribes (READ-ONLY) to rt/lowstate (12 joint angles + IMU orientation) and drives
a MuJoCo Go2 that matches the real robot exactly:
  - joints + body tilt come straight from the robot
  - body POSITION is derived from the legs (foot-anchored leg odometry): whichever
    feet are planted stay fixed on the ground, so the body moves exactly as the real
    gait dictates -> feet plant instead of sliding, a true 1:1 copy
  - the MPPI course is overlaid from the deployment YAML: obstacle(s) (drawn radius +
    keep-out ring), the goal (+ arrival tolerance), and the robot's trajectory trail
  - camera follows the robot

Never publishes to the robot, so it cannot affect it. Start it when the robot is
standing at its start spot, right as you enter RL control (key 3), so the twin's
origin lines up with the deployment's start frame (where the obstacle/goal live).

    conda activate unitree
    # if it can't find the robot / crashes in DDS, the ROS-Jazzy CycloneDDS is
    # shadowing Unitree's (same issue as the C++ deployer) -- prepend Unitree's lib:
    #   export LD_LIBRARY_PATH=/home/localadmin/opt/unitree_robotics/lib:$LD_LIBRARY_PATH
    python mirror.py --iface enp128s31f6
    python mirror.py --iface enp128s31f6 --test       # headless DDS check (is data arriving?)
    python mirror.py --selftest                        # headless RENDER check (no robot needed)
Course overlay comes from --config (default: the hw MPPI config). Camera: --cam-* .
"""
import argparse, os, time, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import yaml
import mujoco
import cv2
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

SCENE = os.path.expanduser("~/unitree_mujoco/unitree_robots/go2/scene_mppi.xml")  # flat; course drawn as overlays
COURSE_CFG = os.path.expanduser("~/Alex/go2_deploy_cbf/params/simple_rl_mppi_hw_config.yaml")
SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
              "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
              "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
FEET = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

_ls = {"q": np.zeros(12), "quat": np.array([1.0, 0, 0, 0]), "n": 0}


def _on_lowstate(msg: LowState_):
    for i in range(12):
        _ls["q"][i] = msg.motor_state[i].q
    _ls["quat"][:] = msg.imu_state.quaternion
    _ls["n"] += 1


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def load_course(path):
    """Read the MPPI deployment YAML -> (obstacles[(cx,cy,r)], margin, (gx,gy), tol).
    All in the robot's START frame, exactly as the deployer enforces them."""
    try:
        with open(os.path.expanduser(path)) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[course] config not found: {path} -- no overlay")
        return [], 0.0, (0.0, 0.0), 0.0
    margin = float(cfg.get("cbf_margin", 0.0) or 0.0)
    obstacles = [(float(o[0]), float(o[1]), float(o[2])) for o in (cfg.get("cbf_obstacles") or [])]
    goal = (float(cfg.get("cbf_goal_x", 0.0) or 0.0), float(cfg.get("cbf_goal_y", 0.0) or 0.0))
    tol = float(cfg.get("cbf_goal_tol", 0.0) or 0.0)
    print(f"[course] {len(obstacles)} obstacle(s), margin={margin}, goal={goal}, tol={tol}  (from {os.path.basename(path)})")
    return obstacles, margin, goal, tol


def add_geom(scn, gtype, size, pos, rgba):
    """Append one visualization geom to an already-populated MjvScene."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom], int(gtype),
        np.asarray(size, dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def draw_course(scn, obstacles, margin, goal, tol, trail):
    """Overlay the MPPI obstacle(s), goal, and trajectory trail onto the scene."""
    CYL = mujoco.mjtGeom.mjGEOM_CYLINDER
    SPH = mujoco.mjtGeom.mjGEOM_SPHERE
    for (cx, cy, r) in obstacles:
        # drawn radius: a short translucent red pillar (the physical obstacle)
        add_geom(scn, CYL, [r, 0.25, 0], [cx, cy, 0.25], [0.85, 0.15, 0.15, 0.55])
        # keep-out (drawn r + body margin): translucent ground disk the body must clear
        add_geom(scn, CYL, [r + margin, 0.005, 0], [cx, cy, 0.006], [0.95, 0.35, 0.20, 0.18])
    gx, gy = goal
    add_geom(scn, SPH, [0.08, 0, 0], [gx, gy, 0.12], [0.15, 0.85, 0.25, 0.95])   # goal marker
    if tol > 0:
        add_geom(scn, CYL, [tol, 0.005, 0], [gx, gy, 0.006], [0.15, 0.85, 0.25, 0.25])  # arrival tol
    for (tx, ty) in trail:
        add_geom(scn, SPH, [0.025, 0, 0], [tx, ty, 0.03], [0.20, 0.45, 0.95, 0.9])  # path trail


def build_world(scene_path, w, h):
    m = mujoco.MjModel.from_xml_path(scene_path)
    m.vis.global_.offwidth = w; m.vis.global_.offheight = h
    d = mujoco.MjData(m)
    qadr = [m.joint(n).qposadr[0] for n in SDK_JOINTS]
    foot_ids = [m.body(n).id for n in FEET]
    return m, d, qadr, foot_ids


def make_place(m, d, qadr, foot_ids, yaw0_ref, base_xy_ref):
    """Closure that sets joints + start-frame orientation + base, drops to floor."""
    def place():
        for k, a in enumerate(qadr):
            d.qpos[a] = _ls["q"][k]
        qz = np.array([math.cos(-yaw0_ref[0]/2), 0, 0, math.sin(-yaw0_ref[0]/2)])
        d.qpos[3:7] = quat_mul(qz, _ls["quat"])
        d.qpos[0:2] = base_xy_ref; d.qpos[2] = 0.0
        mujoco.mj_forward(m, d)
        fz = np.array([d.xpos[fid][2] for fid in foot_ids])
        d.qpos[2] = 0.02 - fz.min()
        mujoco.mj_forward(m, d)
        fxy = np.array([d.xpos[fid][:2].copy() for fid in foot_ids])
        fz = np.array([d.xpos[fid][2] for fid in foot_ids])
        return fxy, fz
    return place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="enp128s31f6")
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="headless DDS check (data arriving?)")
    ap.add_argument("--selftest", action="store_true", help="headless render check (no robot)")
    ap.add_argument("--scene", default=SCENE)
    ap.add_argument("--config", default=COURSE_CFG, help="deployment YAML for the course overlay")
    ap.add_argument("--mode", choices=["record", "view", "terminal"], default="record",
                    help="record = live window + mp4 | view = live window only | "
                         "terminal = headless live text readout (no window, no file)")
    ap.add_argument("--obstacle", action="append", default=[], metavar="CX,CY,R",
                    help="add a cylindrical CBF obstacle at (CX,CY) with radius R metres, "
                         "in the robot start frame. Repeatable, e.g. --obstacle 1.5,0,0.4")
    ap.add_argument("--obstacle-margin", type=float, default=None,
                    help="keep-out ring drawn around obstacles (default: config cbf_margin)")
    ap.add_argument("--trail", type=int, default=600, help="max trajectory trail points")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--cam-dist", type=float, default=3.2)
    ap.add_argument("--cam-az", type=float, default=120.0)
    ap.add_argument("--cam-el", type=float, default=-22.0)
    args = ap.parse_args()

    obstacles, margin, goal, tol = load_course(args.config)
    # append any cylindrical CBF obstacles given on the command line
    for spec in args.obstacle:
        try:
            cx, cy, r = (float(v) for v in spec.split(","))
            obstacles.append((cx, cy, r))
            print(f"[course] +CLI obstacle at ({cx}, {cy}) r={r}")
        except ValueError:
            print(f"[course] bad --obstacle '{spec}' (expected CX,CY,R) -- skipped")
    if args.obstacle_margin is not None:
        margin = args.obstacle_margin

    # ---- headless RENDER self-test: mock a standing robot, draw the course, save a PNG ----
    if args.selftest:
        _ls["q"][:] = np.array([0.0, 0.9, -1.8] * 4)   # rough stand pose
        _ls["quat"][:] = [1.0, 0, 0, 0]; _ls["n"] = 1
        m, d, qadr, foot_ids = build_world(args.scene, args.w, args.h)
        renderer = mujoco.Renderer(m, args.h, args.w)
        cam = mujoco.MjvCamera()
        cam.distance = args.cam_dist; cam.elevation = args.cam_el; cam.azimuth = args.cam_az
        yaw0 = [0.0]; base_xy = np.zeros(2)
        place = make_place(m, d, qadr, foot_ids, yaw0, base_xy)
        place()
        trail = [(x, 0.0) for x in np.linspace(0, goal[0] * 0.6, 40)] if goal[0] else []
        cam.lookat[:] = [goal[0] * 0.5, 0.0, 0.25]
        renderer.update_scene(d, cam)
        draw_course(renderer.scene, obstacles, margin, goal, tol, trail)
        png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_selftest.png")
        cv2.imwrite(png, cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        print(f"selftest OK: ngeom={renderer.scene.ngeom}, wrote {png}")
        renderer.close()
        return

    ChannelFactoryInitialize(args.domain, args.iface)
    ChannelSubscriber("rt/lowstate", LowState_).Init(_on_lowstate, 10)

    if args.test:
        print(f"listening on rt/lowstate (domain {args.domain}, {args.iface})...")
        for _ in range(10):
            time.sleep(0.3)
            print(f"  msgs={_ls['n']:4d}  q[0:3]={np.round(_ls['q'][:3],3)}  quat={np.round(_ls['quat'],3)}")
        print("OK" if _ls["n"] else "NO DATA")
        return

    m, d, qadr, foot_ids = build_world(args.scene, args.w, args.h)

    terminal = args.mode == "terminal"
    renderer = writer = win = out = cam = None
    if terminal:
        print(f"terminal mode: headless live readout on rt/lowstate "
              f"(domain {args.domain}, {args.iface}) -- Ctrl-C to stop", flush=True)
    else:
        renderer = mujoco.Renderer(m, args.h, args.w)
        cam = mujoco.MjvCamera()
        cam.distance = args.cam_dist; cam.elevation = args.cam_el; cam.azimuth = args.cam_az
        win = "Go2 Digital Twin (1:1 live)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        if args.mode == "record":
            out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           f"twin_{time.strftime('%b%d_%H-%M-%S')}.mp4")
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.w, args.h))
            print(f"recording to {out}  (click window, press q or ESC to stop)", flush=True)
        else:
            print("view mode: live window, not recording  (click window, press q or ESC to stop)", flush=True)

    base_xy = np.zeros(2)
    planted = [None] * 4          # world xy where each foot is anchored
    yaw0 = [None]                 # IMU yaw captured at start (start-frame reference)
    trail = []                    # leg-odometry trajectory (start frame)
    dt = 1.0 / args.fps
    place = make_place(m, d, qadr, foot_ids, yaw0, base_xy)

    try:
        while True:
            t0 = time.time()
            if yaw0[0] is None:
                if _ls["n"] == 0:
                    time.sleep(0.05); continue
                yaw0[0] = yaw_of(_ls["quat"])

            fxy, fz = place()
            contact = [i for i in range(4) if fz[i] < fz.min() + 0.03]
            # leg odometry: move the base so anchored (planted) feet stay put
            drifts = [planted[i] - fxy[i] for i in contact if planted[i] is not None]
            if drifts:
                base_xy += np.mean(drifts, axis=0)
                fxy, fz = place()
            # update anchors: feet in contact keep/set their planted xy; others release
            for i in range(4):
                planted[i] = (planted[i] if (i in contact and planted[i] is not None)
                              else (fxy[i].copy() if i in contact else None))

            # record the trajectory trail (subsample by distance to cap geom count)
            if not trail or np.hypot(*(base_xy - trail[-1])) > 0.03:
                trail.append(base_xy.copy())
                if len(trail) > args.trail:
                    trail.pop(0)

            if terminal:
                # live text readout: msg count, body pose (start frame), and CBF distances
                gd = math.hypot(base_xy[0] - goal[0], base_xy[1] - goal[1]) if (goal[0] or goal[1]) else float("nan")
                od = min((math.hypot(base_xy[0] - cx, base_xy[1] - cy) - r
                          for (cx, cy, r) in obstacles), default=float("nan"))
                yaw = math.degrees(yaw_of(_ls["quat"]) - yaw0[0])
                print(f"\r[live] msgs={_ls['n']:5d} feet_down={len(contact)} "
                      f"base=({base_xy[0]:+.2f},{base_xy[1]:+.2f})m yaw={yaw:+6.1f}deg "
                      f"goal={gd:5.2f}m nearest_obst={od:6.2f}m   ", end="", flush=True)
                time.sleep(max(0.0, dt - (time.time() - t0)))
                continue

            cam.lookat[:] = [base_xy[0], base_xy[1], 0.25]   # follow the robot
            renderer.update_scene(d, cam)
            draw_course(renderer.scene, obstacles, margin, goal, tol, trail)
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            if writer is not None:
                writer.write(frame)
            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print()
    finally:
        if writer is not None:
            writer.release(); print(f"saved {out}")
        if not terminal:
            cv2.destroyAllWindows(); renderer.close()


if __name__ == "__main__":
    main()
