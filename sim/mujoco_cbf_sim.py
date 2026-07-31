"""Phase 2: CBF safety filter in closed loop with MuJoCo physics + the trained policy.

Self-contained, in-process (no DDS). Replicates the go2_deploy SimpleRLController
exactly:
  - joint state read from NAMED sensors  -> SDK order (FR,FL,RR,RL) == policy order
  - PD law matches the sim bridge:  ctrl = kp*(q_des - q) + kd*(0 - dq)
  - 45-dim observation, action_scale, stand_pos, gains all from simple_rl_config.yaml

The CBF (cbf_filter.py, single-integrator, one circular keep-out) is applied to the
WORLD-frame velocity command, then rotated into the policy's BODY-frame `cmd`.
Ground-truth body (x,y) comes straight from MuJoCo qpos.

Run (headless, saves plots):
    conda run -n go2_cbf python mujoco_cbf_sim.py
Watch live (needs a display):
    conda run -n go2_cbf python mujoco_cbf_sim.py --viewer
Disable the filter to see the policy plow through the zone (baseline):
    conda run -n go2_cbf python mujoco_cbf_sim.py --no-cbf
"""

import argparse
import os

import numpy as np
import mujoco
import torch

from cbf_filter import CircularKeepOut, SingleIntegratorCBF

SCENE = os.path.expanduser("~/unitree_mujoco/unitree_robots/go2/scene.xml")
POLICY = os.environ.get("GO2_POLICY", os.path.expanduser(
    "~/go2_deploy/models/May23_18-28-23_simple_rl_genesis_ite-1.pt"))

# ---- from simple_rl_config.yaml ----
CTRL_DT = 0.02
STAND_KP, STAND_KD = 60.0, 5.0
CTRL_KP, CTRL_KD = 20.0, 0.5
ACTION_SCALE = 0.25
LIN_VEL_SCALE, ANG_VEL_SCALE = 1.0, 0.25
DOF_POS_SCALE, DOF_VEL_SCALE = 1.0, 0.05
NUM_OBS = 45
STAND_POS = np.array([0.0, 0.8, -1.5] * 4, dtype=np.float64)  # per-leg (hip,thigh,calf)

# joint sensors in SDK order (matches actuator/ctrl order)
LEGS = ["FR", "FL", "RR", "RL"]
JOINTS = ["hip", "thigh", "calf"]
QPOS_SENSORS = [f"{l}_{j}_pos" for l in LEGS for j in JOINTS]
QVEL_SENSORS = [f"{l}_{j}_vel" for l in LEGS for j in JOINTS]


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rot_vec_by_quat(q, v):
    """Active rotation of v by quaternion q=[w,x,y,z]."""
    w, u = q[0], q[1:]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def add_zone_geom(scene, center, radius, height=0.6, rgba=(1.0, 0.2, 0.2, 0.35)):
    """Draw a translucent cylinder (keep-out obstacle, or a thin ground ring)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CYLINDER,
        np.array([radius, height / 2, 0.0]),          # cylinder size = (radius, half-height)
        np.array([center[0], center[1], height / 2]),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32))
    scene.ngeom += 1


def add_goal_geom(scene, goal, r=0.12):
    """Draw the goal as a green sphere."""
    if goal is None or scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, r, r]),
        np.array([goal[0], goal[1], r]), np.eye(3).flatten(),
        np.array([0.1, 0.9, 0.2, 0.9], dtype=np.float32))
    scene.ngeom += 1


class Deployer:
    """In-process replica of the go2_deploy RL controller, on MuJoCo."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.decim = max(1, round(CTRL_DT / self.model.opt.timestep))

        adr = lambda name: self.model.sensor(name).adr[0]
        self.qpos_adr = np.array([adr(n) for n in QPOS_SENSORS])
        self.qvel_adr = np.array([adr(n) for n in QVEL_SENSORS])
        self.quat_adr = adr("imu_quat")
        self.gyro_adr = adr("imu_gyro")
        self.fvel_adr = adr("frame_vel")

        self.policy = torch.jit.load(POLICY, map_location="cpu").eval()
        self.actions = np.zeros(12)

    # --- sensor reads (SDK order) ---
    def joint_pos(self):
        return self.data.sensordata[self.qpos_adr]

    def joint_vel(self):
        return self.data.sensordata[self.qvel_adr]

    def base_quat(self):
        return self.data.sensordata[self.quat_adr:self.quat_adr + 4].copy()

    def base_gyro(self):
        return self.data.sensordata[self.gyro_adr:self.gyro_adr + 3].copy()

    def projected_gravity(self):
        return rot_vec_by_quat(quat_conj(self.base_quat()), np.array([0, 0, -1.0]))

    def base_xy(self):
        return self.data.qpos[0:2].copy()

    def world_vel_xy(self):
        return self.data.sensordata[self.fvel_adr:self.fvel_adr + 2].copy()

    def base_z(self):
        return float(self.data.qpos[2])

    def yaw(self):
        h = rot_vec_by_quat(self.base_quat(), np.array([1.0, 0, 0]))
        return np.arctan2(h[1], h[0])

    # --- low level ---
    def apply_pd(self, q_des, kp, kd):
        tau = kp * (q_des - self.joint_pos()) + kd * (0.0 - self.joint_vel())
        self.data.ctrl[:] = tau

    def step_physics(self):
        for _ in range(self.decim):
            mujoco.mj_step(self.model, self.data)

    def reset_to_stand(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # 'home'
        self.data.qpos[2] = 0.33
        self.data.qpos[7:19] = STAND_POS
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def policy_step(self, cmd):
        """cmd = (vx_body, vy_body, wz). Returns jpos_des, runs one control tick."""
        obs = np.zeros(NUM_OBS, dtype=np.float32)
        obs[0] = cmd[0] * LIN_VEL_SCALE
        obs[1] = cmd[1] * LIN_VEL_SCALE
        obs[2] = cmd[2] * ANG_VEL_SCALE
        obs[3:6] = self.projected_gravity()
        obs[6:9] = self.base_gyro() * ANG_VEL_SCALE
        obs[9:21] = (self.joint_pos() - STAND_POS) * DOF_POS_SCALE
        obs[21:33] = self.joint_vel() * DOF_VEL_SCALE
        obs[33:45] = self.actions
        with torch.no_grad():
            out = self.policy(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
        self.actions = out
        jpos_des = out * ACTION_SCALE + STAND_POS
        self.apply_pd(jpos_des, CTRL_KP, CTRL_KD)
        self.step_physics()
        return jpos_des


def run(args):
    dep = Deployer()
    print(f"timestep={dep.model.opt.timestep}  decimation={dep.decim}  "
          f"control={1/CTRL_DT:.0f} Hz")

    # CBF protects the base POINT, so inflate the enforced radius by the robot's
    # bounding radius to keep the whole body out of the drawn keep-out zone.
    r_eff = args.obs_r + args.margin
    obstacle = CircularKeepOut(center=(args.obs_x, args.obs_y), radius=r_eff)
    cbf = SingleIntegratorCBF(obstacle, gamma=args.gamma)

    goal = None
    if args.goal_x is not None and args.goal_y is not None:
        goal = np.array([args.goal_x, args.goal_y])

    dep.reset_to_stand()
    # settle: hold stand pose with stiff gains
    for _ in range(args.settle_steps):
        dep.apply_pd(STAND_POS, STAND_KP, STAND_KD)
        dep.step_physics()
    print(f"settled: base_z={dep.base_z():.3f}  xy={dep.base_xy()}")

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(dep.model, dep.data)
        add_zone_geom(viewer.user_scn, obstacle.center, args.obs_r)
        add_goal_geom(viewer.user_scn, goal)

    renderer = cam = frames = None
    if args.video:
        renderer = mujoco.Renderer(dep.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = dep.model.body("base_link").id
        cam.distance, cam.elevation, cam.azimuth = 3.2, -25, 110
        frames = []
        render_every = max(1, round((1 / CTRL_DT) / args.fps))

    log = {k: [] for k in
           ("t", "x", "y", "z", "h", "cmd_des", "cmd_safe", "active", "yaw")}
    n = int(args.duration / CTRL_DT)
    for i in range(n):
        p = dep.base_xy()
        theta = dep.yaw()

        v = dep.world_vel_xy()
        speed = np.linalg.norm(v)
        d_hat = v / speed if speed > 0.15 else np.array([np.cos(theta), np.sin(theta)])

        if goal is not None:
            # GO-TO-GOAL + CBF.
            # Outer: a go-to-goal velocity, deflected by the 2D single-integrator
            # CBF so the safe direction skirts the keep-out (Phase-1 behaviour).
            to_goal = goal - p
            dist = float(np.linalg.norm(to_goal))
            g_to = to_goal / dist if dist > 1e-6 else np.zeros(2)
            # Circulation: a pure go-to-goal field points radially into an
            # obstacle that sits between robot and goal -> the CBF leaves no
            # tangential motion and the robot parks. Blend in a tangential flow
            # near the surface (weight w) so the nominal direction goes AROUND;
            # the CBF still guarantees safety.
            pc = p - obstacle.center
            dc = float(np.linalg.norm(pc))
            n_hat = pc / dc if dc > 1e-6 else np.array([1.0, 0.0])  # outward radial
            d_surf = dc - obstacle.radius                          # signed dist to boundary
            # circulation is part of the avoidance stack -> off when CBF is off,
            # so the baseline is a pure go-to-goal that drives straight through.
            w = (float(np.clip((args.band - d_surf) / args.band, 0.0, 1.0))
                 if args.cbf else 0.0)
            t_hat = np.array([-n_hat[1], n_hat[0]])                # tangent...
            if t_hat @ g_to < 0:                                   # ...toward goal side
                t_hat = -t_hat
            dir_des = (1.0 - w) * g_to + w * t_hat
            nrm = np.linalg.norm(dir_des)
            dir_des = dir_des / nrm if nrm > 1e-9 else g_to
            u_des = np.zeros(2) if dist < args.goal_tol else args.vdes * dir_des
            if args.cbf:
                u_safe, info = cbf.filter(p, u_des)
            else:
                u_safe, info = u_des, {"h": obstacle.h(p), "active": False}
            h = info["h"]

            # Inner: steer the robot's ACTUAL motion direction onto u_safe via yaw
            # (auto-compensates the policy's lateral veer), scale forward speed by
            # how well aligned we are, and stop at the goal.
            des_mag = float(np.linalg.norm(u_safe))
            g_hat = u_safe / des_mag if des_mag > 1e-6 else d_hat
            e = np.arctan2(d_hat[0] * g_hat[1] - d_hat[1] * g_hat[0], d_hat @ g_hat)
            omega = float(np.clip(args.kw * e, -args.wmax, args.wmax))
            s_cmd = min(des_mag, args.vdes) * max(0.0, np.cos(e))
            if dist < args.goal_tol:
                s_cmd, omega = 0.0, 0.0
            cmd = (s_cmd, 0.0, omega)
            active = info.get("active", False)
            s_des = des_mag
        else:
            # FORWARD-SPEED CBF (stop-before-zone): scale forward speed so that
            # h_dot + gamma*h >= 0 with h_dot = s * (grad_h . d_hat).
            h = obstacle.h(p)
            a = obstacle.grad_h(p) @ d_hat       # rate of h per unit forward speed
            s_des = args.vdes
            if args.cbf and a < 0.0:             # only constrains when approaching
                s_cmd = min(s_des, max(0.0, (-cbf.gamma * h) / a))
            else:
                s_cmd = s_des
            cmd = (s_cmd, 0.0, 0.0)
            active = s_cmd < s_des - 1e-6

        dep.policy_step(cmd)

        log["t"].append(i * CTRL_DT)
        log["x"].append(p[0]); log["y"].append(p[1]); log["z"].append(dep.base_z())
        log["h"].append(h); log["yaw"].append(theta)
        log["cmd_des"].append(s_des)
        log["cmd_safe"].append(s_cmd)
        log["active"].append(active)

        if dep.base_z() < 0.15:
            print(f"!! robot fell at t={i*CTRL_DT:.2f}s — policy/obs mismatch?")
            break
        if viewer is not None:
            viewer.sync()
        if renderer is not None and i % render_every == 0:
            renderer.update_scene(dep.data, cam)
            add_zone_geom(renderer.scene, obstacle.center, args.obs_r)       # red keep-out obstacle
            add_goal_geom(renderer.scene, goal)
            frames.append(renderer.render())

    if viewer is not None:
        viewer.close()
    if renderer is not None:
        import imageio
        imageio.mimsave(args.video, frames, fps=args.fps)
        renderer.close()
        print(f"saved {args.video}  ({len(frames)} frames)")

    hs = np.array(log["h"])
    print(f"steps={len(log['t'])}  final xy=({log['x'][-1]:.2f},{log['y'][-1]:.2f})"
          f"  base_z_final={log['z'][-1]:.3f}")
    print(f"min h(t) = {hs.min():.4f}   ({'SAFE' if hs.min() >= 0 else 'VIOLATED'})")
    print(f"filter active on {sum(log['active'])}/{len(log['active'])} steps")
    if goal is not None:
        d_goal = np.hypot(log['x'][-1] - goal[0], log['y'][-1] - goal[1])
        print(f"distance to goal = {d_goal:.3f} m  "
              f"({'REACHED' if d_goal < args.goal_tol + 0.05 else 'not reached'})")
    plot(log, obstacle, args, goal)


def plot(log, obstacle, args, goal=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(log["t"])
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.add_patch(plt.Circle(obstacle.center, args.obs_r, color="tomato",
                           alpha=0.4, label="keep-out"))
    a.add_patch(plt.Circle(obstacle.center, obstacle.radius, color="tomato",
                           fill=False, ls="--", alpha=0.6,
                           label="enforced (+robot margin)"))
    if goal is not None:
        a.plot(goal[0], goal[1], "g*", ms=16, label="goal")
    a.plot(log["x"], log["y"], "b-", lw=2, label="body path")
    a.plot(log["x"][0], log["y"][0], "go", label="start")
    a.set_aspect("equal"); a.grid(alpha=0.3); a.legend(fontsize=8)
    a.set_title(f"Top-down path  (CBF {'ON' if args.cbf else 'OFF'})")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]")

    a = ax[0, 1]
    a.plot(t, log["h"], "b-", lw=2); a.axhline(0, color="r", ls="--")
    a.set_title("Barrier h(t)"); a.set_xlabel("t [s]"); a.set_ylabel("h"); a.grid(alpha=0.3)

    a = ax[1, 0]
    a.plot(t, log["cmd_des"], "k--", label="|u_des|")
    a.plot(t, log["cmd_safe"], "b-", lw=2, label="|u_safe|")
    a.set_title("World velocity command"); a.set_xlabel("t [s]")
    a.set_ylabel("[m/s]"); a.legend(fontsize=8); a.grid(alpha=0.3)

    a = ax[1, 1]
    a.plot(t, log["z"], "b-"); a.axhline(0.15, color="r", ls="--", label="fall thresh")
    a.set_title("Base height (upright check)"); a.set_xlabel("t [s]")
    a.set_ylabel("z [m]"); a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout()
    out = "cbf_phase2.png" if args.cbf else "cbf_phase2_nocbf.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")

    # standalone barrier-value graph (for the side-by-side)
    if args.h_plot:
        f2, a2 = plt.subplots(figsize=(6, 4))
        a2.plot(t, log["h"], "b-", lw=2)
        a2.axhline(0, color="r", ls="--", label="h = 0 (zone boundary)")
        a2.fill_between(t, np.array(log["h"]), 0, where=np.array(log["h"]) < 0,
                        color="red", alpha=0.3)
        a2.set_xlabel("time [s]"); a2.set_ylabel("barrier value  h(x)")
        a2.set_xlim(0, 10)
        a2.set_title("CBF barrier h(x) vs time  (h >= 0 = safe)")
        a2.legend(); a2.grid(alpha=0.3); f2.tight_layout()
        f2.savefig(args.h_plot, dpi=120)
        print(f"saved {args.h_plot}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--no-cbf", dest="cbf", action="store_false")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--vdes", type=float, default=0.6)
    p.add_argument("--obs-x", type=float, default=0.98)
    p.add_argument("--obs-y", type=float, default=-1.40)
    p.add_argument("--obs-r", type=float, default=0.45)
    p.add_argument("--margin", type=float, default=0.35,
                   help="robot bounding radius added to enforced keep-out")
    p.add_argument("--duration", type=float, default=12.0)
    p.add_argument("--settle-steps", type=int, default=100)
    p.add_argument("--video", type=str, default=None, help="output mp4 path")
    p.add_argument("--h-plot", type=str, default=None, help="standalone h(x)-vs-time png")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--goal-x", type=float, default=None)
    p.add_argument("--goal-y", type=float, default=None)
    p.add_argument("--kw", type=float, default=1.5, help="yaw steering gain")
    p.add_argument("--wmax", type=float, default=1.0, help="max yaw rate cmd")
    p.add_argument("--goal-tol", type=float, default=0.25)
    p.add_argument("--band", type=float, default=0.8,
                   help="circulation influence band [m] outside the enforced surface")
    p.set_defaults(cbf=True)
    run(p.parse_args())
