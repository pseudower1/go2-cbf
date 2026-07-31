"""Phase 1 kinematic demo for the single-integrator CBF safety filter.

A point (the Go2 body) tries to drive straight to a goal that sits directly
behind a circular keep-out zone. The nominal controller would plow through the
zone; the CBF filter minimally edits the velocity command so the trajectory
slides around it and h(t) >= 0 holds for all time.

Run:  conda run -n go2_cbf python sim_kinematic.py
Output: cbf_phase1.png  (trajectory + h(t) + speed), plus a console summary.
"""

import matplotlib
matplotlib.use("Agg")  # headless: save a PNG instead of opening a window
import matplotlib.pyplot as plt
import numpy as np

from cbf_filter import CircularKeepOut, SingleIntegratorCBF


def nominal_controller(p, goal, k_p=1.5, v_max=0.6):
    """Simple go-to-goal P controller, saturated at the Go2's command limit."""
    u = k_p * (goal - p)
    speed = np.linalg.norm(u)
    if speed > v_max:
        u = u * (v_max / speed)
    return u


def run(start=(-2.0, 0.05), goal=(2.0, 0.0),
        obs_center=(0.0, 0.0), obs_radius=0.5,
        gamma=2.0, dt=0.02, t_final=12.0):
    obstacle = CircularKeepOut(center=obs_center, radius=obs_radius)
    cbf = SingleIntegratorCBF(obstacle, gamma=gamma)

    p = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    ts, traj, hs, speeds_des, speeds_safe, active = [], [], [], [], [], []
    n_steps = int(t_final / dt)
    for i in range(n_steps):
        u_des = nominal_controller(p, goal)
        u_safe, info = cbf.filter(p, u_des)

        ts.append(i * dt)
        traj.append(p.copy())
        hs.append(info["h"])
        speeds_des.append(np.linalg.norm(u_des))
        speeds_safe.append(np.linalg.norm(u_safe))
        active.append(info["active"])

        p = p + dt * u_safe  # single-integrator forward Euler

        if np.linalg.norm(p - goal) < 0.05:
            break

    traj = np.array(traj)
    hs = np.array(hs)
    print(f"steps={len(ts)}  reached_goal={np.linalg.norm(p - goal) < 0.05}")
    print(f"min h(t) = {hs.min():.4f}   (>= 0 means the keep-out was respected)")
    print(f"filter active on {sum(active)}/{len(active)} steps")
    plot(traj, hs, ts, speeds_des, speeds_safe, obstacle, goal, start)
    return hs.min()


def plot(traj, hs, ts, speeds_des, speeds_safe, obstacle, goal, start):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- trajectory ---
    ax1.add_patch(plt.Circle(obstacle.center, obstacle.radius,
                             color="tomato", alpha=0.4, label="keep-out"))
    ax1.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="path (filtered)")
    ax1.plot([start[0], goal[0]], [start[1], goal[1]], "k--", alpha=0.4,
             label="nominal straight line")
    ax1.plot(*start, "go", label="start")
    ax1.plot(*goal, "m*", ms=15, label="goal")
    ax1.set_aspect("equal"); ax1.set_title("Trajectory"); ax1.legend(fontsize=8)
    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.grid(alpha=0.3)

    # --- barrier value ---
    ax2.plot(ts, hs, "b-", lw=2)
    ax2.axhline(0, color="r", ls="--", label="h = 0 (boundary)")
    ax2.set_title("Barrier h(t)"); ax2.set_xlabel("t [s]"); ax2.set_ylabel("h")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # --- speed: desired vs filtered ---
    ax3.plot(ts, speeds_des, "k--", label="|u_des|")
    ax3.plot(ts, speeds_safe, "b-", lw=2, label="|u_safe|")
    ax3.set_title("Command speed"); ax3.set_xlabel("t [s]")
    ax3.set_ylabel("|u| [m/s]"); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    fig.tight_layout()
    out = "cbf_phase1.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    run()
