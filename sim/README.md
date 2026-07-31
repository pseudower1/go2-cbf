# Go2 CBF Safety Filter

Recreating Aaron Ames-style Control Barrier Functions on the Unitree Go2, as a
safety filter wrapping the existing RL locomotion policy. The CBF shapes the
velocity **command** `cmd = {vx, vy, wz}` that the policy consumes; the RL policy
acts as the low-level velocity tracker.

## Environment
Isolated conda env (does not touch `lr_gen`/`unitree`/etc.):
```
conda activate go2_cbf      # python 3.10, numpy, matplotlib
```

## Phase 1 (done) — simplest CBF, kinematic prototype
- `cbf_filter.py` — single-integrator CBF for one circular keep-out zone.
  Barrier `h(p)=||p-p_o||^2 - r^2`. One constraint => closed-form half-space
  projection, no QP solver needed.
- `sim_kinematic.py` — point drives to a goal behind the keep-out; filter
  deflects it around. Saves `cbf_phase1.png`.
```
conda run -n go2_cbf python sim_kinematic.py
```
Result: `min h(t) >= 0`, goal reached, path rides the safe-set boundary.

## Phase 2 (done) — CBF in closed loop with MuJoCo + trained policy
- `mujoco_cbf_sim.py` — self-contained, in-process (no DDS). Replicates the
  go2_deploy SimpleRLController exactly (named-sensor joint order = SDK order,
  bridge PD law, 45-dim obs, gains/scales from `simple_rl_config.yaml`). Loads
  the exported policy via TorchScript (CPU).
```
conda run -n go2_cbf python mujoco_cbf_sim.py            # CBF on: stops at zone
conda run -n go2_cbf python mujoco_cbf_sim.py --no-cbf   # baseline: enters zone
conda run -n go2_cbf python mujoco_cbf_sim.py --viewer   # watch live (needs display)
# go AROUND the zone and reach a goal (obstacle on the straight line):
conda run -n go2_cbf python mujoco_cbf_sim.py --goal-x 2.5 --goal-y -1.8 \
    --obs-x 1.0 --obs-y -0.9 --obs-r 0.4 --duration 13 --video out.mp4
```

Two control modes:
- no `--goal`: forward-speed CBF, robot stops before the zone.
- with `--goal`: go-to-goal + CBF. Outer 2D CBF filter gives a safe velocity
  direction; a **circulation term** (tangential flow near the surface, weight
  fades with distance, `--band`) breaks the radial deadlock so the field flows
  AROUND; inner loop steers actual motion direction onto it via yaw + speed
  (auto-compensates the policy veer). Result: reaches goal, min h>=0 throughout.
Result: CBF OFF -> min h = -0.20 (enters). CBF ON -> min h = +0.04 (stops at
boundary), base upright throughout. Plots: `cbf_phase2.png`, `cbf_phase2_nocbf.png`.

Findings worth keeping:
- This Genesis-trained policy ignores forward commands below ~0.5 m/s and has a
  strong lateral veer in MuJoCo (sim2sim gap) -> can't strafe reliably.
- So Phase 2 uses a **forward-speed CBF**: modulate forward speed along the
  robot's actual motion direction (the only control we trust). Robust to veer
  and the natural variant for hardware. The 2D world-velocity filter in
  `cbf_filter.py` still backs it.
- Needs deps in go2_cbf: `mujoco` (3.9), `torch` (2.12 cpu) -- installed.

## Roadmap
- **Phase 3** — port the closed-form filter to C++ in
  `~/go2_deploy/include/user_controller.hpp` (before obs construction).
- **Phase 4** — harder CBFs: unicycle/nonholonomic model, multiple obstacles /
  geofence polygon (needs a real QP solver, e.g. proxsuite), HOCBF.
- **Phase 5** — hardware. Key dependency: a body (x,y) estimate (sport-mode
  odometry / velocity integration). Keep obstacles virtual. Existing safety
  checklist: 2 m clearance, second person on E-stop.

> **Note:** the MPPI-CBF line (autonomous A→B navigation by biasing an MPPI
> sampler with the CBF) now lives in its own folder: `~/Alex/go2_mppi_cbf/`.
