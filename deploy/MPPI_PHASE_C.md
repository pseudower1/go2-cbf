# MPPI-CBF on the real Go2 (Phase C)

C++ port of the sim MPPI-CBF planner (`~/Alex/go2_mppi_cbf/`) into this deployer. The
robot now drives **itself** point-A → point-B around **known** virtual obstacles, using
the same sampling-based, CBF-biased planner validated in sim — no human driving.

This supersedes the reactive `cbf_nav` go-around mode (which handled a single obstacle
via a hand-tuned circulation term). Both still exist; the planner is selected by config.

## What was added
| file | role |
|---|---|
| `include/mppi_cbf_planner.hpp` | header-only C++ port of `MPPICBFPlanner` (no torch/Eigen). Plans (v, ω) over a horizon; CBF biases the sampling; multiple obstacles. |
| `test/test_mppi.cpp` | behavioral-parity test (no torch/SDK): closed-loop reach around single/double/slalom obstacles, body stays clear. |
| `params/simple_rl_mppi_config.yaml` | the planner config: goal, obstacle list, MPPI params. |
| wiring in `include/param/cfg.hpp`, `include/user_controller.hpp`, `src/main.cpp` | parse `cbf_mppi`/`cbf_obstacles`/`mppi_*`; run the planner in `Calculate()`; register `simple_rl_mppi`. |

State comes from the existing **dead-reckoning** (`dr_x_`,`dr_y_`) + relative IMU heading,
reset on RL-control entry — everything is in the robot's **start frame** (origin where you
press L1+A, +x straight ahead). The planner output is EMA-smoothed and passed through the
CBF output-guard, then becomes the policy's velocity command `(v, 0, ω)`.

## Build & test
```bash
# build the deployer (libtorch via lr_gen)
conda run -n lr_gen bash -c "cd ~/Alex/go2_deploy_cbf/build && make -j4"

# planner unit test (standalone, no torch) — expect ALL PASS
cd ~/Alex/go2_deploy_cbf/test && g++ -std=c++17 -O2 -I../include test_mppi.cpp -o test_mppi && ./test_mppi
```
Timing: K=384 ≈ 1 ms/plan, K=1024 ≈ 2.8 ms/plan (dev CPU); replans every 7 ticks, so it
fits comfortably inside one 20 ms control tick even at a few× slower onboard. Bump
`mppi_samples` for harder fields; drop it if the onboard CPU can't keep up.

## Configure the course (`params/simple_rl_mppi_config.yaml`)
All in the start frame. `cbf_margin` (body radius) is added to every obstacle radius.
```yaml
cbf_mppi: true
cbf_goal_x: 3.0     # drive 3 m ahead
cbf_goal_y: 0.0
cbf_goal_tol: 0.30  # >= ~0.3 so the policy deadzone doesn't stall the last creep
cbf_obstacles:      # [x, y, r] drawn radius
  - [1.5, 0.25, 0.4]
  # - [2.4, -0.4, 0.4]   # add more for a course
```
Keep obstacles slightly **off** the straight start→goal line — a keep-out exactly on the
line is a symmetry the sampler only escapes by luck (the real robot's settling drift
usually breaks it anyway).

## Run — sim2sim first (no robot)
```bash
# T1: cd ~/unitree_mujoco/simulate/build && ./unitree_mujoco -s scene_cbf.xml
# T2: conda activate lr_gen && cd ~/Alex/go2_deploy_cbf/build && ./go2_deploy simple_rl_mppi
#     arm: press Enter, then  1 (sit) -> 2 (stand) -> 3 (RL control)   [4 = damping/E-stop]
#     the robot then drives ITSELF toward the goal around the obstacle(s).
#     watch the [MPPI] dr=(x,y) h=... dist2goal=... cmd=(v,0,w) readout.
```
(Optional: run `~/Alex/digital_twin/mirror.py --iface ...` to visualize.)

## Run — hardware
```bash
sudo ip addr add 192.168.123.100/24 dev enp128s31f6 && sudo ip link set enp128s31f6 up
cd ~/Alex/go2_deploy_cbf/build && ./go2_deploy simple_rl_mppi enp128s31f6
# L1+R1 sit -> L1+R2 stand -> L1+A RL control (robot STARTS NAVIGATING IMMEDIATELY) -> L1+Y damping (E-stop)
```

### ⚠️ Safety — read before the first hardware run
- **The robot moves on its own the instant you enter RL control (L1+A).** Unlike the
  safety-filter mode (where you drive), here it immediately walks toward the goal. Keyboard
  driving does nothing in this mode.
- Second person on **L1+Y (damping)** at all times; ≥2 m clearance all around the whole path.
- For the first run keep it small: one obstacle, goal ≤ 3 m, on flat open floor. Place a
  real box at the configured obstacle location so the avoidance is visible (obstacles are
  virtual/known — the robot doesn't sense them, it trusts the config + dead-reckoning).
- Dead-reckoning drifts (~0.35 m over a run) and assumes the unicycle model; the `cbf_margin`
  absorbs some, but don't trust tight clearances on hardware. Start with generous margins.

## Known limitations (carried over from sim, all in the planner notes)
- **Deadzone endgame:** near the goal MPPI commands ~0.2 m/s (the policy's <0.5 deadzone),
  so it can stall the last ~0.25 m. `cbf_goal_tol ≥ 0.30` declares arrival before that.
- **Tight gaps:** corridors below ~0.6 m stack both obstacles' soft costs and won't thread;
  offset obstacles so there's always a clear side.
- **No perception:** obstacles are known/configured (Phase D adds L1 LiDAR sensing).
