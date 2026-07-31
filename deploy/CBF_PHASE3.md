# Phase 3 — CBF safety filter in the C++ deployer

A copy of `~/go2_deploy` (so the original is untouched) with the Control Barrier
Function safety filter ported to C++ and wired into the RL controller. The filter
minimally edits the velocity command so the body (x,y) cannot enter a virtual
circular keep-out zone. The human still drives via keyboard/gamepad; the CBF only
intervenes near the zone.

## What changed (vs ~/go2_deploy)
- `include/cbf_safety_filter.hpp` — NEW. Header-only closed-form single-integrator
  CBF (port of `~/Alex/go2_cbf/cbf_filter.py`), with body<->world rotation by yaw
  and the robot bounding-radius margin. No QP solver, no torch dependency.
- `include/user_controller.hpp` — `SimpleRLController` owns a `CBFSafetyFilter`,
  loads its params, and filters `cmd` at the top of `Calculate()` (before obs).
  `BasicUserController::setOdometry(x,y,yaw)` added (no-op default).
- `include/robot_controller.hpp` — subscribes to `rt/sportmodestate`, stores
  (x,y), and calls `ctrl->setOdometry(x, y, imu_yaw)` each control step.
- `include/param/cfg.hpp` — parses optional `cbf_*` keys (defaults: disabled).
- `src/main.cpp` — new arg `simple_rl_cbf` -> `params/simple_rl_cbf_config.yaml`.
- `params/simple_rl_cbf_config.yaml` — NEW. Same as simple_rl + an enabled zone.

State source: position from `rt/sportmodestate` (the sim bridge publishes it; the
real Go2 onboard estimator publishes it too). Yaw from the IMU (`robot_interface.rpy[2]`).

## Verification done
- C++ unit test matches the Python reference exactly:
  `cd test && g++ -std=c++17 -I../include test_cbf.cpp -o test_cbf && ./test_cbf`
  -> spot-checks identical; rollout steps=518, min_h=0.0011, reached (== Python).
- Builds clean (libtorch via lr_gen). `simple_rl` (no CBF) loads unchanged;
  `simple_rl_cbf` prints `[CBF] enabled: center=(2,0) r=0.5 margin=0.35 gamma=2`.

## Build
```bash
conda activate lr_gen          # provides libtorch for the build
cd ~/Alex/go2_deploy_cbf && mkdir -p build && cd build
cmake .. && make -j4
```

## Sim2sim runbook (needs YOUR display + keyboard — two terminals)
Terminal 1 — simulator with the FLAT CBF scene (red keep-out cylinder drawn at
(2,0), visual-only). `-s` picks the scene; no config.yaml edit needed:
```bash
cd ~/unitree_mujoco/simulate/build && ./unitree_mujoco -s scene_cbf.xml
```
(Scene file: ~/unitree_mujoco/unitree_robots/go2/scene_cbf.xml. The default
scene.xml has ramps/stairs at x~2 that overlap the zone, hence a flat scene.)

Terminal 2 — deployer with CBF:
```bash
conda activate lr_gen
cd ~/Alex/go2_deploy_cbf/build && ./go2_deploy simple_rl_cbf
```
Controls are now TOGGLES (press once = on at fixed speed, press again = off):
`Enter -> i -> s -> c` to stand and enter RL control, then `w`=forward,
`x`=back, `a`/`e`=turn left/right, space=stop. Drive `w` toward the red cylinder.
The deployer prints a live readout every ~0.5s:
```
[CBF] odom=(x,y) h=...                 # h shrinks as you approach (2,0)
[CBF] odom=(x,y) h=...  <-- ACTIVE (clamping)  cmd=(...)   # filter intervening
```
As the body nears the zone at (2,0), h -> 0 and the filter clamps the forward
command so it stops before entering. Compare with `./go2_deploy simple_rl` (no
filter), which walks straight in.

Notes:
- Keyboard forward axis was fixed for SimpleRLController: `w` now drives
  gamepad.lx (cmd[0]=forward). The stock deployer fed gamepad.ly, so `w` strafed.
- This policy veers laterally (Genesis->MuJoCo gap), so "forward" curves. Watch
  the printed odom and, if the robot misses the zone, move it onto the actual
  path via `cbf_cx`/`cbf_cy` in params/simple_rl_cbf_config.yaml (no rebuild
  needed -- yaml is read at startup).

Tune the zone / aggressiveness in `params/simple_rl_cbf_config.yaml`
(`cbf_cx,cbf_cy,cbf_r,cbf_margin,cbf_gamma`); `cbf_enable: false` disables it.

## Hardware caveat (Phase 5)
On the real Go2, low-level control and the sport service can conflict — the
deployer deactivates `mcf`, and `rt/sportmodestate` odometry may stop or drift
while in low-level mode. Confirm live position is still published (and its frame)
before trusting a position-based keep-out on hardware. Yaw from IMU and position
from the estimator must share one frame.
