# go2-cbf

A Control Barrier Function (CBF) safety filter for the Unitree Go2 quadruped,
taken from a kinematic prototype all the way to a real-time C++ filter running
on the physical robot. The CBF wraps an existing reinforcement-learning
locomotion policy and minimally edits its velocity command so the robot's body
can never enter a keep-out zone, while the learned policy handles low-level
tracking.

## Repository layout

| Folder | What it is |
| --- | --- |
| [`sim/`](sim/) | Python CBF filter + MuJoCo closed-loop simulation. Kinematic single-integrator prototype (`sim_kinematic.py`), the closed-form CBF (`cbf_filter.py`), and the MuJoCo + trained-policy loop (`mujoco_cbf_sim.py`). |
| [`deploy/`](deploy/) | The on-robot deployer: the CBF safety filter ported to a header-only C++ implementation (`include/cbf_safety_filter.hpp`) and wired into the RL controller (Unitree SDK2 + LibTorch). See [`deploy/CBF_PHASE3.md`](deploy/CBF_PHASE3.md). |
| [`digital_twin/`](digital_twin/) | A MuJoCo Go2 that live-mirrors the real robot from its `rt/lowstate` DDS channel (read-only), reconstructing body pose from foot-anchored leg odometry. See [`digital_twin/README_twin.txt`](digital_twin/README_twin.txt). |

## The arc

1. **Prototype** — a closed-form, single-integrator CBF for one circular
   keep-out zone; closed-form half-space projection, no QP solver.
2. **Closed loop in simulation** — the same filter shaping the velocity command
   of a trained locomotion policy in MuJoCo; the barrier `h(t)` stays
   non-negative while the base remains upright.
3. **On the real robot** — the filter ported to a real-time C++ safety filter in
   the deployer, verified against the Python reference, and run on the physical
   Go2 alongside a live digital twin for side-by-side comparison.

## Not included

This is a **code-only** repository — the source for every stage is present, and
the scripts regenerate their own figures and videos when you run them. The
following are intentionally omitted (`.gitignore`d): compiled `build/` output,
trained policy weights (`deploy/models/*.pt`), the raw video recordings from the
digital twin, and generated plots/screenshots. See each folder's README for the
setup and commands to reproduce everything locally.

### External dependencies you provide

- **Trained policy weights** (`deploy/models/*.pt`) — supply your own genesis_lr /
  RL policy; the deployer and MuJoCo loop load it at runtime.
- **MuJoCo scene XML** and **Unitree SDK2 / LibTorch** — install per
  [`deploy/README.md`](deploy/README.md).
- **Conda env `go2_cbf`** (python 3.10, mujoco 3.9, torch cpu, numpy, matplotlib)
  for the simulation code — see [`sim/README.md`](sim/README.md).

## Acknowledgement

The deployer builds on [lupinjia/go2_deploy](https://github.com/lupinjia/go2_deploy)
and the Unitree SDK2 / LibTorch stack; the CBF safety filter, MuJoCo closed-loop
harness, and digital twin are original work.
