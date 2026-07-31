===============================================================================
 Go2 DIGITAL TWIN  (digital_twin/mirror.py)
===============================================================================

WHAT IT IS
-------------------------------------------------------------------------------
A MuJoCo Go2 that faithfully replays what the REAL robot is doing, live.

It SUBSCRIBES (read-only) to the robot's rt/lowstate channel over DDS:
  - 12 measured joint angles + body tilt come straight from the robot
  - body POSITION is reconstructed from the legs (foot-anchored leg odometry):
    planted feet stay fixed on the ground, so the body moves exactly as the
    real gait dictates (feet plant instead of sliding -> a true 1:1 copy)

It NEVER publishes to the robot, so it CANNOT affect it. Safe to run alongside
the deployer and the camera bridge (all on DDS domain 0, iface enp128s31f6).

Because it reads STATE (not commands), it works no matter which control
interface your scripts use (lowcmd / sportmodecmd / anything).


SETUP (every terminal that runs it)
-------------------------------------------------------------------------------
    cd ~/Alex/digital_twin
    conda activate unitree
    # If DDS can't find the robot / it crashes on startup, the ROS-Jazzy
    # CycloneDDS is shadowing Unitree's. Prepend Unitree's lib:
    export LD_LIBRARY_PATH=/home/localadmin/opt/unitree_robotics/lib:$LD_LIBRARY_PATH


THE THREE MODES  (--mode)
-------------------------------------------------------------------------------
  record   (default)  live 3D window + saves an mp4 to digital_twin/twin_*.mp4
  view                live 3D window only, no file written
  terminal            NO window, NO file -- prints a live text readout instead

  RECORD a video:
    python mirror.py --iface enp128s31f6 --mode record

  WATCH live only (no recording):
    python mirror.py --iface enp128s31f6 --mode view

  TERMINAL live readout (headless, what's happening right now):
    python mirror.py --iface enp128s31f6 --mode terminal
      example line:
      [live] msgs=1234 feet_down=3 base=(+1.20,-0.05)m yaw=+12.3deg goal= 3.55m nearest_obst=  0.82m
        msgs         = lowstate messages received so far
        feet_down    = how many feet are currently in contact
        base         = body x,y in the START frame (metres), from leg odometry
        yaw          = heading relative to start (degrees)
        goal         = distance to goal (if a goal is configured)
        nearest_obst = distance from body centre to the nearest obstacle SURFACE
      Stop with Ctrl-C.

  In record/view modes: click the window and press  q  or  Esc  to stop.


ADDING CYLINDRICAL OBSTACLES (to represent a CBF keep-out)
-------------------------------------------------------------------------------
Obstacles are drawn as a solid red cylinder plus a translucent keep-out ring
(radius + margin). Two ways to add them:

  1) From a deployment YAML (cbf_obstacles / cbf_goal / cbf_margin):
       python mirror.py --iface enp128s31f6 --config <path-to-yaml>

  2) From the command line -- repeatable, coords are in the robot START frame:
       --obstacle CX,CY,R          (metres)
       --obstacle-margin M         (keep-out ring width, metres)

     example: one obstacle 1.5 m ahead, 0.4 m radius, 0.5 m keep-out:
       python mirror.py --iface enp128s31f6 --config /dev/null \
         --obstacle 1.5,0,0.4 --obstacle-margin 0.5

     two obstacles:
       python mirror.py --iface enp128s31f6 --config /dev/null \
         --obstacle 1.5,0.3,0.4 --obstacle 2.5,-0.4,0.3 --obstacle-margin 0.5

Notes:
  - CLI obstacles are ADDED on top of anything in --config.
  - Use --config /dev/null to suppress a config course and show ONLY your
    CLI obstacles (or nothing).
  - For the NP3O + camera-bridge run the obstacles are DYNAMIC (camera-
    detected), so no static YAML course applies -- use --config /dev/null
    and add --obstacle markers only if you want a fixed reference drawn.


HANDY CHECKS (no robot / diagnostics)
-------------------------------------------------------------------------------
  python mirror.py --iface enp128s31f6 --test      # is lowstate data arriving?
  python mirror.py --selftest                       # render check, NO robot,
                                                     # writes twin_selftest.png
  python mirror.py --selftest --obstacle 1.5,0,0.4 --obstacle-margin 0.5
                                                     # preview an obstacle layout


TIMING
-------------------------------------------------------------------------------
Start the twin as the robot begins moving from its start spot (right as you
enter RL control, key 3), so the leg-odometry origin lines up with the
deployment's obstacle/goal frame.

Caveat: body POSITION comes from leg odometry, so on very slippery ground or
long runs it can drift slightly from true world position. Joint angles and
body tilt are always exact.


RUNNING ALONGSIDE THE FULL STACK  (3 terminals)
-------------------------------------------------------------------------------
T1 -- deployer:
    cd ~/Alex/go2_deploy_cbf/build
    LD_LIBRARY_PATH=/home/localadmin/opt/unitree_robotics/lib:$LD_LIBRARY_PATH ./go2_deploy np3o enp128s31f6

T2 -- camera CBF bridge:
    cd ~/Alex/mobile_cbf
    conda run -n unitree python hardware/go2_camera_cbf.py --live --output bridge \
      --classes 56 --person-height 0.711 --goal-dist 5.0 --vx-max 0.5 \
      --r-min 0.6 --alpha 0.3 --duration 40

T3 -- digital twin (pick a mode):
    cd ~/Alex/digital_twin
    conda activate unitree
    export LD_LIBRARY_PATH=/home/localadmin/opt/unitree_robotics/lib:$LD_LIBRARY_PATH
    # record a video:
    python mirror.py --iface enp128s31f6 --mode record --config /dev/null
    # or watch live text:
    python mirror.py --iface enp128s31f6 --mode terminal --config /dev/null
    # or add a CBF obstacle marker at 2 m ahead:
    python mirror.py --iface enp128s31f6 --mode record --config /dev/null \
      --obstacle 2.0,0,0.4 --obstacle-margin 0.5

T1/T2 are unchanged from your normal run. T3 is read-only and cannot affect
either of them.
===============================================================================
