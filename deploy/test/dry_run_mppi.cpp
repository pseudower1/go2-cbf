// Headless dry run of the deployer's MPPI control path using the REAL config file.
// Replicates SimpleRLController::Calculate()'s use_mppi_ branch + the dead-reckoning
// integration exactly, on the kinematic unicycle the dead-reckoning assumes. Validates
// config -> planner -> EMA smoothing -> CBF output-guard -> reaches goal, body clear.
// (The interactive sim2sim still needs a display+keyboard; this checks the logic.)
//   g++ -std=c++17 -O2 -I../include dry_run_mppi.cpp -o dry_run_mppi -lyaml-cpp && ./dry_run_mppi
#include <cstdio>
#include <cmath>
#include "param/cfg.hpp"
#include "mppi_cbf_planner.hpp"
using namespace unitree::common;

int main(int argc, char** argv)
{
    const char* cfg_path = (argc > 1) ? argv[1]
        : "/home/localadmin/Alex/go2_deploy_cbf/params/simple_rl_mppi_config.yaml";
    SimpleRLCfg cfg(cfg_path);

    // ---- build the planner exactly as user_controller.hpp loadParam() does ----
    MPPIParams mp;
    mp.num_samples = cfg.mppi_samples; mp.horizon = cfg.mppi_horizon; mp.dt = cfg.mppi_dt;
    mp.lambda = cfg.mppi_lambda; mp.sigma_v = cfg.mppi_sigma_v; mp.sigma_w = cfg.mppi_sigma_w;
    mp.v_max = cfg.cbf_vdes; mp.w_max = cfg.cbf_wmax; mp.gamma = cfg.cbf_gamma;
    mp.v_deadzone = cfg.mppi_deadzone; mp.w_near = cfg.mppi_w_near; mp.band = cfg.mppi_band;
    mp.w_head = cfg.mppi_w_head; mp.w_smooth = cfg.mppi_w_smooth;
    const double margin = cfg.cbf_margin;
    if (!cfg.cbf_obstacles.empty())
        for (const auto& o : cfg.cbf_obstacles) mp.obstacles.push_back({o[0], o[1], o[2] + margin});
    else
        mp.obstacles.push_back({cfg.cbf_cx, cfg.cbf_cy, cfg.cbf_r + margin});
    MPPICBFPlanner planner(mp);
    const double gx = cfg.cbf_goal_x, gy = cfg.cbf_goal_y, goal_tol = cfg.cbf_goal_tol;
    const double smooth = cfg.mppi_smooth; const int replan_every = cfg.mppi_replan_every;
    const double dt = cfg.dt;   // 0.02 control tick

    printf("config=%s\n", cfg_path);
    printf("goal=(%.2f,%.2f) [%.1f ft]  tol=%.2f  %zu obstacle(s)  K=%d H=%d  smooth=%.2f replan=%d\n",
           gx, gy, gx * 3.28084, goal_tol, mp.obstacles.size(), mp.num_samples, mp.horizon,
           smooth, replan_every);
    for (auto& o : mp.obstacles)
        printf("  keep-out enforced (%.2f,%.2f) r_eff=%.2f  (drawn surface at %.2f m)\n",
               o.cx, o.cy, o.r_eff, o.r_eff - margin);

    // ---- closed loop: deployer Calculate() mppi branch + dead-reckoning ----
    double dr_x = 0, dr_y = 0, heading = 0;       // start frame (RL begins here, +x ahead)
    double mppi_v = 0, mppi_w = 0, v_filt = 0, w_filt = 0;
    double min_clear = 1e9; int tick = 0; bool reached = false;
    for (int i = 0; i < 2000; ++i) {
        double tgx = gx - dr_x, tgy = gy - dr_y, dist = std::hypot(tgx, tgy);
        if (dist < goal_tol) { reached = true; break; }
        if (tick % replan_every == 0)
            planner.plan(dr_x, dr_y, heading, gx, gy, mppi_v, mppi_w);
        ++tick;
        v_filt = smooth * v_filt + (1 - smooth) * mppi_v;
        w_filt = smooth * w_filt + (1 - smooth) * mppi_w;
        double v = v_filt, w = w_filt;
        double cap = planner.forwardSpeedCap(dr_x, dr_y, std::cos(heading), std::sin(heading));
        if (v > cap) v = cap;
        // clearance to nearest drawn obstacle surface
        for (auto& o : mp.obstacles) {
            double d = std::hypot(dr_x - o.cx, dr_y - o.cy) - (o.r_eff - margin);
            if (d < min_clear) min_clear = d;
        }
        // dead-reckoning + kinematic truth (heading from "IMU" = integral of omega)
        heading += w * dt;
        dr_x += v * std::cos(heading) * dt;
        dr_y += v * std::sin(heading) * dt;
        if (i % 25 == 0)   // mirrors the deployer's throttled [MPPI] readout
            printf("[MPPI] t=%4.1f dr=(%.2f,%.2f) h=%+.3f dist2goal=%.2f cmd=(%.2f,0,%+.2f)\n",
                   i * dt, dr_x, dr_y, planner.lastH(), dist, v, w);
    }
    double dg = std::hypot(gx - dr_x, gy - dr_y);
    printf("---\nfinal dr=(%.2f,%.2f)  dist2goal=%.3f  min_clearance=%+.3f m\n",
           dr_x, dr_y, dg, min_clear);
    bool ok = reached && min_clear >= -1e-3;
    printf("%s\n", ok ? "DRY RUN OK: drives to the goal around the keep-out, body stays clear"
                      : "DRY RUN FAIL: check config/tuning");
    return ok ? 0 : 1;
}
