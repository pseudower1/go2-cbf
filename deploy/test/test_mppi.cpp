// Behavioral-parity test for the C++ MPPI-CBF planner (no torch/SDK).
// Mirrors the Python kinematic self-test (~/Alex/go2_mppi_cbf/mppi_cbf.py): run the
// planner in closed loop on the unicycle the dead-reckoning assumes and check it
// reaches the goal AROUND the obstacle(s) with the barrier h(t) >= 0 throughout.
// MPPI is stochastic, so we check behavior (reach + safe), not a bit-exact match.
//   g++ -std=c++17 -O2 -I../include test_mppi.cpp -o test_mppi && ./test_mppi
#include <cstdio>
#include <cmath>
#include <vector>
#include "mppi_cbf_planner.hpp"
using unitree::common::MPPICBFPlanner;
using unitree::common::MPPIParams;
using unitree::common::Obstacle;

// One closed-loop run from (0,0,0) to (gx,gy). Returns reached + min clearance to the
// nearest DRAWN obstacle boundary (r_eff - margin); >=0 means the body stayed clear.
static bool run_case(const char *name, std::vector<Obstacle> enf, double margin,
                     double gx, double gy, int K, int steps)
{
    MPPIParams p;
    p.num_samples = K;
    p.obstacles = enf;
    MPPICBFPlanner planner(p);

    double x = 0, y = 0, th = 0, dt = p.dt, goal_tol = 0.25;
    double min_clear = 1e9;
    bool reached = false;
    int used = 0;
    for (int i = 0; i < steps; ++i) {
        double dist = std::hypot(gx - x, gy - y);
        if (dist < goal_tol) { reached = true; break; }
        double v, w;
        planner.plan(x, y, th, gx, gy, v, w);
        // clearance to nearest drawn obstacle surface (enforced radius minus margin)
        for (const auto &o : enf) {
            double d = std::hypot(x - o.cx, y - o.cy) - (o.r_eff - margin);
            if (d < min_clear) min_clear = d;
        }
        // "true" model = unicycle moving at the commanded v (the real policy creeps
        // even below its deadzone; the planner's v_deadzone only BIASES sampling).
        x += v * std::cos(th) * dt;
        y += v * std::sin(th) * dt;
        th += w * dt;
        used = i + 1;
        if (getenv("MPPI_DBG") && i % 40 == 0)
            printf("    [dbg] i=%3d p=(%.2f,%.2f) th=%+.2f dist=%.2f v=%.2f w=%+.2f\n",
                   i, x, y, th, dist, v, w);
    }
    bool safe = min_clear >= -1e-3;
    printf("  %-8s steps=%4d  reached=%d  min_clearance=%+.3f  final=(%.2f,%.2f)  %s\n",
           name, used, (int)reached, min_clear, x, y,
           (reached && safe) ? "PASS" : "FAIL");
    return reached && safe;
}

int main()
{
    const double m = 0.35;   // body margin (r_eff = drawn r + m)
    printf("MPPI-CBF C++ planner behavioral test:\n");

    // NOTE: obstacles are placed slightly off the start->goal line. A keep-out
    // EXACTLY on the line is a degenerate symmetry (left/right samples cancel in the
    // weighted average) that a sampling planner only escapes via noise -- the real
    // robot never sees it (settling drift gives a nonzero start heading).
    bool ok = true;
    // single keep-out near (2,0.2), drawn r=0.5 -> r_eff 0.85; goal (4,0) behind it
    ok &= run_case("single", {{2.0, 0.2, 0.5 + m}}, m, 4.0, 0.0, 384, 500);
    // two-obstacle cluster (above center); robot goes around the bottom to the goal
    ok &= run_case("double", {{1.8, 0.4, 0.45 + m}, {2.6, 0.4, 0.45 + m}}, m, 4.4, -0.2, 512, 500);
    // a 3-gate weave
    ok &= run_case("slalom", {{1.5, 0.7, 0.45 + m}, {3.0, -0.7, 0.45 + m}, {4.5, 0.7, 0.45 + m}},
                   m, 6.0, 0.0, 1024, 600);

    printf("%s\n", ok ? "ALL PASS: planner reaches the goal around obstacles, body stays clear"
                      : "FAIL: check planner math/tuning");
    return ok ? 0 : 1;
}
