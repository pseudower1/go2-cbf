// Standalone unit test for cbf_safety_filter.hpp (no torch / no SDK needed).
// Compile:  g++ -std=c++17 -I../include test_cbf.cpp -o test_cbf && ./test_cbf
//
// Replicates ~/Alex/go2_cbf/sim_kinematic.py to verify the C++ port produces the
// same trajectory / min-h / step count as the Python reference.

#include <cstdio>
#include <cmath>
#include "cbf_safety_filter.hpp"

using unitree::common::CBFParams;
using unitree::common::CBFSafetyFilter;

int main()
{
    CBFParams p;
    p.enable = true;
    p.cx = 0.0; p.cy = 0.0; p.r = 0.5; p.margin = 0.0; p.gamma = 2.0;
    CBFSafetyFilter cbf(p);

    // ---- spot checks (yaw=0 => body frame == world frame) ----
    printf("== spot checks (c=(0,0), r=0.5, gamma=2) ==\n");
    struct Case { double x, y, vx, vy; } cases[] = {
        {-2.0, 0.05, 0.6, 0.0},   // far, approaching -> unmodified
        {-0.6, 0.0,  0.6, 0.0},   // near boundary, head-on -> reduced
        {-0.55,0.0,  0.6, 0.0},   // just outside, head-on
        { 0.0, 0.7,  0.0,-0.6},   // above zone, driving down into it
    };
    for (auto &c : cases) {
        float vx = (float)c.vx, vy = (float)c.vy;
        bool act = cbf.filter(c.x, c.y, 0.0, vx, vy);
        printf("p=(%+.2f,%+.2f) u_des=(%+.2f,%+.2f) -> u_safe=(%+.3f,%+.3f) h=%+.4f active=%d\n",
               c.x, c.y, c.vx, c.vy, vx, vy, cbf.lastH(), (int)act);
    }

    // ---- kinematic rollout (mirrors sim_kinematic.py) ----
    const double k_p = 1.5, v_max = 0.6, dt = 0.02;
    const double gx = 2.0, gy = 0.0;
    double x = -2.0, y = 0.05;
    double min_h = 1e9;
    int steps = 0; bool reached = false;
    for (int i = 0; i < 600; ++i) {
        // go-to-goal P controller, saturated
        double ux = k_p * (gx - x), uy = k_p * (gy - y);
        double sp = std::hypot(ux, uy);
        if (sp > v_max) { ux *= v_max / sp; uy *= v_max / sp; }

        float vx = (float)ux, vy = (float)uy;     // yaw=0 -> body==world
        cbf.filter(x, y, 0.0, vx, vy);
        if (cbf.lastH() < min_h) min_h = cbf.lastH();

        x += dt * vx; y += dt * vy;               // single-integrator Euler
        steps = i + 1;
        if (std::hypot(gx - x, gy - y) < 0.05) { reached = true; break; }
    }
    printf("\n== kinematic rollout ==\n");
    printf("steps=%d reached=%d min_h=%.4f final=(%.3f,%.3f)\n",
           steps, (int)reached, min_h, x, y);
    printf("expected (python): steps=518 reached=1 min_h~=0.0011\n");
    return 0;
}
