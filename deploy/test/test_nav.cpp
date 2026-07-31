// Unit test for the deployer's go-around navigation math (no torch/SDK).
// Replicates SimpleRLController's nav block on a kinematic unicycle (the model the
// dead-reckoning assumes) and checks it reaches the goal AROUND the zone with h>=0.
//   g++ -std=c++17 -I../include test_nav.cpp -o test_nav && ./test_nav
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "cbf_safety_filter.hpp"
using unitree::common::CBFParams;
using unitree::common::CBFSafetyFilter;

int main()
{
    CBFParams p; p.enable = true;
    p.cx = 1.5; p.cy = 0.0; p.r = 0.7; p.margin = 0.4; p.gamma = 2.0;
    CBFSafetyFilter cbf(p);
    const double gx = 50.0, gy = 0.0, vdes = 0.5, band = 0.8, kw = 1.5, wmax = 1.0,
                 goal_tol = 0.2, dt = 0.02, reff = p.r + p.margin;

    double x = 0, y = 0, th = 0, minh = 1e9;
    int steps = 0; bool reached = false;
    for (int i = 0; i < 2000; ++i) {
        double tgx = gx - x, tgy = gy - y, dist = std::hypot(tgx, tgy);
        double ux = 0, uy = 0;
        if (dist > goal_tol) {
            double ghx = tgx/dist, ghy = tgy/dist;
            double pcx = x - p.cx, pcy = y - p.cy, dc = std::hypot(pcx, pcy);
            double nhx = pcx/dc, nhy = pcy/dc;
            double w = std::max(0.0, std::min(1.0, (band - (dc - reff)) / band));
            double thx = -nhy, thy = nhx;
            if (thx*ghx + thy*ghy < 0) { thx = -thx; thy = -thy; }
            double dx = (1-w)*ghx + w*thx, dy = (1-w)*ghy + w*thy;
            double dn = std::hypot(dx, dy); if (dn > 1e-9) { dx/=dn; dy/=dn; }
            ux = vdes*dx; uy = vdes*dy;
        }
        cbf.filterWorld(x, y, ux, uy);
        if (cbf.lastH() < minh) minh = cbf.lastH();
        double desmag = std::hypot(ux, uy);
        double gx2 = desmag>1e-6 ? ux/desmag : std::cos(th);
        double gy2 = desmag>1e-6 ? uy/desmag : std::sin(th);
        double ch = std::cos(th), sh = std::sin(th);
        double e = std::atan2(ch*gy2 - sh*gx2, ch*gx2 + sh*gy2);
        double omega = std::max(-wmax, std::min(wmax, kw*e));
        double s = (dist > goal_tol) ? std::min(desmag, vdes) * std::max(0.0, std::cos(e)) : 0.0;
        // kinematic unicycle (what the dead-reckoning models)
        th += omega*dt; x += s*std::cos(th)*dt; y += s*std::sin(th)*dt;
        steps = i+1;
        if (std::hypot(gx-x, gy-y) < goal_tol) { reached = true; break; }
    }
    printf("steps=%d  reached=%d  min_h=%.4f  final=(%.2f,%.2f)  final_heading=%.1f deg\n",
           steps, (int)reached, minh, x, y, th * 180.0 / M_PI);
    double head_deg = std::fabs(th * 180.0 / M_PI);
    printf("%s\n", (minh >= -1e-3 && head_deg < 15.0)
           ? "PASS: goes around the zone (h>=0) and continues straight (heading ~0)"
           : "FAIL: check nav math");
    return 0;
}
