#pragma once

#include <cmath>

// Control Barrier Function safety filter (C++ port of ~/Alex/go2_cbf/cbf_filter.py).
//
// Model:    p_dot = u,            p,u in R^2   (body (x,y), world velocity)
// Barrier:  h(p)  = ||p - c||^2 - r_eff^2,     r_eff = r + margin
// Filter:   u* = argmin ||u - u_des||^2  s.t.  h_dot + gamma*h >= 0
//
// One constraint => closed-form half-space projection (no QP solver). The user's
// velocity command is body-frame {vx,vy}; we rotate to world by yaw, filter, and
// rotate back. The RL policy is the low-level velocity tracker.

namespace unitree::common
{

struct CBFParams
{
    bool   enable = false;
    double cx = 0.0;        // keep-out center x (world)
    double cy = 0.0;        // keep-out center y (world)
    double r = 0.5;         // keep-out radius
    double margin = 0.0;    // robot bounding radius added to r (protects body, not just base point)
    double gamma = 2.0;     // extended class-K gain: alpha(h) = gamma * h
};

class CBFSafetyFilter
{
public:
    CBFSafetyFilter() = default;
    explicit CBFSafetyFilter(const CBFParams &p) : params_(p) {}

    void setParams(const CBFParams &p) { params_ = p; }
    const CBFParams &params() const { return params_; }
    double lastH() const { return last_h_; }
    bool lastActive() const { return last_active_; }

    // Filter a body-frame velocity command in place given world pose (x, y, yaw[rad]).
    // Returns true if the command was modified.
    bool filter(double x, double y, double yaw, float &vx, float &vy)
    {
        const double r_eff = params_.r + params_.margin;
        const double c = std::cos(yaw), s = std::sin(yaw);

        // body -> world:  u = R(yaw) * v
        double ux = c * vx - s * vy;
        double uy = s * vx + c * vy;

        // barrier and its gradient
        const double dx = x - params_.cx, dy = y - params_.cy;
        const double h = dx * dx + dy * dy - r_eff * r_eff;
        const double ax = 2.0 * dx, ay = 2.0 * dy;     // grad h
        const double b = -params_.gamma * h;           // require a . u >= b
        const double aa = ax * ax + ay * ay;
        const double slack = ax * ux + ay * uy - b;    // >= 0 means already safe

        last_h_ = h;
        last_active_ = false;
        if (aa > 1e-12 && slack < 0.0)
        {
            const double k = -slack / aa;              // project onto a . u = b
            ux += k * ax;
            uy += k * ay;
            last_active_ = true;
        }

        // world -> body:  v = R(yaw)^T * u
        vx = static_cast<float>(c * ux + s * uy);
        vy = static_cast<float>(-s * ux + c * uy);
        return last_active_;
    }

    // Filter a WORLD-frame velocity (ux,uy) in place at world position (x,y).
    // Same closed-form projection as filter(), but no body<->world rotation —
    // used by the go-to-goal navigation which works directly in the world frame.
    bool filterWorld(double x, double y, double &ux, double &uy)
    {
        const double r_eff = params_.r + params_.margin;
        const double dx = x - params_.cx, dy = y - params_.cy;
        const double h = dx * dx + dy * dy - r_eff * r_eff;
        const double ax = 2.0 * dx, ay = 2.0 * dy;
        const double b = -params_.gamma * h;
        const double aa = ax * ax + ay * ay;
        const double slack = ax * ux + ay * uy - b;
        last_h_ = h;
        last_active_ = false;
        if (aa > 1e-12 && slack < 0.0)
        {
            const double k = -slack / aa;
            ux += k * ax;
            uy += k * ay;
            last_active_ = true;
        }
        return last_active_;
    }

private:
    CBFParams params_{};
    double last_h_ = 0.0;
    bool last_active_ = false;
};

} // namespace unitree::common
