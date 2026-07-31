#pragma once

#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

// MPPI-CBF planner (C++ port of ~/Alex/go2_mppi_cbf/mppi_cbf.py).
//
// Autonomous point-A -> point-B navigation with CBF-biased sampling. Plans in the
// UNICYCLE command space (v, omega) -- exactly the (s_cmd, 0, omega) the RL policy
// tracks -- over a short horizon, and uses the forward-speed CBF to BIAS the sampling
// (clip each sampled control into the safe set) so samples don't waste themselves
// driving into a keep-out. Multiple circular obstacles just sum cost terms (no QP).
//
// State (x, y, theta) comes from the deployer's dead-reckoning + relative IMU heading.
// Obstacles are enforced circles (cx, cy, r_eff) -- r_eff ALREADY includes the body margin.
//
// Validation: behavioral parity with the Python planner (reaches goal, min h >= 0);
// see test/test_mppi.cpp. Exact bit-match isn't meaningful (the planner is stochastic).

namespace unitree::common
{

struct Obstacle { double cx, cy, r_eff; };   // r_eff = drawn radius + body margin

struct MPPIParams
{
    int    horizon      = 45;
    int    num_samples  = 256;     // hardware default; sim uses up to 1024
    double dt           = 0.15;    // rollout (lookahead) timestep
    double v_max        = 0.6;
    double v_min        = 0.0;
    double w_max        = 1.0;
    double lambda       = 8.0;     // softmax temperature
    double sigma_v      = 0.3;
    double sigma_w      = 0.6;
    double gamma        = 2.0;     // CBF class-K gain
    double v_deadzone   = 0.5;     // policy ignores forward cmd below this -> model it
    double w_goal       = 1.0;
    double w_term       = 8.0;
    double w_collide    = 1.0e4;
    double w_near       = 10.0;
    double band         = 0.3;
    double w_head       = 1.0;
    double w_w          = 0.2;
    double w_smooth     = 3.0;
    unsigned long seed  = 0;
    std::vector<Obstacle> obstacles;
};

class MPPICBFPlanner
{
public:
    MPPICBFPlanner() = default;
    explicit MPPICBFPlanner(const MPPIParams &p) { setParams(p); }

    void setParams(const MPPIParams &p)
    {
        p_ = p;
        rng_.seed(p_.seed);
        U_v_.assign(p_.horizon, 0.0);
        U_w_.assign(p_.horizon, 0.0);
        last_uv_ = last_uw_ = 0.0;
        V_v_.assign((size_t)p_.num_samples * p_.horizon, 0.0);
        V_w_.assign((size_t)p_.num_samples * p_.horizon, 0.0);
        cost_.assign(p_.num_samples, 0.0);
        wts_.assign(p_.num_samples, 0.0);
    }
    const MPPIParams &params() const { return p_; }

    // Reset the warm-started nominal (call when RL control begins).
    void reset()
    {
        std::fill(U_v_.begin(), U_v_.end(), 0.0);
        std::fill(U_w_.begin(), U_w_.end(), 0.0);
        last_uv_ = last_uw_ = 0.0;
        rng_.seed(p_.seed);
    }

    double lastH() const { return last_h_; }   // min barrier over obstacles at last plan() pose

    // Max forward speed allowed by the forward-speed CBF over all obstacles, given
    // a unit motion direction (dcos, dsin). Returns +inf if nothing constrains.
    double forwardSpeedCap(double px, double py, double dcos, double dsin) const
    {
        double cap = std::numeric_limits<double>::infinity();
        for (const auto &o : p_.obstacles)
        {
            double dx = px - o.cx, dy = py - o.cy;
            double h = dx * dx + dy * dy - o.r_eff * o.r_eff;
            double a = 2.0 * dx * dcos + 2.0 * dy * dsin;   // grad_h . d_hat
            if (a < 0.0)                                     // approaching
            {
                double c = std::max(0.0, (-p_.gamma * h) / a);
                cap = std::min(cap, c);
            }
        }
        return cap;
    }

    // Plan one tick. Returns (v, omega) in (v_out, w_out); warm-starts internally.
    void plan(double x, double y, double theta, double gx, double gy,
              double &v_out, double &w_out)
    {
        const int H = p_.horizon, K = p_.num_samples;
        const double dt = p_.dt;
        std::normal_distribution<double> N01(0.0, 1.0);

        for (int k = 0; k < K; ++k)
        {
            double px = x, py = y, th = theta;
            double prev_v = last_uv_, prev_w = last_uw_;
            double cost = 0.0;
            const size_t base = (size_t)k * H;
            for (int t = 0; t < H; ++t)
            {
                double v = U_v_[t] + p_.sigma_v * N01(rng_);
                double w = U_w_[t] + p_.sigma_w * N01(rng_);
                v = clamp(v, p_.v_min, p_.v_max);
                w = clamp(w, -p_.w_max, p_.w_max);

                // heading-to-goal alignment
                double hx = gx - px, hy = gy - py;
                double dn = std::hypot(hx, hy) + 1e-9;
                double align = (std::cos(th) * hx + std::sin(th) * hy) / dn;
                cost += p_.w_head * (1.0 - align);

                // CBF sample-repair: clip forward speed so this step stays safe
                double cap = forwardSpeedCap(px, py, std::cos(th), std::sin(th));
                if (v > cap) v = cap;
                V_v_[base + t] = v;
                V_w_[base + t] = w;

                // smoothness (control-rate)
                cost += p_.w_smooth * ((v - prev_v) * (v - prev_v) + (w - prev_w) * (w - prev_w));
                prev_v = v; prev_w = w;

                // advance unicycle (deadzone-aware)
                double v_eff = (p_.v_deadzone > 0.0 && v < p_.v_deadzone) ? 0.0 : v;
                px += v_eff * std::cos(th) * dt;
                py += v_eff * std::sin(th) * dt;
                th += w * dt;

                // running costs
                cost += p_.w_goal * ((px - gx) * (px - gx) + (py - gy) * (py - gy));
                cost += p_.w_w * (w * w);
                for (const auto &o : p_.obstacles)
                {
                    double d = std::hypot(px - o.cx, py - o.cy) - o.r_eff;
                    if (d < 0.0) cost += p_.w_collide;
                    if (d < p_.band) cost += p_.w_near * (p_.band - d);
                }
            }
            cost += p_.w_term * ((px - gx) * (px - gx) + (py - gy) * (py - gy));
            cost_[k] = cost;
        }

        // importance weights
        double beta = cost_[0];
        for (int k = 1; k < K; ++k) beta = std::min(beta, cost_[k]);
        double sum = 0.0;
        for (int k = 0; k < K; ++k) { wts_[k] = std::exp(-(cost_[k] - beta) / p_.lambda); sum += wts_[k]; }
        double inv = 1.0 / (sum + 1e-9);
        for (int k = 0; k < K; ++k) wts_[k] *= inv;

        // nominal update: U += sum_k w_k (V_k - U)
        for (int t = 0; t < H; ++t)
        {
            double duv = 0.0, duw = 0.0;
            for (int k = 0; k < K; ++k)
            {
                const size_t idx = (size_t)k * H + t;
                duv += wts_[k] * (V_v_[idx] - U_v_[t]);
                duw += wts_[k] * (V_w_[idx] - U_w_[t]);
            }
            U_v_[t] = clamp(U_v_[t] + duv, p_.v_min, p_.v_max);
            U_w_[t] = clamp(U_w_[t] + duw, -p_.w_max, p_.w_max);
        }

        v_out = U_v_[0];
        w_out = U_w_[0];
        last_uv_ = v_out; last_uw_ = w_out;

        // record min barrier at the current pose (for logging)
        double h = std::numeric_limits<double>::infinity();
        for (const auto &o : p_.obstacles)
        {
            double dx = x - o.cx, dy = y - o.cy;
            h = std::min(h, dx * dx + dy * dy - o.r_eff * o.r_eff);
        }
        last_h_ = p_.obstacles.empty() ? 0.0 : h;

        // warm start: shift the nominal forward one step
        for (int t = 0; t < H - 1; ++t) { U_v_[t] = U_v_[t + 1]; U_w_[t] = U_w_[t + 1]; }
    }

private:
    static double clamp(double x, double lo, double hi)
    { return x < lo ? lo : (x > hi ? hi : x); }

    MPPIParams p_{};
    std::mt19937 rng_{0};
    std::vector<double> U_v_, U_w_;     // nominal (v, omega) sequence
    std::vector<double> V_v_, V_w_;     // sampled controls (K*H), reused each plan
    std::vector<double> cost_, wts_;
    double last_uv_ = 0.0, last_uw_ = 0.0;
    double last_h_ = 0.0;
};

} // namespace unitree::common
