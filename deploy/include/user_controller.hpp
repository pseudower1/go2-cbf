#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <deque>

#include "torch/script.h"
#include "robot_interface.hpp"
#include "gamepad.hpp"
#include "param/cfg.hpp"
#include "cbf_safety_filter.hpp"
#include "mppi_cbf_planner.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/idl/ros2/Twist_.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>

namespace fs = std::filesystem;

namespace unitree::common
{
class BasicUserController
{
public:
    BasicUserController() {}

    virtual void loadParam() = 0;

    virtual void loadPolicy() = 0;

    virtual void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad) = 0;

    virtual ~BasicUserController() = default;

    virtual void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad) = 0;

    virtual void DummyCalculate() = 0;

    virtual void Calculate() = 0;

    virtual std::vector<float> GetLog() = 0;

    // Feed world odometry (x, y, yaw[rad]) for the CBF safety filter. No-op by default.
    virtual void setOdometry(double /*x*/, double /*y*/, double /*yaw*/) {}
    // Reset dead-reckoned position to origin (called when RL control begins). No-op by default.
    virtual void resetOdometry() {}
    // Expose dead-reckoned pose for the digital twin: x,y in the start frame + yaw0 (IMU yaw at RL start).
    virtual void getTwinState(double &x, double &y, double &yaw0) { x = 0; y = 0; yaw0 = 0; }

    void save_jpos(BasicRobotInterface &robot_interface)
    {
        std::copy(robot_interface.jpos.begin(), robot_interface.jpos.end(), start_pos.begin());
    }

    float dt;
    float stand_kp;
    float stand_kd;
    float ctrl_kp;
    float ctrl_kd;
    std::array<float, 12> start_pos; // 阻尼状态的位置
    std::array<float, 12> stand_pos; // 站立状态最终位置
    std::array<float, 12> jpos_des;
    std::array<float, 12> sit_pos; // sit状态最终位置
};


class SimpleRLController : public BasicUserController
{
public:
    SimpleRLController(const std::string& cfg_file)
    {
        // observation init to 0
        base_ang_vel.fill(0.0);
        projected_gravity.fill(0.);
        projected_gravity.at(2) = -1.0;
        cmd.fill(0.0);
        jpos_processed.fill(0.0);
        jvel.fill(0.0);
        actions.fill(0.0);
        config_file_name = cfg_file;
    }
    void loadParam()
    {
        SimpleRLCfg cfg(config_file_name);
        dt = cfg.dt;
        stand_kp = cfg.stand_kp;
        stand_kd = cfg.stand_kd;
        action_scale = cfg.action_scale;
        lin_vel_scale = cfg.lin_vel_scale;
        ang_vel_scale = cfg.ang_vel_scale;
        dof_pos_scale = cfg.dof_pos_scale;
        dof_vel_scale = cfg.dof_vel_scale;
        policy_name = cfg.policy_name;
        ctrl_kp = cfg.ctrl_kp;
        ctrl_kd = cfg.ctrl_kd;
        num_obs = cfg.num_obs;
        obs.resize(num_obs);
        for (int i = 0; i < 12; ++i)
        {
            stand_pos.at(i) = cfg.stand_pos.at(i);
            sit_pos.at(i) = cfg.sit_pos.at(i);
        }
        // CBF safety filter
        CBFParams cbf_p;
        cbf_p.enable = cfg.cbf_enable;
        cbf_p.cx = cfg.cbf_cx; cbf_p.cy = cfg.cbf_cy;
        cbf_p.r = cfg.cbf_r; cbf_p.margin = cfg.cbf_margin; cbf_p.gamma = cfg.cbf_gamma;
        cbf_.setParams(cbf_p);
        if (cbf_p.enable)
            std::cout << "[CBF] enabled: center=(" << cbf_p.cx << "," << cbf_p.cy
                      << ") r=" << cbf_p.r << " margin=" << cbf_p.margin
                      << " gamma=" << cbf_p.gamma << std::endl;
        // autonomous go-around navigation
        nav_ = cfg.cbf_nav; gx_ = cfg.cbf_goal_x; gy_ = cfg.cbf_goal_y;
        vdes_ = cfg.cbf_vdes; band_ = cfg.cbf_band; kw_ = cfg.cbf_kw;
        wmax_ = cfg.cbf_wmax; goal_tol_ = cfg.cbf_goal_tol;
        if (nav_)
            std::cout << "[NAV] go-around enabled: goal=(" << gx_ << "," << gy_
                      << ") vdes=" << vdes_ << " band=" << band_ << std::endl;
        // MPPI-CBF planner (supersedes the reactive nav when enabled). Reuses the
        // cbf_* goal/margin/gamma/vdes/wmax; obstacles from cbf_obstacles list (or the
        // single cbf_cx/cy/r as a fallback).
        use_mppi_ = cfg.cbf_mppi;
        if (use_mppi_)
        {
            MPPIParams mp;
            mp.num_samples = cfg.mppi_samples; mp.horizon = cfg.mppi_horizon; mp.dt = cfg.mppi_dt;
            mp.lambda = cfg.mppi_lambda; mp.sigma_v = cfg.mppi_sigma_v; mp.sigma_w = cfg.mppi_sigma_w;
            mp.v_max = cfg.cbf_vdes; mp.w_max = cfg.cbf_wmax; mp.gamma = cfg.cbf_gamma;
            mp.v_deadzone = cfg.mppi_deadzone; mp.w_near = cfg.mppi_w_near; mp.band = cfg.mppi_band;
            mp.w_head = cfg.mppi_w_head; mp.w_smooth = cfg.mppi_w_smooth;
            const double margin = cfg.cbf_margin;
            // An empty cbf_obstacles list means OBSTACLE-FREE (pure go-to-goal).
            // Do NOT fall back to cbf_cx/cy/r here: that silently planted a phantom
            // keep-out in the path, making the robot swerve/stall ("wobble") on a
            // walk the user intended to be obstacle-free. To use the single
            // cbf_cx/cy/r obstacle, list it explicitly under cbf_obstacles.
            for (const auto &o : cfg.cbf_obstacles)
                mp.obstacles.push_back({o[0], o[1], o[2] + margin});
            mppi_planner_.setParams(mp);
            mppi_smooth_ = cfg.mppi_smooth; mppi_replan_every_ = cfg.mppi_replan_every;
            use_sm_ = cfg.cbf_use_sm;
            std::cout << "[MPPI] position source: " << (use_sm_ ? "sportmodestate (ground truth)"
                                                               : "dead-reckoning (hardware fallback)") << std::endl;
            // launch the background planner thread (replans ~every replan_every ticks
            // of wall time, but OFF the control loop so the 50Hz tick never overruns).
            plan_period_s_ = std::max(1, mppi_replan_every_) * static_cast<double>(dt);
            running_.store(true);
            plan_thread_ = std::thread(&SimpleRLController::planLoop, this);
            std::cout << "[MPPI] enabled: " << mp.obstacles.size() << " obstacle(s), goal=("
                      << gx_ << "," << gy_ << ") K=" << mp.num_samples << " H=" << mp.horizon
                      << " dt=" << mp.dt << " async replan ~" << plan_period_s_ << "s" << std::endl;
        }
    }
    void loadPolicy()
    {
        fs::path model_path = fs::current_path() / "../models";
        policy = torch::jit::load(model_path / policy_name);
        std::cout << "Load policy from: " << model_path / policy_name << std::endl;
    }
    void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad)
    {
        // do nothing
    }
    void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad)
    {
        // save necessary data from input
        std::copy(robot_interface.gyro.begin(), robot_interface.gyro.end(), base_ang_vel.begin());
        std::copy(robot_interface.projected_gravity.begin(), robot_interface.projected_gravity.end(), projected_gravity.begin());
        // record command
        cmd.at(0) = gamepad.lx; // linear_x: [-1,1]  (lx/ly swapped on Go2 remote)
        cmd.at(1) = -gamepad.ly; // linear_y; [-1,1]
        cmd.at(2) = -gamepad.rx; // angular_z: [-1,1]
        // record robot state
        for (int i = 0; i < 12; ++i)
        {
            jpos_processed.at(i) = robot_interface.jpos.at(i) - stand_pos.at(i);
            jvel.at(i) = robot_interface.jvel.at(i);
        }
    }
    void DummyCalculate()
    {
        // warmup the neural network
        for(int i = 0; i < num_obs; ++i)
        {
            obs.at(i) = 0.0;
        }
        std::vector<torch::jit::IValue> policy_input;
        torch::Tensor policy_input_tensor = torch::zeros({1, num_obs});
        for(int i = 0; i < num_obs; ++i)
        {
            policy_input_tensor[0][i] = obs.at(i);
        }
        policy_input.push_back(policy_input_tensor);
        torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
    }
    void Calculate()
    {
        // CBF safety filter: minimally edit the velocity command so the body
        // (x,y) cannot enter the keep-out zone. Runs before the policy sees cmd.
        // CBF uses a DEAD-RECKONED position (integrate the commanded velocity by
        // IMU yaw): rt/sportmodestate odometry is dead in low-level mode on the
        // real robot. sm=(sportmodestate) is printed alongside for comparison
        // (ground truth in sim, ~zeros on hardware).
        // heading relative to where RL control started -> zone is defined in the
        // robot's start frame (2 m straight ahead of where you begin walking).
        const double heading = odom_yaw_ - yaw0_;
        if (use_mppi_)
        {
            // AUTONOMOUS MPPI-CBF. The heavy plan() runs in planLoop() (background
            // thread), so this 50Hz loop never blocks. Here we just publish the latest
            // pose, read the freshest planned command, EMA-smooth it, and re-cap forward
            // speed with the CBF output-guard (heading as motion direction). Overrides cmd.
            // Position estimate in the START frame. sportmodestate (use_sm_) is true
            // ground truth in sim -> rotate its world displacement-from-start into the
            // start frame. On hardware low-level mode it's dead, so fall back to the
            // dead-reckoned dr_ (which DRIFTS as the policy veers -> can clip far obstacles).
            double px, py;
            if (use_sm_)
            {
                double ddx = sm_x_ - sm0_x_, ddy = sm_y_ - sm0_y_;
                double c0 = std::cos(yaw0_), s0 = std::sin(yaw0_);
                px =  c0 * ddx + s0 * ddy;     // rotate world disp by -yaw0_ into start frame
                py = -s0 * ddx + c0 * ddy;
            }
            else { px = dr_x_; py = dr_y_; }

            double tgx = gx_ - px, tgy = gy_ - py;
            double dist = std::sqrt(tgx * tgx + tgy * tgy);
            double pv, pw;
            {
                std::lock_guard<std::mutex> lk(io_mtx_);
                in_x_ = px; in_y_ = py; in_th_ = heading; in_active_ = true;
                pv = out_v_; pw = out_w_;
            }
            v_filt_ = mppi_smooth_ * v_filt_ + (1.0 - mppi_smooth_) * pv;
            w_filt_ = mppi_smooth_ * w_filt_ + (1.0 - mppi_smooth_) * pw;
            double v = v_filt_, w = w_filt_;
            double cap = mppi_planner_.forwardSpeedCap(px, py, std::cos(heading), std::sin(heading));
            if (v > cap) v = cap;
            if (dist < goal_tol_) { v = 0.0; w = 0.0; }
            cmd.at(0) = static_cast<float>(v);
            cmd.at(1) = 0.0f;
            cmd.at(2) = static_cast<float>(w);
            if (++dbg_count_ % 25 == 0)
            {
                double h = 1e9;            // min barrier at the position estimate (read-only on params)
                for (const auto &o : mppi_planner_.params().obstacles)
                { double ex = px - o.cx, ey = py - o.cy; h = std::min(h, ex * ex + ey * ey - o.r_eff * o.r_eff); }
                std::cout << "[MPPI] pos=(" << px << "," << py << ")" << (use_sm_ ? "[sm]" : "[dr]")
                          << " dr=(" << dr_x_ << "," << dr_y_ << ") h=" << h
                          << " dist2goal=" << dist << " cmd=(" << cmd.at(0) << ",0," << cmd.at(2) << ")\n"
                          << std::flush;
            }
        }
        else if (nav_)
        {
            // AUTONOMOUS go-around: go-to-goal + circulation near the zone, CBF-filtered,
            // then steer (yaw + forward speed) toward the safe direction. Overrides cmd.
            const auto &P = cbf_.params();
            const double reff = P.r + P.margin;
            double tgx = gx_ - dr_x_, tgy = gy_ - dr_y_;
            double dist = std::sqrt(tgx * tgx + tgy * tgy);
            double ux = 0.0, uy = 0.0;
            if (dist > goal_tol_)
            {
                double ghx = tgx / dist, ghy = tgy / dist;          // toward goal
                double pcx = dr_x_ - P.cx, pcy = dr_y_ - P.cy;
                double dc = std::sqrt(pcx * pcx + pcy * pcy);
                double nhx = (dc > 1e-6) ? pcx / dc : 1.0;
                double nhy = (dc > 1e-6) ? pcy / dc : 0.0;          // outward radial
                double w = (band_ - (dc - reff)) / band_;          // circulation weight
                w = std::max(0.0, std::min(1.0, w));
                double thx = -nhy, thy = nhx;                       // tangent...
                if (thx * ghx + thy * ghy < 0) { thx = -thx; thy = -thy; }  // ...toward goal
                double dx = (1 - w) * ghx + w * thx, dy = (1 - w) * ghy + w * thy;
                double dn = std::sqrt(dx * dx + dy * dy);
                if (dn > 1e-9) { dx /= dn; dy /= dn; } else { dx = ghx; dy = ghy; }
                ux = vdes_ * dx; uy = vdes_ * dy;
            }
            cbf_.filterWorld(dr_x_, dr_y_, ux, uy);                 // CBF on world desired velocity (updates h)
            double desmag = std::sqrt(ux * ux + uy * uy);
            double gx2 = (desmag > 1e-6) ? ux / desmag : std::cos(heading);
            double gy2 = (desmag > 1e-6) ? uy / desmag : std::sin(heading);
            double ch = std::cos(heading), sh = std::sin(heading);
            double e = std::atan2(ch * gy2 - sh * gx2, ch * gx2 + sh * gy2);
            double omega = std::max(-wmax_, std::min(wmax_, kw_ * e));
            double s_cmd = (dist > goal_tol_) ? std::min(desmag, vdes_) * std::max(0.0, std::cos(e)) : 0.0;
            cmd.at(0) = static_cast<float>(s_cmd);
            cmd.at(1) = 0.0f;
            cmd.at(2) = static_cast<float>(omega);
            if (++dbg_count_ % 25 == 0)
                std::cout << "[NAV] dr=(" << dr_x_ << "," << dr_y_ << ") h=" << cbf_.lastH()
                          << " dist2goal=" << dist << " cmd=(" << cmd.at(0) << ",0," << cmd.at(2) << ")\n"
                          << std::flush;
        }
        else if (cbf_.params().enable)
        {
            bool act = cbf_.filter(dr_x_, dr_y_, heading, cmd.at(0), cmd.at(1));
            if (act || (++dbg_count_ % 25 == 0))
                std::cout << "[CBF] dr=(" << dr_x_ << "," << dr_y_ << ") sm=("
                          << sm_x_ << "," << sm_y_ << ") h=" << cbf_.lastH()
                          << (act ? "  <-- ACTIVE (clamping)" : "")
                          << "  cmd=(" << cmd.at(0) << "," << cmd.at(1) << ")\n"
                          << std::flush;
        }
        // Dead-reckoning: integrate the (post-CBF) commanded body velocity into the
        // start frame via relative heading. Runs every control tick (CTRL only).
        {
            double c = std::cos(heading), s = std::sin(heading);
            dr_x_ += (c * cmd.at(0) - s * cmd.at(1)) * dt;
            dr_y_ += (s * cmd.at(0) + c * cmd.at(1)) * dt;
        }
        obs.at(0) = cmd.at(0) * lin_vel_scale;
        obs.at(1) = cmd.at(1) * lin_vel_scale;
        obs.at(2) = cmd.at(2) * ang_vel_scale;
        // Fill observation
        for(int i = 0; i < 3; ++i)
        {
            obs.at(3 + i) = projected_gravity.at(i);
            obs.at(6 + i) = base_ang_vel.at(i) * ang_vel_scale;
        }
        for(int i = 0; i < 12; ++i)
        {
            obs.at(9 + i) = jpos_processed.at(i) * dof_pos_scale;
            obs.at(21 + i) = jvel.at(i) * dof_vel_scale;
            obs.at(33 + i) = actions.at(i);
        }
        // Conduct Policy Inference
        std::vector<torch::jit::IValue> policy_input;
        torch::Tensor policy_input_tensor = torch::zeros({1, num_obs});
        for(int i = 0; i < num_obs; ++i)
        {
            policy_input_tensor[0][i] = obs.at(i);
        }
        policy_input.push_back(policy_input_tensor);
        torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
        std::array<float, 12> actions_scaled;
        for(int i = 0; i < 12; ++i)
        {
            actions.at(i) = policy_output_tensor[0][i].item<float>();
            actions_scaled.at(i) = actions.at(i) * action_scale;
            jpos_des.at(i) = actions_scaled.at(i) + stand_pos.at(i);
        }
    }
    std::vector<float> GetLog()
    {
        // record input, output and other info into a vector
        std::vector<float> log;
        log.push_back(cmd.at(0));
        log.push_back(cmd.at(1));
        log.push_back(cmd.at(2));
        return log;
    }

    std::string config_file_name;
    
    // observation
    std::array<float, 3> base_ang_vel;
    std::array<float, 3> projected_gravity;
    std::array<float, 3> cmd;
    std::array<float, 12> jpos_processed;       //joint sequence: sim
    std::array<float, 12> jvel;
    std::array<float, 12> actions;
    std::vector<float> obs;                     // current observation
    int num_obs;                                // length of observation
    // normalization parameters
    float lin_vel_scale;
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;
    // control params
    float action_scale;
    // NN model
    std::string policy_name;
    torch::jit::script::Module policy;

    // CBF safety filter
    void setOdometry(double x, double y, double yaw) override
    {
        sm_x_ = x; sm_y_ = y; odom_yaw_ = yaw;
        if (capture_yaw0_) { yaw0_ = yaw; sm0_x_ = x; sm0_y_ = y; capture_yaw0_ = false; }  // pose at RL start
    }
    void resetOdometry() override
    {
        dr_x_ = 0.0; dr_y_ = 0.0; capture_yaw0_ = true;
        reset_req_.store(true);            // planLoop() resets the planner's warm-start
        std::lock_guard<std::mutex> lk(io_mtx_);
        in_active_ = false; out_v_ = 0.0; out_w_ = 0.0;
        v_filt_ = 0.0; w_filt_ = 0.0;
    }
    void getTwinState(double &x, double &y, double &yaw0) override { x = dr_x_; y = dr_y_; yaw0 = yaw0_; }

    // Background planning loop: continuously re-plan from the latest published pose so
    // the control loop only ever reads the freshest command (never blocks on plan()).
    void planLoop()
    {
        while (running_.load())
        {
            if (reset_req_.exchange(false)) mppi_planner_.reset();
            bool active; double sx, sy, sth;
            { std::lock_guard<std::mutex> lk(io_mtx_); active = in_active_; sx = in_x_; sy = in_y_; sth = in_th_; }
            if (active)
            {
                double v, w;
                mppi_planner_.plan(sx, sy, sth, gx_, gy_, v, w);   // the expensive part, OFF the ctrl loop
                std::lock_guard<std::mutex> lk(io_mtx_);
                if (in_active_) { out_v_ = v; out_w_ = w; }        // drop result if a reset intervened
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(plan_period_s_));
        }
    }
    ~SimpleRLController() { running_.store(false); if (plan_thread_.joinable()) plan_thread_.join(); }
    CBFSafetyFilter cbf_;
    bool   nav_ = false;               // autonomous go-around navigation (reactive)
    // MPPI-CBF planner (autonomous, multi-obstacle; supersedes nav_ when on).
    // plan() runs in plan_thread_ (planLoop) so the 50Hz control loop never blocks on
    // it (synchronous replan overran the 20ms tick -> jitter). The control loop publishes
    // the latest pose to in_*, reads the freshest command from out_* (io_mtx_-protected).
    bool   use_mppi_ = false;
    MPPICBFPlanner mppi_planner_;
    int    mppi_replan_every_ = 7;
    double mppi_smooth_ = 0.5, v_filt_ = 0.0, w_filt_ = 0.0;
    std::thread plan_thread_;
    std::mutex  io_mtx_;
    std::atomic<bool> running_{false};
    std::atomic<bool> reset_req_{false};
    double in_x_ = 0.0, in_y_ = 0.0, in_th_ = 0.0;   // pose published to the planner
    bool   in_active_ = false;                        // plan only while in RL control
    double out_v_ = 0.0, out_w_ = 0.0;                // latest planned command
    double plan_period_s_ = 0.14;
    double gx_ = 2.0, gy_ = 0.0, vdes_ = 0.5, band_ = 0.8, kw_ = 1.5, wmax_ = 1.0, goal_tol_ = 0.2;
    bool   use_sm_ = false;            // use sportmodestate position (ground truth) instead of dead-reckoning
    double dr_x_ = 0.0, dr_y_ = 0.0;   // dead-reckoned position (start-frame)
    double sm_x_ = 0.0, sm_y_ = 0.0;   // sportmodestate position (world frame)
    double sm0_x_ = 0.0, sm0_y_ = 0.0; // sportmodestate position captured at RL start
    double odom_yaw_ = 0.0;            // absolute IMU yaw
    double yaw0_ = 0.0;                // IMU yaw captured at RL start
    bool   capture_yaw0_ = false;      // grab yaw0_ on the next odom update after reset
    int dbg_count_ = 0;
};

/**
 * @brief Inheritance of BasicUserController, controller for Walk These Ways
 * 
 */
class WTWController : public BasicUserController
{
    public:
        WTWController(const std::string& cfg_file): BasicUserController()
         {
            // observation init to 0
            base_ang_vel.fill(0.0);
            projected_gravity.fill(0.);
            projected_gravity.at(2) = -1.0;
            cmd.fill(0.0);
            jpos_processed.fill(0.0);
            jvel.fill(0.0);
            actions.fill(0.0);
            clock_input.fill(0.0);
            theta.fill(0.0);
            gait_choice = 0; // default to the first gait
            config_file_name = cfg_file;
        }

        void loadParam()
        {
            WTWCfg cfg(config_file_name);
            dt = cfg.dt;
            stand_kp = cfg.stand_kp;
            stand_kd = cfg.stand_kd;
            action_scale = cfg.action_scale;
            lin_vel_scale = cfg.lin_vel_scale;
            ang_vel_scale = cfg.ang_vel_scale;
            dof_pos_scale = cfg.dof_pos_scale;
            dof_vel_scale = cfg.dof_vel_scale;
            policy_name = cfg.policy_name;
            ctrl_kp = cfg.ctrl_kp;
            ctrl_kd = cfg.ctrl_kd;
            frame_stack = cfg.frame_stack;
            num_single_obs = cfg.num_single_obs;
            num_gaits = cfg.num_gaits;
            theta_fl.resize(num_gaits);
            theta_fr.resize(num_gaits);
            theta_rl.resize(num_gaits);
            theta_rr.resize(num_gaits);
            for (int i = 0; i < 12; ++i)
            {
                stand_pos.at(i) = cfg.stand_pos.at(i);
                sit_pos.at(i) = cfg.sit_pos.at(i);
            }
            for (int i = 0; i < num_gaits; ++i)
            {
                theta_fl.at(i) = cfg.theta_fl.at(i);
                theta_fr.at(i) = cfg.theta_fr.at(i);
                theta_rl.at(i) = cfg.theta_rl.at(i);
                theta_rr.at(i) = cfg.theta_rr.at(i);
            }
            // Read behavior param range
            for (int i = 0; i < 2; ++i)
            {
                gait_period_range.at(i) = cfg.gait_period_range.at(i);
                base_height_target_range.at(i) = cfg.base_height_target_range.at(i);
                foot_clearance_target_range.at(i) = cfg.foot_clearance_target_range.at(i);
                pitch_target_range.at(i) = cfg.pitch_target_range.at(i);
            }
            gait_period = gait_period_range.at(1);
            base_height_target = base_height_target_range.at(0);
            foot_clearance_target = foot_clearance_target_range.at(0);
            pitch_target = pitch_target_range.at(1);
            theta.at(0) = theta_fl.at(gait_choice);
            theta.at(1) = theta_fr.at(gait_choice);
            theta.at(2) = theta_rl.at(gait_choice);
            theta.at(3) = theta_rr.at(gait_choice);
            // initialize history observation buffer
            // prepare history buffers
            single_step_obs.resize(num_single_obs);
            single_step_obs.clear();
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.push_back(0.0);
            }
            history_obs.resize(frame_stack);
            for(int i = 0; i < frame_stack; ++i)
            {
                history_obs.at(i).resize(num_single_obs);
                history_obs.at(i) = single_step_obs;
            }
        }

        /**
         * @brief 载入运动策略
         * @note  载入后缀为.pt的模型文件, 该模型文件使用torch.jit.save()保存.
         * @note  默认模型文件路径为../models/
        */
        void loadPolicy()
        {
            fs::path model_path = fs::current_path() / "../models";
            fs::path full_path = model_path / policy_name;
            std::ifstream model_stream(full_path.string(), std::ios::binary);
            policy = torch::jit::load(model_stream);
            std::cout << "Load policy from: " << full_path << std::endl;
        }

        void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad)
        {
            // save necessary data from input
            std::copy(robot_interface.gyro.begin(), robot_interface.gyro.end(), base_ang_vel.begin());
            std::copy(robot_interface.projected_gravity.begin(), robot_interface.projected_gravity.end(), projected_gravity.begin());

            // Left and Right for gait period
            if(gamepad.left.pressed)
            {
                gait_period += 0.01;
                gait_period = std::min(gait_period, gait_period_range.at(1));
            }
            else if(gamepad.right.pressed)
            {
                gait_period -= 0.01;
                gait_period = std::max(gait_period, gait_period_range.at(0));
            }
            // Up and Down for base height target
            if(gamepad.up.pressed)
            {
                base_height_target += 0.01;
                base_height_target = std::min(base_height_target, base_height_target_range.at(1));
            }
            else if(gamepad.down.pressed)
            {
                base_height_target -= 0.01;
                base_height_target = std::max(base_height_target, base_height_target_range.at(0));
            }
            // A and B for foot clearance target
            if(gamepad.A.pressed)
            {
                foot_clearance_target += 0.01;
                foot_clearance_target = std::min(foot_clearance_target, foot_clearance_target_range.at(1));
            }
            else if(gamepad.B.pressed)
            {
                foot_clearance_target -= 0.01;
                foot_clearance_target = std::max(foot_clearance_target, foot_clearance_target_range.at(0));
            }
            // X and Y for pitch target
            if(gamepad.X.pressed)
            {
                pitch_target += 0.01;
                pitch_target = std::min(pitch_target, pitch_target_range.at(1));
            }
            else if(gamepad.Y.pressed)
            {
                pitch_target -= 0.01;
                pitch_target = std::max(pitch_target, pitch_target_range.at(0));
            }
            // R1 and R2 for gait choice
            if(gamepad.R1.on_press)
            {
                gait_choice = (gait_choice + 1) % num_gaits;
            }
            else if(gamepad.R2.on_press)
            {
                gait_choice = (gait_choice - 1 + num_gaits) % num_gaits;
            }
            // record command
            cmd.at(0) = gamepad.ly; // linear_x: [-1,1]
            cmd.at(1) = -gamepad.lx; // linear_y; [-1,1]
            cmd.at(2) = -gamepad.rx; // angular_z: [-1,1]

            // record robot state
            for (int i = 0; i < 12; ++i)
            {
                jpos_processed.at(i) = robot_interface.jpos.at(i) - stand_pos.at(i);
                jvel.at(i) = robot_interface.jvel.at(i);
            }
        }

        void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad)
        {
            gait_time = 0.0;
            phi = 0.0;
            gait_choice = 0; 
            gait_period = gait_period_range.at(1);
            base_height_target = base_height_target_range.at(1);
            foot_clearance_target = foot_clearance_target_range.at(1);
            pitch_target = pitch_target_range.at(1);
            theta.at(0) = theta_fl.at(gait_choice);
            theta.at(1) = theta_fr.at(gait_choice);
            theta.at(2) = theta_rl.at(gait_choice);
            theta.at(3) = theta_rr.at(gait_choice);
            // reset history observation buffer
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.at(i) = 0.0;
            }
            for(int i = 0; i < frame_stack; ++i)
            {
                history_obs.at(i).resize(num_single_obs);
                history_obs.at(i) = single_step_obs;
            }
        }

        void DummyCalculate()
        {
            // warmup the neural network
            for(int i = 0; i < num_single_obs; ++i)
            {
                single_step_obs.at(i) = 0.0;
            }
            history_obs.pop_front();
            history_obs.push_back(single_step_obs);
            std::vector<torch::jit::IValue> policy_input;
            torch::Tensor policy_input_tensor = torch::zeros({1, frame_stack*num_single_obs});
            for(int i = 0; i < frame_stack; ++i)
            {
                for(int j = 0; j < num_single_obs; ++j)
                {
                    policy_input_tensor[0][i*num_single_obs + j] = history_obs.at(i).at(j);
                }
            }
            policy_input.push_back(policy_input_tensor);
            torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
        }

        void Calculate()
        {
            pre_process();
            // put current observation into history buffer
            fill_single_step_obs();
            history_obs.pop_front();
            history_obs.push_back(single_step_obs);
            std::vector<torch::jit::IValue> policy_input;
            torch::Tensor policy_input_tensor = torch::zeros({1, frame_stack*num_single_obs});
            for(int i = 0; i < frame_stack; ++i)
            {
                for(int j = 0; j < num_single_obs; ++j)
                {
                    policy_input_tensor[0][i*num_single_obs + j] = history_obs.at(i).at(j);
                }
            }
            policy_input.push_back(policy_input_tensor);
            torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
            /***** 计算期望位置 *****/
            std::array<float, 12> actions_scaled;
            for(int i = 0; i < 12; ++i)
            {
                actions.at(i) = policy_output_tensor[0][i].item<float>();
                actions_scaled.at(i) = actions.at(i) * action_scale;
                jpos_des.at(i) = stand_pos.at(i) + actions_scaled.at(i);
            }
        }

        std::vector<float> GetLog()
        {
            // record input, output and other info into a vector
            std::vector<float> log;
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(cmd.at(i));
            }
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(projected_gravity.at(i));
            }
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(base_ang_vel.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jvel.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(actions.at(i));
            }
            for (int i = 0; i < 4; i++)
            {
                log.push_back(clock_input.at(i));
            }
            log.push_back(gait_period);
            log.push_back(base_height_target);
            log.push_back(foot_clearance_target);
            log.push_back(pitch_target);
            for (int i = 0; i < 4; i++)
            {
                log.push_back(theta.at(i));
            }
            return log;
        }
        
        std::string config_file_name;
    
        // observation
        std::array<float, 3> base_ang_vel;
        std::array<float, 3> projected_gravity;
        std::array<float, 3> cmd;
        std::array<float, 12> jpos_processed;       //joint sequence: sim
        std::array<float, 12> jvel;
        std::array<float, 12> actions;
        std::array<float, 8> clock_input;
        float gait_period;
        float base_height_target;
        float foot_clearance_target;
        float pitch_target;
        std::array<float, 4> theta; // theta_fl, theta_fr, theta_rl, theta_rr
        int frame_stack;
        int num_single_obs; // length of single step observation
        std::vector<float> single_step_obs; // 用于记录单步观测数据
        std::deque<std::vector<float>> history_obs; // 用于记录历史观测数据

        // normalization parameters
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_pos_scale;
        float dof_vel_scale;
        // control params
        float action_scale;
        // gait
        int num_gaits;
        int gait_choice;
        std::array<float, 2> gait_period_range;
        std::array<float, 2> base_height_target_range;
        std::array<float, 2> foot_clearance_target_range;
        std::array<float, 2> pitch_target_range;
        std::vector<float> theta_fl;
        std::vector<float> theta_fr;
        std::vector<float> theta_rl;
        std::vector<float> theta_rr;
        float gait_time;
        float phi;
        // NN model
        std::string policy_name;
        torch::jit::script::Module policy;

    private:
        
        void calc_periodic_obs()
        {
            for (int i = 0; i < 4; i++)
            {
                clock_input.at(i) = sin(2 * M_PI * (phi + theta.at(i)));
                clock_input.at(i + 4) = cos(2 * M_PI * (phi + theta.at(i)));
            }
        }

        void pre_process()
        {
            // gait time step
            gait_time += dt;
            if(gait_time > (gait_period - dt/2))
            {
                gait_time = 0.0;
            }
            phi = gait_time / gait_period;
            // choose gait
            theta.at(0) = theta_fl.at(gait_choice);
            theta.at(1) = theta_fr.at(gait_choice);
            theta.at(2) = theta_rl.at(gait_choice);
            theta.at(3) = theta_rr.at(gait_choice);
            // calculate clock input
            calc_periodic_obs();
        }

        void fill_single_step_obs()
        {
            single_step_obs.at(0) = cmd.at(0) * lin_vel_scale;
            single_step_obs.at(1) = cmd.at(1) * lin_vel_scale;
            single_step_obs.at(2) = cmd.at(2) * ang_vel_scale;
            for(int i = 0; i < 3; ++i)
            {
                single_step_obs.at(i+3) = projected_gravity.at(i);
                single_step_obs.at(i+6) = base_ang_vel.at(i) * ang_vel_scale;
            }
            for(int i = 0; i < 12; ++i)
            {
                single_step_obs.at(i+9) = jpos_processed.at(i) * dof_pos_scale;
                single_step_obs.at(i+21) = jvel.at(i) * dof_vel_scale;
                single_step_obs.at(i+33) = actions.at(i);
            }
            for(int i = 0; i < 4; ++i)
            {
                single_step_obs.at(i+45) = clock_input.at(i);
                single_step_obs.at(i+49) = clock_input.at(i+4);
                single_step_obs.at(i+57) = theta.at(i);
            }
            single_step_obs.at(53) = gait_period;
            single_step_obs.at(54) = base_height_target;
            single_step_obs.at(55) = foot_clearance_target; 
            single_step_obs.at(56) = pitch_target;
        }
};

class TSController : public BasicUserController
{
public:
    TSController(const std::string& cfg_file)
    {
        // observation init to 0
        base_ang_vel.fill(0.0);
        projected_gravity.fill(0.);
        projected_gravity.at(2) = -1.0;
        cmd.fill(0.0);
        jpos_processed.fill(0.0);
        jvel.fill(0.0);
        actions.fill(0.0);
        config_file_name = cfg_file;
        // External velocity-command bridge (camera CBF -> np3o). Subscribe once;
        // ChannelFactory is already Init'd in main() before this ctor runs. Harmless
        // if nobody publishes (we only override the command when a fresh msg arrives).
        ext_cmd_sub_ = unitree::robot::ChannelSubscriberPtr<geometry_msgs::msg::dds_::Twist_>(
            new unitree::robot::ChannelSubscriber<geometry_msgs::msg::dds_::Twist_>("rt/cbf/cmd_vel"));
        ext_cmd_sub_->InitChannel(
            std::bind(&TSController::onExtCmd, this, std::placeholders::_1), 1);
    }
    // Latest external command handler + freshness clock (thread-safe via atomics).
    static double now_sec_()
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void onExtCmd(const void* msg)
    {
        const auto* m = static_cast<const geometry_msgs::msg::dds_::Twist_*>(msg);
        ext_vx_.store(m->linear().x());
        ext_vy_.store(m->linear().y());
        ext_wz_.store(m->angular().z());
        ext_cmd_time_.store(now_sec_());
    }
    void loadParam()
    {
        TSCfg cfg(config_file_name);
        use_genesis = cfg.use_genesis;
        dt = cfg.dt;
        stand_kp = cfg.stand_kp;
        stand_kd = cfg.stand_kd;
        action_scale = cfg.action_scale;
        lin_vel_scale = cfg.lin_vel_scale;
        ang_vel_scale = cfg.ang_vel_scale;
        dof_pos_scale = cfg.dof_pos_scale;
        dof_vel_scale = cfg.dof_vel_scale;
        policy_name = cfg.policy_name;
        ctrl_kp = cfg.ctrl_kp;
        ctrl_kd = cfg.ctrl_kd;
        frame_stack = cfg.frame_stack;
        num_single_obs = cfg.num_single_obs;
        hip_scale = cfg.hip_scale;
        action_filter = cfg.action_filter;
        a_filt.fill(0.0f);
        for (int i = 0; i < 12; ++i)
        {
            stand_pos.at(i) = cfg.stand_pos.at(i);
            sit_pos.at(i) = cfg.sit_pos.at(i);
        }
        // initialize history observation buffer
        // prepare history buffers
        single_step_obs.resize(num_single_obs);
        single_step_obs.clear();
        last_obs.resize(num_single_obs);
        last_obs.clear();
        for(int i = 0; i < num_single_obs; ++i)
        {
            single_step_obs.push_back(0.0);
            last_obs.push_back(0.0);
        }
        history_obs.resize(frame_stack);
        for(int i = 0; i < frame_stack; ++i)
        {
            history_obs.at(i).resize(num_single_obs);
            history_obs.at(i) = single_step_obs;
        }
    }
    void loadPolicy()
    {
        fs::path model_path = fs::current_path() / "../models";
        policy = torch::jit::load(model_path / policy_name);
        std::cout << "Load policy from: " << model_path / policy_name << std::endl;
    }
    void reset(BasicRobotInterface &robot_interface, Gamepad &gamepad)
    {
        a_filt.fill(0.0f);
        // reset history observation buffer
        for(int i = 0; i < num_single_obs; ++i)
        {
            single_step_obs.at(i) = 0.0;
            last_obs.at(i) = 0.0;
        }
        for(int i = 0; i < frame_stack; ++i)
        {
            history_obs.at(i).resize(num_single_obs);
            history_obs.at(i) = last_obs;
        }
    }
    void GetInput(BasicRobotInterface &robot_interface, Gamepad &gamepad)
    {
        // save necessary data from input
        std::copy(robot_interface.gyro.begin(), robot_interface.gyro.end(), base_ang_vel.begin());
        std::copy(robot_interface.projected_gravity.begin(), robot_interface.projected_gravity.end(), projected_gravity.begin());
        // record command  (lx/ly swapped on the Go2 remote -> match SimpleRLController)
        cmd.at(0) = gamepad.lx; // linear_x: [-1,1]
        cmd.at(1) = -gamepad.ly; // linear_y; [-1,1]
        cmd.at(2) = -gamepad.rx; // angular_z: [-1,1]
        // External command bridge: a FRESH Twist on rt/cbf/cmd_vel (from the camera
        // CBF, in m/s and rad/s) overrides the gamepad STICKS. Gamepad BUTTONS -- incl.
        // the damping e-stop -- are handled elsewhere and are untouched. If the command
        // goes stale (publisher stopped/crashed) we fall back to the sticks, which at
        // rest command zero -> the robot stops. This is the software watchdog.
        if (now_sec_() - ext_cmd_time_.load() < ext_cmd_timeout_)
        {
            cmd.at(0) = static_cast<float>(ext_vx_.load());
            cmd.at(1) = static_cast<float>(ext_vy_.load());
            cmd.at(2) = static_cast<float>(ext_wz_.load());
        }
        // record robot state
        if(use_genesis)
        {
            for (int i = 0; i < 12; ++i) // same index
            {
                jpos_processed.at(i) = robot_interface.jpos.at(i) - stand_pos.at(i);
                jvel.at(i) = robot_interface.jvel.at(i);
            }
        }
        else
        {
            for(int i = 0; i < 12; ++i) // real2sim index mapping
            {
                jpos_processed.at(r2s_index_map[i]) = robot_interface.jpos.at(i) - stand_pos.at(i);
                jvel.at(r2s_index_map[i]) = robot_interface.jvel.at(i);
            }
        }
        
    }
    void DummyCalculate()
    {
        // warmup the neural network
        for(int i = 0; i < num_single_obs; ++i)
        {
            last_obs.at(i) = 0.0;
            single_step_obs.at(i) = 0.0;
        }
        history_obs.pop_front();
        history_obs.push_back(last_obs);
        torch::Tensor obs_tensor = torch::zeros({1, num_single_obs});
        for(int i = 0; i < num_single_obs; i++)
            obs_tensor[0][i] = single_step_obs.at(i);
        torch::Tensor obs_his_tensor = torch::zeros({1, frame_stack*num_single_obs});
        for(int i = 0; i < frame_stack; ++i)
        {
            for(int j = 0; j < num_single_obs; ++j)
            {
                obs_his_tensor[0][i*num_single_obs + j] = history_obs.at(i).at(j);
            }
        }
        std::vector<torch::jit::IValue> policy_input;
        policy_input.push_back(obs_tensor);
        policy_input.push_back(obs_his_tensor);
        torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
    }
    void Calculate()
    {
        // Fill observation
        fill_single_step_obs();
        // put current observation into history buffer
        history_obs.pop_front();
        history_obs.push_back(last_obs);
        torch::Tensor obs_tensor = torch::zeros({1, num_single_obs});
        for(int i = 0; i < num_single_obs; i++)
            obs_tensor[0][i] = single_step_obs.at(i);
        torch::Tensor obs_his_tensor = torch::zeros({1, frame_stack*num_single_obs});
        for(int i = 0; i < frame_stack; ++i)
        {
            for(int j = 0; j < num_single_obs; ++j)
            {
                obs_his_tensor[0][i*num_single_obs + j] = history_obs.at(i).at(j);
            }
        }
        std::vector<torch::jit::IValue> policy_input;
        policy_input.push_back(obs_tensor);
        policy_input.push_back(obs_his_tensor);
        torch::Tensor policy_output_tensor = policy.forward(policy_input).toTensor();
        std::array<float, 12> actions_scaled;
        if(use_genesis)
        {
            for(int i = 0; i < 12; ++i)
            {
                // raw action feeds back into the next observation (NP3O uses the
                // unfiltered action in obs); filter + hip_scale only shape the target.
                actions.at(i) = policy_output_tensor[0][i].item<float>();
                a_filt.at(i) = action_filter * a_filt.at(i) + (1.0f - action_filter) * actions.at(i);
                actions_scaled.at(i) = a_filt.at(i) * action_scale;
                if (i % 3 == 0) actions_scaled.at(i) *= hip_scale;   // hip joints: idx 0,3,6,9
                jpos_des.at(i) = actions_scaled.at(i) + stand_pos.at(i);
            }
        }
        else
        {
            for(int i = 0; i < 12; ++i)
            {
                actions.at(i) = policy_output_tensor[0][i].item<float>();
                actions_scaled.at(i) = actions.at(i) * action_scale;
            }
            for(int i = 0; i < 12; ++i)
            { 
                jpos_des.at(i) = actions_scaled.at(r2s_index_map[i]) + stand_pos.at(i);
            }
        }
    }
    std::vector<float> GetLog()
    {
        // record input, output and other info into a vector
        std::vector<float> log;
        log.push_back(cmd.at(0));
        log.push_back(cmd.at(1));
        log.push_back(cmd.at(2));
        return log;
    }

    // real2sim joint index mapping, for policies trained in IsaacGym
    // real: 0-FR_hip, 1-FR_thigh, 2-FR_calf,
    //       3-FL_hip, 4-FL_thigh, 5-FL_calf,
    //       6-RR_hip, 7-RR_thigh, 8-RR_calf,
    //       9-RL_hip, 10-RL_thigh, 11-RL_calf
    // sim:  0-FL_hip, 1-FL_thigh, 2-FL_calf,
    //       3-FR_hip, 4-FR_thigh, 5-FR_calf,
    //       6-RL_hip, 7-RL_thigh, 8-RL_calf,
    //       9-RR_hip, 10-RR_thigh, 11-RR_calf
    uint16_t r2s_index_map[12] = {3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8};

    std::string config_file_name;
    bool use_genesis;
    
    // observation
    std::array<float, 3> base_ang_vel;
    std::array<float, 3> projected_gravity;
    std::array<float, 3> cmd;
    // external velocity-command bridge (camera CBF -> np3o); wired in ctor + GetInput
    unitree::robot::ChannelSubscriberPtr<geometry_msgs::msg::dds_::Twist_> ext_cmd_sub_;
    std::atomic<double> ext_vx_{0.0}, ext_vy_{0.0}, ext_wz_{0.0};
    std::atomic<double> ext_cmd_time_{-1e9};
    double ext_cmd_timeout_ = 0.3;   // s: stale command -> ignore (robot stops)
    std::array<float, 12> jpos_processed;       //joint sequence: sim
    std::array<float, 12> jvel;
    std::array<float, 12> actions;
    int frame_stack;
    int num_single_obs; // length of single step observation
    std::vector<float> single_step_obs; // 用于记录单步观测数据
    std::vector<float> last_obs;        // 用于记录上一步的观测数据
    std::deque<std::vector<float>> history_obs; // 用于记录历史观测数据

    // normalization parameters
    float lin_vel_scale;
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;
    // control params
    float action_scale;
    float hip_scale = 1.0f;         // NP3O hip_scale_reduction (1.0 = off)
    float action_filter = 0.0f;     // low-pass alpha (0 = off)
    std::array<float, 12> a_filt{}; // filtered action state
    // NN model
    std::string policy_name;
    torch::jit::script::Module policy;

private:
    void fill_single_step_obs()
    {
        // save last observation
        std::copy(single_step_obs.begin(), single_step_obs.end(), last_obs.begin());
        // update current observation
        single_step_obs.at(0) = cmd.at(0) * lin_vel_scale;
        single_step_obs.at(1) = cmd.at(1) * lin_vel_scale;
        single_step_obs.at(2) = cmd.at(2) * ang_vel_scale;
        for(int i = 0; i < 3; ++i)
        {
            single_step_obs.at(i+3) = projected_gravity.at(i);
            single_step_obs.at(i+6) = base_ang_vel.at(i) * ang_vel_scale;
        }
        for(int i = 0; i < 12; ++i)
        {
            single_step_obs.at(i+9) = jpos_processed.at(i) * dof_pos_scale;
            single_step_obs.at(i+21) = jvel.at(i) * dof_vel_scale;
            single_step_obs.at(i+33) = actions.at(i);
        }
    }
};

} // namespace unitree::common
