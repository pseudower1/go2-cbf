#pragma once

#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

namespace unitree::common
{
    class BaseCfg
    {
    public:
        BaseCfg(const std::string &filename){};

    };

    class SimpleRLCfg : public BaseCfg
    {
    public:
        SimpleRLCfg(const std::string &filename) : BaseCfg(filename)
        {
            std::ifstream _fin(filename);
            auto cfg = YAML::Load(_fin);
            try
            {
                policy_name = cfg["policy_name"].as<std::string>();
                dt = cfg["dt"].as<float>();
                stand_kp = cfg["stand_kp"].as<float>();
                stand_kd = cfg["stand_kd"].as<float>();
                ctrl_kp = cfg["ctrl_kp"].as<float>();
                ctrl_kd = cfg["ctrl_kd"].as<float>();
                action_scale = cfg["action_scale"].as<float>();
                lin_vel_scale = cfg["lin_vel_scale"].as<float>();
                ang_vel_scale = cfg["ang_vel_scale"].as<float>();
                dof_pos_scale = cfg["dof_pos_scale"].as<float>();
                dof_vel_scale = cfg["dof_vel_scale"].as<float>();
                num_obs = cfg["num_obs"].as<int>();
                for(const auto& v : cfg["stand_pos"])
                {
                    stand_pos.push_back(v.as<float>());
                }
                for(const auto& v : cfg["sit_pos"])
                {
                    sit_pos.push_back(v.as<float>());
                }
                // CBF safety filter (optional block; defaults keep it disabled)
                if (cfg["cbf_enable"])  cbf_enable = cfg["cbf_enable"].as<bool>();
                if (cfg["cbf_cx"])      cbf_cx     = cfg["cbf_cx"].as<double>();
                if (cfg["cbf_cy"])      cbf_cy     = cfg["cbf_cy"].as<double>();
                if (cfg["cbf_r"])       cbf_r      = cfg["cbf_r"].as<double>();
                if (cfg["cbf_margin"])  cbf_margin = cfg["cbf_margin"].as<double>();
                if (cfg["cbf_gamma"])   cbf_gamma  = cfg["cbf_gamma"].as<double>();
                // autonomous go-around navigation (optional)
                if (cfg["cbf_nav"])     cbf_nav    = cfg["cbf_nav"].as<bool>();
                if (cfg["cbf_goal_x"])  cbf_goal_x = cfg["cbf_goal_x"].as<double>();
                if (cfg["cbf_goal_y"])  cbf_goal_y = cfg["cbf_goal_y"].as<double>();
                if (cfg["cbf_vdes"])    cbf_vdes   = cfg["cbf_vdes"].as<double>();
                if (cfg["cbf_band"])    cbf_band   = cfg["cbf_band"].as<double>();
                if (cfg["cbf_kw"])      cbf_kw     = cfg["cbf_kw"].as<double>();
                if (cfg["cbf_wmax"])    cbf_wmax   = cfg["cbf_wmax"].as<double>();
                if (cfg["cbf_goal_tol"]) cbf_goal_tol = cfg["cbf_goal_tol"].as<double>();
                // MPPI-CBF planner (optional; supersedes the reactive nav when on).
                // cbf_obstacles is a list of [x,y,r] (drawn radius; margin added later).
                if (cfg["cbf_mppi"]) cbf_mppi = cfg["cbf_mppi"].as<bool>();
                if (cfg["cbf_use_sm"]) cbf_use_sm = cfg["cbf_use_sm"].as<bool>();
                if (cfg["cbf_obstacles"])
                    for (const auto& o : cfg["cbf_obstacles"])
                        cbf_obstacles.push_back({o[0].as<double>(), o[1].as<double>(), o[2].as<double>()});
                if (cfg["mppi_samples"])      mppi_samples      = cfg["mppi_samples"].as<int>();
                if (cfg["mppi_horizon"])      mppi_horizon      = cfg["mppi_horizon"].as<int>();
                if (cfg["mppi_dt"])           mppi_dt           = cfg["mppi_dt"].as<double>();
                if (cfg["mppi_lambda"])       mppi_lambda       = cfg["mppi_lambda"].as<double>();
                if (cfg["mppi_sigma_v"])      mppi_sigma_v      = cfg["mppi_sigma_v"].as<double>();
                if (cfg["mppi_sigma_w"])      mppi_sigma_w      = cfg["mppi_sigma_w"].as<double>();
                if (cfg["mppi_deadzone"])     mppi_deadzone     = cfg["mppi_deadzone"].as<double>();
                if (cfg["mppi_w_near"])       mppi_w_near       = cfg["mppi_w_near"].as<double>();
                if (cfg["mppi_band"])         mppi_band         = cfg["mppi_band"].as<double>();
                if (cfg["mppi_w_head"])       mppi_w_head       = cfg["mppi_w_head"].as<double>();
                if (cfg["mppi_w_smooth"])     mppi_w_smooth     = cfg["mppi_w_smooth"].as<double>();
                if (cfg["mppi_smooth"])       mppi_smooth       = cfg["mppi_smooth"].as<double>();
                if (cfg["mppi_replan_every"]) mppi_replan_every = cfg["mppi_replan_every"].as<int>();
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }
        }
        
        float stand_kp;
        float stand_kd;
        float ctrl_kp;
        float ctrl_kd;
        float dt;
        float action_scale;
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_pos_scale;
        float dof_vel_scale;
        int num_obs;
        std::string policy_name;
        std::vector<float> stand_pos;
        std::vector<float> sit_pos;
        // CBF safety filter params (defaults: disabled)
        bool   cbf_enable = false;
        double cbf_cx = 0.0;
        double cbf_cy = 0.0;
        double cbf_r = 0.5;
        double cbf_margin = 0.35;
        double cbf_gamma = 2.0;
        // autonomous go-around navigation params (defaults: disabled)
        bool   cbf_nav = false;
        double cbf_goal_x = 2.0;
        double cbf_goal_y = 0.0;
        double cbf_vdes = 0.5;
        double cbf_band = 0.8;     // circulation influence band [m] outside enforced boundary
        double cbf_kw = 1.5;       // yaw steering gain
        double cbf_wmax = 1.0;     // max yaw rate
        double cbf_goal_tol = 0.2;
        // MPPI-CBF planner params (defaults: disabled; match the validated sim tuning)
        bool   cbf_mppi = false;
        bool   cbf_use_sm = false;   // use rt/sportmodestate position (true ground truth in
                                     // sim) instead of dead-reckoning; keep false on hardware
                                     // low-level mode where sportmodestate is dead.
        std::vector<std::array<double, 3>> cbf_obstacles;   // (x,y,r) drawn; margin added at build
        int    mppi_samples = 384;
        int    mppi_horizon = 45;
        double mppi_dt = 0.15;
        double mppi_lambda = 8.0;
        double mppi_sigma_v = 0.3;
        double mppi_sigma_w = 0.6;
        double mppi_deadzone = 0.5;
        double mppi_w_near = 10.0;
        double mppi_band = 0.3;
        double mppi_w_head = 1.0;
        double mppi_w_smooth = 3.0;
        double mppi_smooth = 0.5;    // EMA output smoothing
        int    mppi_replan_every = 7;

    };

    class WTWCfg : public BaseCfg
    {
    public:
        WTWCfg(const std::string &filename) : BaseCfg(filename)
        {
            std::ifstream _fin(filename);
            auto cfg = YAML::Load(_fin);
            try
            {
                policy_name = cfg["policy_name"].as<std::string>();
                use_genesis = cfg["use_genesis"].as<bool>();
                dt = cfg["dt"].as<float>();
                stand_kp = cfg["stand_kp"].as<float>();
                stand_kd = cfg["stand_kd"].as<float>();
                ctrl_kp = cfg["ctrl_kp"].as<float>();
                ctrl_kd = cfg["ctrl_kd"].as<float>();
                action_scale = cfg["action_scale"].as<float>();
                lin_vel_scale = cfg["lin_vel_scale"].as<float>();
                ang_vel_scale = cfg["ang_vel_scale"].as<float>();
                dof_pos_scale = cfg["dof_pos_scale"].as<float>();
                dof_vel_scale = cfg["dof_vel_scale"].as<float>();
                frame_stack = cfg["frame_stack"].as<int>();
                num_single_obs = cfg["num_single_obs"].as<int>();
                num_gaits = cfg["num_gaits"].as<int>();
                for(const auto& v : cfg["gait_period_range"])
                {
                    gait_period_range.push_back(v.as<float>());
                }
                for(const auto& v : cfg["base_height_target_range"])
                {
                    base_height_target_range.push_back(v.as<float>());
                }
                for(const auto& v : cfg["foot_clearance_target_range"])
                {
                    foot_clearance_target_range.push_back(v.as<float>());
                }
                for(const auto& v : cfg["pitch_target_range"])
                {
                    pitch_target_range.push_back(v.as<float>());
                }
                for(const auto& v : cfg["theta_fl"])
                {
                    theta_fl.push_back(v.as<float>());
                }
                for(const auto& v : cfg["theta_fr"])
                {
                    theta_fr.push_back(v.as<float>());
                }
                for(const auto& v : cfg["theta_rl"])
                {
                    theta_rl.push_back(v.as<float>());
                }
                for(const auto& v : cfg["theta_rr"])
                {
                    theta_rr.push_back(v.as<float>());
                }
                for(const auto& v : cfg["stand_pos"])
                {
                    stand_pos.push_back(v.as<float>());
                }
                for(const auto& v : cfg["sit_pos"])
                {
                    sit_pos.push_back(v.as<float>());
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }
        }

        bool use_genesis;
        float ctrl_kp;
        float ctrl_kd;
        float stand_kp;
        float stand_kd;
        float dt;
        float action_scale;
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_pos_scale;
        float dof_vel_scale;
        int frame_stack;
        int num_single_obs;
        std::string policy_name;
        std::vector<float> stand_pos;
        std::vector<float> sit_pos;
        // gait parameters
        int num_gaits;
        std::vector<float> gait_period_range;
        std::vector<float> base_height_target_range;
        std::vector<float> foot_clearance_target_range;
        std::vector<float> pitch_target_range;
        std::vector<float> theta_fl;
        std::vector<float> theta_fr;
        std::vector<float> theta_rl;
        std::vector<float> theta_rr;
    };

    class TSCfg : public BaseCfg
    {
    public:
        TSCfg(const std::string &filename) : BaseCfg(filename)
        {
            std::ifstream _fin(filename);
            auto cfg = YAML::Load(_fin);
            try
            {
                policy_name = cfg["policy_name"].as<std::string>();
                use_genesis = cfg["use_genesis"].as<bool>();
                dt = cfg["dt"].as<float>();
                stand_kp = cfg["stand_kp"].as<float>();
                stand_kd = cfg["stand_kd"].as<float>();
                ctrl_kp = cfg["ctrl_kp"].as<float>();
                ctrl_kd = cfg["ctrl_kd"].as<float>();
                action_scale = cfg["action_scale"].as<float>();
                lin_vel_scale = cfg["lin_vel_scale"].as<float>();
                ang_vel_scale = cfg["ang_vel_scale"].as<float>();
                dof_pos_scale = cfg["dof_pos_scale"].as<float>();
                dof_vel_scale = cfg["dof_vel_scale"].as<float>();
                frame_stack = cfg["frame_stack"].as<int>();
                num_single_obs = cfg["num_single_obs"].as<int>();
                // optional NP3O-style control extras (default = no-op for legacy TS policies)
                hip_scale = cfg["hip_scale"] ? cfg["hip_scale"].as<float>() : 1.0f;
                action_filter = cfg["action_filter"] ? cfg["action_filter"].as<float>() : 0.0f;
                for(const auto& v : cfg["stand_pos"])
                {
                    stand_pos.push_back(v.as<float>());
                }
                for(const auto& v : cfg["sit_pos"])
                {
                    sit_pos.push_back(v.as<float>());
                }
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                exit(EXIT_FAILURE);
            }
        }

        bool use_genesis;
        float ctrl_kp;
        float ctrl_kd;
        float stand_kp;
        float stand_kd;
        float dt;
        float action_scale;
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_pos_scale;
        float dof_vel_scale;
        int frame_stack;
        int num_single_obs;
        float hip_scale;        // NP3O hip_scale_reduction (1.0 = off)
        float action_filter;    // low-pass alpha: a_f = alpha*a_prev + (1-alpha)*a (0 = off)
        std::string policy_name;
        std::string encoder_name;
        std::vector<float> stand_pos;
        std::vector<float> sit_pos;
    };
}