#pragma once

#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <filesystem>
#include <mutex>
#include <fstream>
#include <future>
#include <algorithm>
#include <sstream>
#include <atomic>
#include <termios.h>
#include <unistd.h>

#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/idl/go2/SportModeState_.hpp"
#include "unitree/common/thread/thread.hpp"

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/common/time/time_tool.hpp"
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>

#include "state_machine.hpp"
#include "gamepad.hpp"
#include "robot_interface.hpp"
#include "user_controller.hpp"

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree::robot::go2;
namespace fs = std::filesystem;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"
#define TOPIC_HIGHSTATE "rt/sportmodestate"
#define TOPIC_TWIN "rt/cbf_twin"     // dead-reckoned (x,y,yaw0) for the digital twin

// RobotController为一个框架, 将有限状态机整体的逻辑具象化为一个类
class RobotController
{
public:
    RobotController(BasicUserController* ctrl) : ctrl(ctrl) {}

    RobotController(fs::path &log_file_name, BasicUserController* ctrl) : ctrl(ctrl)
    {
        // set log file
        log_file = std::ofstream(log_file_name);
        std::stringstream header;
        header << "state,";
        header << "cmd_lin_x,cmd_lin_y,cmd_ang_z,";
        header << "projected_g0,projected_g1,projected_g2,";
        header << "base_ang_vel_x,base_ang_vel_y,base_ang_vel_z,";
        header << "jpos0,jpos1,jpos2,jpos3,jpos4,jpos5,jpos6,jpos7,jpos8,jpos9,jpos10,jpos11,";
        header << "jvel0,jvel1,jvel2,jvel3,jvel4,jvel5,jvel6,jvel7,jvel8,jvel9,jvel10,jvel11,";
        header << "action0,action1,action2,action3,action4,action5,action6,action7,action8,action9,action10,action11,";
        header << "clock0,clock1,clock2,clock3,";
        header << "gait_period,base_height_target,foot_clearance_target,pitch_target,";
        header << "theta0,theta1,theta2,theta3";
        header << std::endl;
        log_file << header.str();
    }

    void loadParam()
    {
        ctrl->loadParam();
    }

    void loadPolicy()
    {
        ctrl->loadPolicy();
    }

    void initRobotStateClient()
    {
        rsc.SetTimeout(10.0f); 
        rsc.Init();
    }

    void activateService(const std::string& serviceName, int activate, int& serviceStatus)
    {
        rsc.ServiceSwitch(serviceName, activate, serviceStatus);
    }

    void InitDdsModel()
    {
        // init dds
        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));

        lowcmd_publisher->InitChannel();
        lowstate_subscriber->InitChannel(std::bind(&RobotController::LowStateMessageHandler, this, std::placeholders::_1), 1);

        // odometry (position/velocity) for the CBF safety filter; published by the
        // sim bridge and by the Go2 onboard estimator on rt/sportmodestate.
        highstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        highstate_subscriber->InitChannel(std::bind(&RobotController::HighStateMessageHandler, this, std::placeholders::_1), 1);

        // publish dead-reckoned pose for the digital-twin viewer
        twin_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::SportModeState_>(TOPIC_TWIN));
        twin_publisher->InitChannel();
    }

    void StartControl()
    {
        std::chrono::milliseconds duration(100);
        init_low_cmd();

        // Single raw-mode keyboard thread — handles startup Enter AND runtime i/s/c/d.
        // Never uses std::cin to avoid iostream-buffer vs read() conflicts on STDIN_FILENO.
        std::atomic<bool> keyboard_start{false};
        std::thread kb_thread([this, &keyboard_start](){
            // No-op when stdin is not a terminal (e.g. systemd service on Jetson)
            if (!isatty(STDIN_FILENO)) {
                return;
            }

            struct termios oldt, newt;
            tcgetattr(STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &= ~(ICANON | ECHO);
            newt.c_cc[VMIN] = 1;
            newt.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &newt);

            // Phase 1: wait for Enter to start
            char ch;
            while (read(STDIN_FILENO, &ch, 1) == 1) {
                if (ch == '\n' || ch == '\r') {
                    keyboard_start.store(true);
                    break;
                }
            }

            // Runtime key handling. WASD drive (toggle: press once = on at fixed
            // speed, press again = off). State machine on number keys.
            //   1/2/3/4 = sit/stand/ctrl/damp
            //   w/s = forward/back;  a/d = turn left/right;  space = stop all
            const float kFwdSpeed = 0.5f;   // forward/back speed when toggled on
            const float kYawRate  = 0.5f;   // turn rate when toggled on
            while (true) {
                if (read(STDIN_FILENO, &ch, 1) == 1) {
                    if      (ch == '1') { kbd_sit.store(true);   std::cout << "[Sit]\n"     << std::flush; }
                    else if (ch == '2') { kbd_stand.store(true); std::cout << "[Stand]\n"   << std::flush; }
                    else if (ch == '3') { kbd_ctrl.store(true);  std::cout << "[Ctrl]\n"    << std::flush; }
                    else if (ch == '4') { kbd_damp.store(true);  std::cout << "[Damping]\n" << std::flush; }
                    else if (ch == 'w') {  // toggle forward
                        float v = (kbd_vx.load() == kFwdSpeed) ? 0.f : kFwdSpeed;
                        kbd_vx.store(v);
                        std::cout << "[forward " << (v != 0.f ? "ON" : "off") << "]\n" << std::flush;
                    }
                    else if (ch == 's') {  // toggle backward
                        float v = (kbd_vx.load() == -kFwdSpeed) ? 0.f : -kFwdSpeed;
                        kbd_vx.store(v);
                        std::cout << "[backward " << (v != 0.f ? "ON" : "off") << "]\n" << std::flush;
                    }
                    else if (ch == 'a') {  // toggle turn left
                        float v = (kbd_wz.load() == kYawRate) ? 0.f : kYawRate;
                        kbd_wz.store(v);
                        std::cout << "[turn-left " << (v != 0.f ? "ON" : "off") << "]\n" << std::flush;
                    }
                    else if (ch == 'd') {  // toggle turn right
                        float v = (kbd_wz.load() == -kYawRate) ? 0.f : -kYawRate;
                        kbd_wz.store(v);
                        std::cout << "[turn-right " << (v != 0.f ? "ON" : "off") << "]\n" << std::flush;
                    }
                    else if (ch == ' ') {  // stop all
                        kbd_vx.store(0.f);
                        kbd_wz.store(0.f);
                        std::cout << "[STOP]\n" << std::flush;
                    }
                }
            }
            tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        });
        kb_thread.detach();

        std::cout << "Press Enter (or gamepad START) to start!" << std::endl;
        while (true)
        {
            std::this_thread::sleep_for(duration);
            InteprateGamePad();
            if (gamepad.start.on_press || keyboard_start.load())
            {
                break;
            }
        }

        std::cout << "Start!" << std::endl;
        Damping();
        ctrl_dt_micro_sec = static_cast<uint64_t>(ctrl->dt * 1000000); // 0.02s, 50Hz

        control_thread_ptr = CreateRecurrentThreadEx("ctrl", UT_CPU_ID_NONE, ctrl_dt_micro_sec, &RobotController::ControlStep, this);

        std::this_thread::sleep_for(duration);
        StartSendCmd();
        std::cout << "Start Send Cmd!" << std::endl;
        std::cout << "Keys (no Enter): [1]=Sit  [2]=Stand  [3]=RL Ctrl  [4]=Damping (E-STOP)\n"
                  << "Manual drive:    [w/s]=fwd/back  [a/d]=turn  [space]=stop  (IGNORED in MPPI/nav auto modes)\n"
                  << "Sequence: 1 -> 2 -> 3   (in MPPI mode the robot then drives ITSELF to the goal; [4]=E-stop)\n"
                  << std::flush;

        while (true)
        {
            std::this_thread::sleep_for(duration);
        }
    }

protected:
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> highstate_subscriber;
    ChannelPublisherPtr<unitree_go::msg::dds_::SportModeState_> twin_publisher;
    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr;
    unitree_go::msg::dds_::LowCmd_ cmd;
    unitree_go::msg::dds_::LowState_ state;

    Gamepad gamepad;
    std::atomic<bool> kbd_sit{false};
    std::atomic<bool> kbd_stand{false};
    std::atomic<bool> kbd_ctrl{false};
    std::atomic<bool> kbd_damp{false};
    std::atomic<float> kbd_vx{0.f};
    std::atomic<float> kbd_wz{0.f};
    REMOTE_DATA_RX rx;

    RLStateMachine state_machine;
    BasicUserController* ctrl;
    BasicRobotInterface robot_interface;

    // latest odometry from rt/sportmodestate (guarded by state_mutex)
    double odom_x = 0.0, odom_y = 0.0;

    std::mutex state_mutex, cmd_mutex;

    std::ofstream log_file;

    uint64_t ctrl_dt_micro_sec = 2000;

    std::vector<float> compute_time;

    RobotStateClient rsc;

private:
        
    // 500Hz
    void LowStateMessageHandler(const void *message)
    {
        state = *(unitree_go::msg::dds_::LowState_ *)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            robot_interface.SetState(state);
        }
    }

    // odometry handler
    void HighStateMessageHandler(const void *message)
    {
        const auto *hs = (const unitree_go::msg::dds_::SportModeState_ *)message;
        std::lock_guard<std::mutex> lock(state_mutex);
        odom_x = hs->position()[0];
        odom_y = hs->position()[1];
    }

    void InteprateGamePad()
    {
        // update gamepad
        memcpy(rx.buff, &state.wireless_remote()[0], 40);
        gamepad.update(rx.RF_RX);
    }

    // 500Hz
    void LowCmdwriteHandler()
    {
        // write low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            lowcmd_publisher->Write(cmd);
        }
        
    }

    void init_low_cmd()
    {
        cmd.head()[0] = 0xFE;
        cmd.head()[1] = 0xEF;
        cmd.level_flag() = 0xFF;
        cmd.gpio() = 0;

        for(int i=0; i<20; i++)
        {
            cmd.motor_cmd()[i].mode() = (0x01);   // motor switch to servo (PMSM) mode
            cmd.motor_cmd()[i].q() = (PosStopF);
            cmd.motor_cmd()[i].kp() = (0);
            cmd.motor_cmd()[i].dq() = (VelStopF);
            cmd.motor_cmd()[i].kd() = (0);
            cmd.motor_cmd()[i].tau() = (0);
        }
    }

    void set_cmd()
    {
        for(int i = 0; i < 12; i++)
        {
            cmd.motor_cmd()[i].q() = robot_interface.jpos_des.at(i);
            cmd.motor_cmd()[i].dq() = robot_interface.jvel_des.at(i);
            cmd.motor_cmd()[i].kp() = robot_interface.kp.at(i);
            cmd.motor_cmd()[i].kd() = robot_interface.kd.at(i);
            cmd.motor_cmd()[i].tau() = robot_interface.tau_ff.at(i);
        }
        cmd.crc() = crc32_core((uint32_t *)&cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    }

    void StartSendCmd()
    {
        low_cmd_write_thread_ptr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &RobotController::LowCmdwriteHandler, this);
    }

   void UpdateStateMachine()
    {
        // L1 + R1 -> Sit
        // L1 + R2 -> Stand  (or keyboard 's')
        // L1 + A  -> Ctrl   (or keyboard 'c')
        // L1 + Y  -> Stop   (or keyboard 'd')
        if((gamepad.R1.on_press && gamepad.L1.pressed) || kbd_sit.exchange(false))
        {
            if(state_machine.Sit())
            {
                SitCallback();
            }
        }
        if ((gamepad.R2.on_press && gamepad.L1.pressed) || kbd_stand.exchange(false))
        {
            if (state_machine.Stand())
            {
                StandCallback();
            }
        }
        if ((gamepad.A.on_press && gamepad.L1.pressed) || kbd_ctrl.exchange(false))
        {
            if (state_machine.Ctrl())
            {
                CtrlCallback();
            }
        }
        if ((gamepad.Y.pressed && gamepad.L1.pressed) || kbd_damp.exchange(false))
        {
            state_machine.Stop();
        }
    }

    // 50Hz
    void ControlStep()
    {
        // main loop

        // update state
        InteprateGamePad(); // 接收数据是500Hz频率, 但是我只有运行主控制线程时才需要处理好的gamepad数据
                            // 所以可以在主控制程序中处理gamepad数据
        // Inject keyboard velocity (overrides physical gamepad for sim2sim use).
        // SimpleRLController::GetInput reads forward from gamepad.lx (cmd[0]),
        // lateral from gamepad.ly (cmd[1]), yaw from gamepad.rx (cmd[2]).
        gamepad.lx = kbd_vx.load();   // 'w'/'x' -> forward/back
        gamepad.ly = 0.f;             // no keyboard strafe
        gamepad.rx = -kbd_wz.load();  // 'a'/'e' -> yaw
        UpdateStateMachine();

        // select control modes according to the state machine
        auto start = std::chrono::high_resolution_clock::now();
        // 读取状态信息
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ctrl->GetInput(robot_interface, gamepad);
            // feed world odometry (yaw from IMU) to the CBF safety filter
            ctrl->setOdometry(odom_x, odom_y, robot_interface.rpy.at(2));
        }
        // 根据状态判断具体执行哪个状态下的处理函数
        if(state_machine.state == STATES::SIT)
        {
            Sitting();
        }
        if (state_machine.state == STATES::STAND)
        {
            Standing();
        }
        if (state_machine.state == STATES::DAMPING)
        {
            Damping();
        }
        if (state_machine.state == STATES::CTRL)
        {
            UserControlStep(true);
            if (CheckTermination())
            {
                state_machine.Stop();
                Damping();
            }
        }

        // update low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            set_cmd();
        }

        // publish dead-reckoned pose for the digital-twin viewer (x,y in start frame, yaw0)
        {
            double tx = 0, ty = 0, tyaw0 = 0;
            ctrl->getTwinState(tx, ty, tyaw0);
            unitree_go::msg::dds_::SportModeState_ twin{};
            twin.position()[0] = static_cast<float>(tx);
            twin.position()[1] = static_cast<float>(ty);
            twin.position()[2] = static_cast<float>(tyaw0);
            twin_publisher->Write(twin);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        compute_time.push_back(duration.count() / 1000.); // us->ms

        // write log
        WriteLog();

        if (compute_time.size() == 50)
        {
            float sum = 0;
            for (auto &t : compute_time)
            {
                sum += t;
            }
            std::cout << "Performance: mean: " << sum / 50
                      << " ms; max: " << *std::max_element(compute_time.begin(), compute_time.end())
                      << " ms; min: " << *std::min_element(compute_time.begin(), compute_time.end())
                      << "ms." << std::endl;

            compute_time.clear();

            std::cout << "Current State: " << static_cast<size_t>(state_machine.state) << std::endl;
        }
    }

    void WriteLog()
    {
        if (log_file.is_open())
        {
            log_file << static_cast<size_t>(state_machine.state) << ",";
            auto log = ctrl->GetLog();
            for (const auto &v : log)
            {
                log_file << v << ",";
            }
            log_file << std::endl;
        }
    }

    void SitCallback()
    {
        // 保存初始关节位置
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ctrl->save_jpos(robot_interface);
        }
        robot_interface.jpos_des = ctrl->start_pos; // 由阻尼状态的初始位置开始
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(ctrl->stand_kp);
        robot_interface.kd.fill(ctrl->stand_kd);
        robot_interface.tau_ff.fill(0.);
    }
    /**
     * @brief 站立状态回调函数
     * @note  进入状态时调用一次, 用于进入站立状态的初始化工作
    */
    void StandCallback()
    {
        // 保存初始关节位置
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ctrl->save_jpos(robot_interface);
        }
        robot_interface.jpos_des = ctrl->start_pos; // 由阻尼状态的初始位置开始
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(ctrl->stand_kp);
        robot_interface.kd.fill(ctrl->stand_kd);
        robot_interface.tau_ff.fill(0.);
    }

    void CtrlCallback()
    {
        // 重置状态
        ctrl->reset(robot_interface, gamepad);
        ctrl->resetOdometry();   // dead-reckoned CBF position starts at (0,0) here

        robot_interface.jpos_des = ctrl->stand_pos;
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(ctrl->ctrl_kp);
        robot_interface.kd.fill(ctrl->ctrl_kd);
        robot_interface.tau_ff.fill(0.);
    }

    void Damping(float kd = 3.0)
    {
        robot_interface.jpos_des.fill(0.);
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(0.);
        robot_interface.kd.fill(kd);
        robot_interface.tau_ff.fill(0.);
    }

    void Sitting(float kp = 60.0, float kd = 5.0)
    {
        state_machine.Sitting(*ctrl);

        robot_interface.jpos_des = ctrl->jpos_des;
    }

    void Standing(float kp = 60.0, float kd = 5.0)
    {
        ctrl->DummyCalculate(); // warmup the neural network

        state_machine.Standing(*ctrl);

        robot_interface.jpos_des = ctrl->jpos_des;
    }

    void UserControlStep(bool send_cmd = false)
    {
        ctrl->Calculate();

        if (send_cmd)
        {
            robot_interface.jpos_des = ctrl->jpos_des;
        }
        
    }

    bool CheckTermination()
    {
        // if (robot_interface.rpy.at(0) > M_PI/3)
        // {
        //     return true;
        // }
        // if(max_abs(ctrl->jpos_des) > 3*M_PI/2)
        // {
        //     return true;
        // }
        return false;
    }
};
