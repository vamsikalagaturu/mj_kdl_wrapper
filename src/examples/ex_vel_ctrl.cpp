/* ex_vel_ctrl.cpp
 * Joint velocity control: drive the Kinova GEN3 from the home pose to a target
 * pose using a proportional velocity controller, then hold.
 *
 * gen3.xml has POSITION actuators, so velocity is implemented by integrating
 * a clamped velocity command into the position setpoint each step:
 *
 *   vel[i]          = clamp(Kv * (target[i] - q[i]), -maxVel, maxVel)
 *   jnt_pos_cmd[i] += vel[i] * dt
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_vel_ctrl [--headless]
 *
 * With --headless runs until convergence (or 5 s timeout) and prints
 * final max joint error. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7]   = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTargetPose[7] = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };

static constexpr double kKv     = 2.0;  // proportional gain [rad/s per rad error]
static constexpr double kMaxVel = 0.6;  // max joint velocity [rad/s]
static constexpr double kTol    = 0.01; // convergence tolerance [rad]

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const std::string mjcf = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    mj_kdl::RobotSpec r;
    r.path = mjcf.c_str();
    sc.robots.push_back(r);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link")) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n = static_cast<unsigned>(robot.n_joints);

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::POSITION;

    mj_kdl::Env env;
    env.spec  = sc;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    const double dt      = model->opt.timestep;
    bool         arrived = false;

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home, false);
        arrived = false;
    };

    mj_kdl::reset(&env);

    auto ctrl_step = [&]() {
        mj_kdl::update(&robot);

        if (arrived) return;

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i) {
            double err = kTargetPose[i] - robot.jnt_pos_msr[i];
            max_err    = std::max(max_err, std::abs(err));
            double vel = std::clamp(kKv * err, -kMaxVel, kMaxVel);
            robot.jnt_pos_cmd[i] += vel * dt;
        }

        if (max_err < kTol) {
            arrived = true;
            for (unsigned i = 0; i < n; ++i) robot.jnt_pos_cmd[i] = robot.jnt_pos_msr[i];
        }
    };

    if (headless) {
        const double timeout = 5.0;
        while (data->time < timeout && !arrived) {
            ctrl_step();
            mj_kdl::step(&robot);
            mj_kdl::pace_realtime(&robot);
        }

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(kTargetPose[i] - robot.jnt_pos_msr[i]));
        std::cout << "max joint error: " << std::fixed << std::setprecision(4) << max_err
                  << " rad  (" << (arrived ? "converged" : "timeout") << ")\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        double prev_sim_time = data->time;
        while (true) {
            if (data->time < prev_sim_time - 1e-6) mj_kdl::reset(&env);
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::step(&robot)) break;
            mj_kdl::pace_realtime(&robot);
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
