/* ex_pos_ctrl.cpp  (MJCF)
 * Joint position control: drive the Kinova GEN3 from the home pose to a target
 * pose using linearly interpolated position setpoints, then hold.
 *
 * The setpoint trajectory is a straight-line interpolation in joint space over
 * kMotionDuration seconds.  After the motion completes the final position is
 * held for 1 s.
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_pos_ctrl [--headless]
 *
 * Runs the full motion once and exits; --headless skips the viewer and prints the final
 * max joint error. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using mj_kdl_examples::kHomePose;
static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 2.0; // seconds for the interpolated move

int main(int argc, char *argv[])
{
    const bool headless = mj_kdl_examples::parse_args(argc, argv).headless;

    const std::string mjcf = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    mj_kdl::RobotSpec r;
    r.path = mjcf;
    sc.robots.push_back(r);

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &sc)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link")) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    unsigned n = static_cast<unsigned>(robot.n_joints);

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::POSITION)) return 1;

    double t_start = 0.0;

    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, q_home);
        t_start = ctx->data->time;
    };

    mj_kdl::reset(&env);

    auto ctrl_step = [&]() {
        mj_kdl::update(&env);

        double alpha = std::clamp((env.data->time - t_start) / kMotionDuration, 0.0, 1.0);
        for (unsigned i = 0; i < n; ++i)
            robot.jnt_pos_cmd[i] = kHomePose[i] + alpha * (kTargetPose[i] - kHomePose[i]);
    };

    // Run until the trajectory is complete plus a short settling period.
    const double end_time = kMotionDuration + 1.0;
    if (headless) {
        while (env.data->time < end_time) {
            ctrl_step();
            mj_kdl::step(&env);
        }

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(kTargetPose[i] - robot.jnt_pos_msr[i]));
        std::cout << "max joint error at end: " << std::fixed << std::setprecision(4) << max_err
                  << " rad\n";
    } else {
        if (!mj_kdl::open_viewer(&env)) {
            std::cerr << "open_viewer() failed\n";
            return 1;
        }
        while (env.data->time - t_start < end_time) {
            ctrl_step();
            if (!mj_kdl::step(&env)) break;
            mj_kdl::pace_realtime(&env);
        }
    }

    mj_kdl::cleanup(&env);
    return 0;
}
