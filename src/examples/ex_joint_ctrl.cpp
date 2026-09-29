/* ex_joint_ctrl.cpp
 * Joint control on the Kinova GEN3 in two motions: POSITION mode drives the arm from home to a
 * target with setpoints interpolated linearly over kMotionDuration and holds it for kHoldTime;
 * then VELOCITY mode (the <joint>_velocity actuators added through RobotSpec::modes) brings it
 * back home with a clamped proportional velocity command:
 *
 *   jnt_vel_cmd[i] = clamp(Kv * (home[i] - q[i]), -maxVel, maxVel)
 *
 * Usage:
 *   ex_joint_ctrl [--headless]
 *
 * Prints each motion's final max joint error; --headless skips the viewer and exits 1 if the
 * POSITION motion ends further than kMaxErr from the target or the VELOCITY motion does not
 * converge within kTol before kTimeout. */

#include "common.hpp"
#include "example_paths.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace ex = mj_kdl_examples;

static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 2.0;   // [s]
static constexpr double kHoldTime       = 1.0;   // [s]
static constexpr double kMaxErr         = 0.01;  // [rad]
static constexpr double kVelGain        = 500.0; // velocity actuator kv [Nm s/rad]
static constexpr double kKv             = 2.0;   // [rad/s per rad]
static constexpr double kMaxVel         = 0.6;   // [rad/s]
static constexpr double kTol            = 0.01;  // [rad]
static constexpr double kTimeout        = 5.0;   // [s]

enum class Motion { Position, Velocity };

int main(int argc, char *argv[])
{
    const bool headless = ex::parse_args(argc, argv).headless;

    mj_kdl::SceneSpec sc = ex::scene_spec();
    mj_kdl::RobotSpec r;
    r.path  = ex::menagerie_model("kinova_gen3/gen3.xml");
    r.modes = { { .mode = mj_kdl::CtrlMode::VELOCITY, .joints = {}, .kv = kVelGain } };
    sc.robots.push_back(r);

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &sc)) return 1;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link")) return 1;

    const unsigned      n       = robot.chain.getNrOfJoints();
    const KDL::JntArray q_home  = ex::home_q(n);
    bool                restart = false;

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home);
        for (unsigned i = 0; i < n; ++i) robot.jnt_pos_cmd[i] = q_home(i);
        restart = true;
    };
    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    constexpr double kNotReached = std::numeric_limits<double>::infinity();
    Motion           motion      = Motion::Position;
    double           t_start     = 0.0;
    double           pos_err     = kNotReached;
    double           vel_err     = kNotReached;
    double           vel_time    = 0.0;
    while (true) {
        if (restart) {
            restart = false;
            if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::POSITION)) return 1;
            motion  = Motion::Position;
            t_start = env.data->time;
            pos_err = vel_err = kNotReached;
        }
        mj_kdl::update(&env);
        const double t       = env.data->time - t_start;
        double       max_err = 0.0;
        if (motion == Motion::Position) {
            const double alpha = ex::clamp01(t / kMotionDuration);
            for (unsigned i = 0; i < n; ++i) {
                robot.jnt_pos_cmd[i] = q_home(i) + alpha * (kTargetPose[i] - q_home(i));
                max_err = std::max(max_err, std::abs(kTargetPose[i] - robot.jnt_pos_msr[i]));
            }
            if (t >= kMotionDuration + kHoldTime) {
                pos_err = max_err;
                if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::VELOCITY)) return 1;
                motion  = Motion::Velocity;
                t_start = env.data->time;
                continue;
            }
        } else {
            for (unsigned i = 0; i < n; ++i) {
                const double err     = q_home(i) - robot.jnt_pos_msr[i];
                max_err              = std::max(max_err, std::abs(err));
                robot.jnt_vel_cmd[i] = ex::clamp_abs(kKv * err, kMaxVel);
            }
            if (max_err < kTol || t >= kTimeout) {
                vel_err  = max_err;
                vel_time = t;
                break;
            }
        }
        if (!mj_kdl::step(&env)) break;
        mj_kdl::pace_realtime(&env);
    }

    const bool pos_ok = pos_err <= kMaxErr;
    const bool vel_ok = vel_err < kTol;
    std::cout << std::fixed << std::setprecision(4) << "POSITION: max joint error at the target "
              << pos_err << " rad (limit " << kMaxErr << " rad)\nVELOCITY: max joint error at home "
              << vel_err << " rad (" << (vel_ok ? "converged" : "not converged")
              << " at t = " << std::setprecision(2) << vel_time << " s, timeout " << kTimeout
              << " s)\n";
    mj_kdl::cleanup(&env);
    return headless ? ex::verdict(pos_ok && vel_ok, "both motions reached their goal") : 0;
}
