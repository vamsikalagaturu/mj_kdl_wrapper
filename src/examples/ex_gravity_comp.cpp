/* ex_gravity_comp.cpp
 * KDL gravity compensation on the Kinova GEN3: each step commands
 * KDL::ChainDynParam::JntToGravity in TORQUE mode, keeping the arm at its home pose.
 *
 * Usage:
 *   ex_gravity_comp [--headless]
 *
 * Runs 15 s (7500 steps) and prints the EE drift; --headless skips the viewer and exits 1 if
 * the drift exceeds kMaxDrift. */

#include "common.hpp"
#include "example_paths.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace ex = mj_kdl_examples;

static constexpr int    kSteps    = 7500;   // 15 s
static constexpr double kMaxDrift = 0.0001; // [m]

int main(int argc, char *argv[])
{
    const bool headless = ex::parse_args(argc, argv).headless;

    mj_kdl::SceneSpec sc = ex::scene_spec();
    mj_kdl::RobotSpec r;
    r.path = ex::menagerie_model("kinova_gen3/gen3.xml");
    sc.robots.push_back(r);

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &sc)) return 1;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link")) return 1;
    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    const unsigned                  n = robot.chain.getNrOfJoints();
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, sc.gravity_z));
    const KDL::JntArray             q_home = ex::home_q(n);

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home);
        ex::prime_gravity(robot, dyn, q_home);
    };
    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    KDL::Frame ee_start;
    fk.JntToCart(q_home, ee_start);
    for (int step = 0; step < kSteps; ++step) {
        mj_kdl::update(&env);
        ex::pd_gravity(robot, dyn, q_home);
        if (!mj_kdl::step(&env)) break;
        mj_kdl::pace_realtime(&env);
    }
    mj_kdl::update(&env);

    const double drift = (ex::tcp_frame(fk, robot).p - ee_start.p).Norm();
    std::cout << "EE drift after " << kSteps << " steps: " << std::fixed << std::setprecision(4)
              << drift * 1000.0 << " mm (limit " << kMaxDrift * 1000.0 << " mm)\n";
    mj_kdl::cleanup(&env);
    return headless ? ex::verdict(drift <= kMaxDrift, "the arm holds its pose") : 0;
}
