/* ex_gravity_comp.cpp  (MJCF)
 * KDL gravity compensation on the Kinova GEN3 loaded from MuJoCo Menagerie MJCF.
 *
 * Each physics step computes joint gravity torques via
 * KDL::ChainDynParam::JntToGravity and applies them via update() in
 * TORQUE mode, keeping the arm floating at home pose.
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_gravity_comp_mjcf [--headless]
 *
 * Runs 500 steps and exits; --headless skips the viewer and prints the final EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <iomanip>
#include <iostream>
#include <string>

using mj_kdl_examples::kHomePose;
static constexpr int    kSteps       = 500;

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

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &sc)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link")) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    unsigned                        n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    env.on_reset = [&](mj_kdl::ResetContext *) { mj_kdl::set_joint_pos(&robot, q_home); };

    mj_kdl::reset(&env);

    KDL::JntArray q(n), g(n);
    auto          ctrl_step = [&]() {
        mj_kdl::update(&env);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
    };

    if (headless) {
        KDL::Frame ee_start;
        fk.JntToCart(q_home, ee_start);

        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            mj_kdl::step(&env);
        }

        KDL::JntArray q_end(n);
        for (unsigned i = 0; i < n; ++i) q_end(i) = robot.jnt_pos_msr[i];
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after " << kSteps << " steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
    } else {
        if (!mj_kdl::open_viewer(&env)) {
            std::cerr << "open_viewer() failed\n";
            return 1;
        }
        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            if (!mj_kdl::step(&env)) break;
            mj_kdl::pace_realtime(&env);
        }
    }

    mj_kdl::cleanup(&env);
    return 0;
}
