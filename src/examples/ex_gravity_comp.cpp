/* ex_gravity_comp.cpp  (MJCF)
 * KDL gravity compensation on the Kinova GEN3 loaded from MuJoCo Menagerie MJCF.
 *
 * Each physics step computes joint gravity torques via
 * KDL::ChainDynParam::JntToGravity and applies them via update() in
 * TORQUE mode, keeping the arm floating at home pose.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_gravity_comp_mjcf [--headless]
 *
 * With --headless runs 500 steps and prints the final EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();

    mj_kdl::SceneSpec sc;
    mj_kdl::RobotSpec r;
    r.path = mjcf.c_str();
    sc.robots.push_back(r);

    mjModel      *model = nullptr;
    mjData       *data  = nullptr;
    mj_kdl::Robot robot;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link")) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned                        n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.spec  = sc;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home, false);
    };

    mj_kdl::reset(&env);

    KDL::JntArray q(n), g(n);
    auto          ctrl_step = [&]() {
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
    };

    if (headless) {
        KDL::Frame ee_start;
        fk.JntToCart(q_home, ee_start);

        for (int step = 0; step < 500; ++step) {
            ctrl_step();
            mj_kdl::step(&robot);
        }

        KDL::JntArray q_end(n);
        for (unsigned i = 0; i < n; ++i) q_end(i) = robot.jnt_pos_msr[i];
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after 500 steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
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
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
