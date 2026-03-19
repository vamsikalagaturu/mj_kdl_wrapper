/* ex_gravity_comp.cpp
 * KDL gravity compensation on the Kinova GEN3.
 *
 * Each physics step computes joint gravity torques via
 * KDL::ChainDynParam::JntToGravity and applies them through
 * mj_kdl::set_torques, keeping the arm floating at home pose.
 *
 * Usage:
 *   ex_gravity_comp [urdf_path] [--headless]
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
    std::string urdf     = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool        headless = false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--headless")
            headless = true;
        else if (a[0] != '-')
            urdf = a;
    }

    mj_kdl::SceneSpec sc;
    mj_kdl::RobotSpec r;
    r.path = urdf.c_str();
    sc.robots.push_back(r);

    mjModel      *model = nullptr;
    mjData       *data  = nullptr;
    mj_kdl::Robot robot;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot(&robot, model, data, urdf.c_str(), "base_link", "EndEffector_Link")) {
        std::cerr << "init_robot() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned                        n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    mj_kdl::sync_from_kdl(&robot, q_home);
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i) data->ctrl[i] = data->qpos[robot.kdl_to_mj_qpos[i]];

    if (headless) {
        KDL::Frame ee_start;
        fk.JntToCart(q_home, ee_start);

        for (int step = 0; step < 500; ++step) {
            KDL::JntArray q(n), g(n);
            mj_kdl::sync_to_kdl(&robot, q);
            dyn.JntToGravity(q, g);
            mj_kdl::set_torques(&robot, g);
            mj_kdl::step(&robot);
        }

        KDL::JntArray q_end(n);
        mj_kdl::sync_to_kdl(&robot, q_end);
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after 500 steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
    } else {
        mj_kdl::run_simulate_ui(model, data, urdf.c_str(), [&](mjModel *, mjData *d) {
            for (unsigned i = 0; i < n; ++i) d->ctrl[i] = d->qpos[robot.kdl_to_mj_qpos[i]];
            KDL::JntArray q(n), g(n);
            mj_kdl::sync_to_kdl(&robot, q);
            dyn.JntToGravity(q, g);
            mj_kdl::set_torques(&robot, g);
        });
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
