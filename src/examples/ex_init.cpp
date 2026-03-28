/* ex_init.cpp  (MJCF)
 * Basic example: load the Kinova GEN3 from MuJoCo Menagerie MJCF,
 * set the arm to home pose, and run the MuJoCo simulate UI.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_init_mjcf [--headless]
 *
 * With --headless exits after printing basic model information. */

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

    unsigned n = static_cast<unsigned>(robot.n_joints);
    std::cout << "nq=" << model->nq << "  nu=" << model->nu << "  nbody=" << model->nbody
              << "  n_joints=" << n << "\n";
    for (unsigned i = 0; i < n; ++i)
        std::cout << "  joint[" << i << "] = " << robot.joint_names[i] << "\n";

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));
    KDL::Frame                      ee;
    fk.JntToCart(q_home, ee);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "EE at home: [" << ee.p.x() << ", " << ee.p.y() << ", " << ee.p.z() << "]\n";

    if (headless) {
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 0;
    }

    mj_kdl::set_joint_pos(&robot, q_home);
    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Viewer viewer;
    if (!mj_kdl::init_window_sim(&viewer, &robot)) {
        std::cerr << "init_window_sim() failed\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    KDL::JntArray q(n), g(n);
    while (mj_kdl::tick(&viewer, model, data)) {
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
    }

    mj_kdl::cleanup(&viewer);
    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
