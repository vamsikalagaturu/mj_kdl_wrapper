/* ex_init.cpp
 * Basic example: load the Kinova GEN3 URDF, set the arm to home pose,
 * and run the MuJoCo simulate UI.
 *
 * Usage:
 *   ex_init [urdf_path] [--headless]
 *
 * With --headless the UI is skipped and the program exits after printing
 * basic model information. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>

#include <filesystem>
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
    if (!mj_kdl::build_scene_from_urdfs(&model, &data, &sc)) {
        std::cerr << "build_scene_from_urdfs() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot(&robot, model, data, urdf.c_str(), "base_link", "EndEffector_Link")) {
        std::cerr << "init_robot() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n = static_cast<unsigned>(robot.n_joints);
    std::cout << "nq=" << model->nq << "  nv=" << model->nv << "  kdl_joints=" << n << "\n";

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    mj_kdl::sync_from_kdl(&robot, q_home);
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i) data->ctrl[i] = data->qpos[robot.kdl_to_mj_qpos[i]];

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::Frame                      ee;
    fk.JntToCart(q_home, ee);
    std::cout << "EE pos at home: [" << ee.p.x() << ", " << ee.p.y() << ", " << ee.p.z() << "]\n";

    if (!headless) mj_kdl::run_simulate_ui(model, data, urdf.c_str());

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
