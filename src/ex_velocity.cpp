/* ex_velocity.cpp
 * FK, IK, and Jacobian on the Kinova GEN3.
 *
 * Computes IK from home pose to a test configuration, then shows the arm
 * at the IK solution (or home pose if IK fails) with KDL gravity compensation.
 *
 * Usage:
 *   ex_velocity [urdf_path] [--headless]
 *
 * With --headless prints FK position, IK error, and Jacobian dimensions only. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    std::string urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool headless    = false;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--headless") headless = true;
        else if (a[0] != '-')  urdf = a;
    }

    mj_kdl::SceneSpec sc;
    mj_kdl::SceneRobot r;
    r.urdf_path = urdf.c_str();
    sc.robots.push_back(r);

    mjModel       *model = nullptr;
    mjData        *data  = nullptr;
    mj_kdl::Robot  robot;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot(&robot, model, data, urdf.c_str(), "base_link", "EndEffector_Link")) {
        std::cerr << "init_robot() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n = static_cast<unsigned>(robot.n_joints);

    KDL::JntArray q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        q_min(i) = robot.joint_limits[i].first;
        q_max(i) = robot.joint_limits[i].second;
    }

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainIkSolverVel_pinv      ik_vel(robot.chain);
    KDL::ChainIkSolverPos_NR_JL     ik(robot.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);
    KDL::ChainJntToJacSolver        jac_solver(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    /* Test config: alternate ±30° per joint. */
    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    /* FK at home and test config. */
    KDL::Frame fk_home, fk_test;
    fk.JntToCart(q_home, fk_home);
    fk.JntToCart(q_test, fk_test);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "FK home:  [" << fk_home.p.x() << ", " << fk_home.p.y()
              << ", " << fk_home.p.z() << "]\n";
    std::cout << "FK test:  [" << fk_test.p.x() << ", " << fk_test.p.y()
              << ", " << fk_test.p.z() << "]\n";

    /* IK from home seed to test-config Cartesian target. */
    KDL::JntArray q_ik(n);
    int ik_ret = ik.CartToJnt(q_home, fk_test, q_ik);
    if (ik_ret >= 0) {
        KDL::Frame fk_ik;
        fk.JntToCart(q_ik, fk_ik);
        double err = (fk_test.p - fk_ik.p).Norm();
        std::cout << "IK pos error: " << err * 1000.0 << " mm\n";
    } else {
        std::cout << "IK did not converge (ret=" << ik_ret << ")\n";
    }

    /* Jacobian at home pose. */
    KDL::Jacobian jac(n);
    jac_solver.JntToJac(q_home, jac);
    std::cout << "Jacobian: " << jac.rows() << " rows x " << jac.columns() << " cols\n";

    if (headless) {
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 0;
    }

    /* Display IK solution (or home pose) with gravity compensation. */
    const KDL::JntArray &q_display = (ik_ret >= 0) ? q_ik : q_home;
    mj_kdl::sync_from_kdl(&robot, q_display);
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i)
        data->ctrl[i] = data->qpos[robot.kdl_to_mj_qpos[i]];

    mj_kdl::run_simulate_ui(model, data, urdf.c_str(), [&](mjModel *, mjData *d) {
        for (unsigned i = 0; i < n; ++i)
            d->ctrl[i] = d->qpos[robot.kdl_to_mj_qpos[i]];
        KDL::JntArray q(n), g(n);
        mj_kdl::sync_to_kdl(&robot, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&robot, g);
    });

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
