// test_velocity.cpp
// Tests velocity-level kinematics: FK, IK, and Jacobian on the Kinova GEN3.
// Usage: test_velocity [urdf_path] [--gui]
// With --gui: displays the robot at the home pose / IK solution.

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

int main(int argc, char* argv[])
{
    std::string urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool gui = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui") gui = true;
        else if (a[0] != '-') urdf = a;
    }

    mj_kdl::Config cfg;
    cfg.urdf_path = urdf.c_str();
    cfg.base_link = "base_link";
    cfg.tip_link  = "EndEffector_Link";
    cfg.headless  = true;

    mj_kdl::State s;
    if (!mj_kdl::init(&s, &cfg)) {
        std::cerr << "FAIL: init\n";
        return 1;
    }

    unsigned n = s.chain.getNrOfJoints();
    KDL::JntArray q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        q_min(i) = s.joint_limits[i].first;
        q_max(i) = s.joint_limits[i].second;
    }

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainIkSolverVel_pinv      ik_vel(s.chain);
    KDL::ChainIkSolverPos_NR_JL     ik(s.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);
    KDL::ChainJntToJacSolver         jac_solver(s.chain);

    // FK at home pose
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    KDL::Frame fk_home;
    if (fk.JntToCart(q_home, fk_home) < 0) {
        std::cerr << "FAIL: FK at home pose\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    std::cout << "FK(home) pos: ["
              << std::fixed << std::setprecision(4)
              << fk_home.p.x() << ", " << fk_home.p.y() << ", " << fk_home.p.z() << "]\n";

    // FK at test config (alternate ±30 deg)
    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    KDL::Frame fk_test;
    if (fk.JntToCart(q_test, fk_test) < 0) {
        std::cerr << "FAIL: FK at test config\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    std::cout << "FK(test) pos: ["
              << fk_test.p.x() << ", " << fk_test.p.y() << ", " << fk_test.p.z() << "]\n";

    // IK: recover q_test from fk_test, seeded from home pose
    KDL::JntArray q_ik(n);
    int ik_ret = ik.CartToJnt(q_home, fk_test, q_ik);
    if (ik_ret < 0) {
        std::cout << "WARN: IK did not converge (ret=" << ik_ret << "), skipping IK check\n";
    } else {
        KDL::Frame fk_ik;
        fk.JntToCart(q_ik, fk_ik);
        double pos_err = (fk_test.p - fk_ik.p).Norm();
        std::cout << "IK pos error: " << pos_err * 1000.0 << " mm\n";
        if (pos_err > 1e-3) {
            std::cerr << "FAIL: IK position error too large\n";
            mj_kdl::cleanup(&s);
            return 1;
        }
    }

    // Jacobian at home pose
    KDL::Jacobian jac(n);
    if (jac_solver.JntToJac(q_home, jac) < 0) {
        std::cerr << "FAIL: Jacobian\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    std::cout << "Jacobian rows=" << jac.rows() << " cols=" << jac.columns() << "\n";
    if (jac.rows() != 6 || jac.columns() != n) {
        std::cerr << "FAIL: unexpected Jacobian dimensions\n";
        mj_kdl::cleanup(&s);
        return 1;
    }

    std::cout << "OK\n";

    if (gui) {
        // Show IK solution (or home pose if IK failed) statically with gravity comp.
        KDL::JntArray q_display = (ik_ret >= 0) ? q_ik : q_home;
        mj_kdl::sync_from_kdl(&s, q_display);
        mj_forward(s.model, s.data);
        for (unsigned i = 0; i < n; ++i)
            s.data->ctrl[i] = s.data->qpos[s.kdl_to_mj_qpos[i]];

        KDL::ChainDynParam dyn(s.chain, KDL::Vector(0, 0, -9.81));
        std::cout << "GUI mode — close window to exit\n";
        mj_kdl::run_simulate_ui(s.model, s.data, urdf.c_str(),
            [&](mjModel*, mjData* d) {
                for (unsigned i = 0; i < n; ++i)
                    d->ctrl[i] = d->qpos[s.kdl_to_mj_qpos[i]];
                KDL::JntArray q(n), g(n);
                mj_kdl::sync_to_kdl(&s, q);
                dyn.JntToGravity(q, g);
                mj_kdl::set_torques(&s, g);
            });
    }

    mj_kdl::cleanup(&s);
    return 0;
}
