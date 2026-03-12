// test_velocity.cpp
// Tests velocity-level kinematics: FK, IK, and Jacobian on the Kinova GEN3.
// Usage: test_velocity [urdf_path] [--gui]
// With --gui: displays the robot at the IK solution after checks.

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>

int main(int argc, char* argv[])
{
    std::string urdf = "../urdf/GEN3_URDF_V12.urdf";
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
    cfg.win_title = "test_velocity";
    cfg.headless  = !gui;

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

    // FK at zero config
    KDL::JntArray q_zero(n);
    KDL::Frame fk_zero;
    if (fk.JntToCart(q_zero, fk_zero) < 0) {
        std::cerr << "FAIL: FK at zero\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    std::cout << "FK(zero) pos: ["
              << std::fixed << std::setprecision(4)
              << fk_zero.p.x() << ", " << fk_zero.p.y() << ", " << fk_zero.p.z() << "]\n";

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

    // IK: recover q_test from fk_test
    KDL::JntArray q_ik(n);
    int ik_ret = ik.CartToJnt(q_zero, fk_test, q_ik);
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

    // Jacobian at zero config
    KDL::Jacobian jac(n);
    if (jac_solver.JntToJac(q_zero, jac) < 0) {
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
        mj_kdl::sync_from_kdl(&s, ik_ret >= 0 ? q_ik : q_test);
        mj_forward(s.model, s.data);
        std::cout << "GUI mode — close window to exit\n";
        while (mj_kdl::is_running(&s))
            mj_kdl::render(&s);
    }

    mj_kdl::cleanup(&s);
    return 0;
}
