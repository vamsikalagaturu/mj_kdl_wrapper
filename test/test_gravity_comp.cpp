// test_gravity_comp.cpp
// Two-part gravity compensation test using KDL dynamics:
//
// Part 1 — KDL vs MuJoCo accuracy at q=0:
//   Compares KDL::ChainDynParam::JntToGravity against MuJoCo qfrc_bias.
//   Checks |kdl_g - mujoco_bias| < 1e-3 Nm for all joints.
//
// Part 2 — KDL gravity comp drift test:
//   Sets arm to alternating ±30° pose, then runs 500 steps applying
//   KDL-computed gravity torques via ChainDynParam::JntToGravity each step.
//   EE drift must stay < 1 mm.
//
// Usage: test_gravity_comp [urdf_path] [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

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
    cfg.win_title = "test_gravity_comp";
    cfg.headless  = !gui;

    mj_kdl::State s;
    if (!mj_kdl::init(&s, &cfg)) {
        std::cerr << "FAIL: init\n";
        return 1;
    }

    unsigned n = s.chain.getNrOfJoints();
    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainDynParam              dyn(s.chain, KDL::Vector(0.0, 0.0, -9.81));

    // -----------------------------------------------------------------------
    // Part 1: KDL gravity accuracy at q=0
    // -----------------------------------------------------------------------
    {
        KDL::JntArray q0(n);
        mj_kdl::sync_from_kdl(&s, q0);
        mj_forward(s.model, s.data);

        KDL::JntArray g(n);
        if (dyn.JntToGravity(q0, g) < 0) {
            std::cerr << "FAIL: JntToGravity returned error\n";
            mj_kdl::cleanup(&s);
            return 1;
        }

        const double* bias = s.data->qfrc_bias;
        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(g(i) - bias[i]));

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Part 1 — KDL gravity vs MuJoCo bias at q=0:\n";
        std::cout << "  max |error| = " << max_err << " Nm\n";

        if (max_err > 1e-3) {
            std::cerr << "FAIL: max error " << max_err << " Nm > 0.001 Nm\n";
            mj_kdl::cleanup(&s);
            return 1;
        }
        std::cout << "  OK\n";
    }

    // -----------------------------------------------------------------------
    // Part 2: KDL gravity comp drift test at alternating ±30° pose
    // -----------------------------------------------------------------------
    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    mj_kdl::sync_from_kdl(&s, q_test);
    mj_forward(s.model, s.data);

    KDL::Frame fk_initial;
    fk.JntToCart(q_test, fk_initial);

    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q, g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
        mj_kdl::step(&s);
    }

    KDL::JntArray q_end;
    mj_kdl::sync_to_kdl(&s, q_end);
    KDL::Frame fk_end;
    fk.JntToCart(q_end, fk_end);
    double drift = (fk_initial.p - fk_end.p).Norm();

    std::cout << "\nPart 2 — KDL gravity comp drift after 500 steps:\n";
    std::cout << "  EE drift = " << std::setprecision(3) << drift * 1000.0 << " mm\n";

    if (drift > 0.001) {
        std::cerr << "FAIL: drift " << drift * 1000.0 << " mm > 1 mm\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    std::cout << "  OK\n\nOK\n";

    if (gui) {
        mj_kdl::sync_from_kdl(&s, q_test);
        mj_forward(s.model, s.data);
        std::cout << "GUI mode — close window to exit\n";
        while (mj_kdl::is_running(&s)) {
            KDL::JntArray q, g(n);
            mj_kdl::sync_to_kdl(&s, q);
            dyn.JntToGravity(q, g);
            mj_kdl::set_torques(&s, g);
            mj_kdl::step(&s);
            mj_kdl::render(&s);
        }
    }

    mj_kdl::cleanup(&s);
    return 0;
}
