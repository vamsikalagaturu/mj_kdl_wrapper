// test_kinova_gen3_menagerie.cpp
// Load Kinova Gen3 from assets/kinova_gen3/gen3.xml (MuJoCo Menagerie MJCF)
// and validate the KDL chain built directly from the compiled model.
//
// Tests:
//   1. MJCF loads: nv=7, nbody=9.
//   2. KDL chain (7 joints) built from model via init_from_mjcf.
//   3. KDL gravity torques agree with MuJoCo qfrc_bias at q=0 within 1e-3 Nm.
//   4. 500-step gravity-comp loop: EE drift < 1 mm.
//
// Usage: test_kinova_gen3_menagerie [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

int main(int argc, char* argv[])
{
    bool gui = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gui") gui = true;

    const fs::path root  = repo_root();
    const std::string mjcf = (root / "assets/kinova_gen3/gen3.xml").string();

    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, mjcf.c_str())) {
        std::cerr << "FAIL: load_mjcf\n";
        return 1;
    }

    if (model->nv != 7) {
        std::cerr << "FAIL: expected nv=7, got " << model->nv << "\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    if (model->nbody != 9) {
        std::cerr << "FAIL: expected nbody=9, got " << model->nbody << "\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "Model: nq=" << model->nq << " nv=" << model->nv
              << " nbody=" << model->nbody << " nu=" << model->nu << "\n";

    mj_kdl::State s;
    if (!mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link")) {
        std::cerr << "FAIL: init_from_mjcf\n";
        mj_kdl::destroy_scene(model, data); return 1;
    }

    unsigned n = s.chain.getNrOfJoints();
    if (n != 7u) {
        std::cerr << "FAIL: expected 7 KDL joints, got " << n << "\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "KDL chain: " << n << " joints\n";

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainDynParam              dyn(s.chain, KDL::Vector(0, 0, -9.81));

    // Test 3: gravity torques vs MuJoCo qfrc_bias at q=0
    {
        KDL::JntArray q0(n);
        mj_kdl::sync_from_kdl(&s, q0);
        mj_forward(model, data);

        KDL::JntArray g(n);
        if (dyn.JntToGravity(q0, g) < 0) {
            std::cerr << "FAIL: JntToGravity\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }

        double max_err = 0.0;
        for (unsigned i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(g(i) - data->qfrc_bias[s.kdl_to_mj_dof[i]]));

        std::cout << std::fixed << std::setprecision(6)
                  << "Gravity accuracy at q=0: max|KDL - MuJoCo| = " << max_err << " Nm\n";

        if (max_err > 1e-3) {
            std::cerr << "FAIL: gravity error " << max_err << " Nm > 1e-3\n";
            mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
        }
        std::cout << "  OK\n";
    }

    // Test 4: 500-step gravity-comp drift at alternating ±30° pose
    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    mj_kdl::sync_from_kdl(&s, q_test);
    mj_forward(model, data);

    KDL::Frame fk_initial;
    fk.JntToCart(q_test, fk_initial);

    int saved_flags = model->opt.disableflags;
    model->opt.disableflags |= mjDSBL_ACTUATION;

    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q(n), g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
        mj_kdl::step(&s);
    }

    model->opt.disableflags = saved_flags;

    KDL::JntArray q_end(n);
    mj_kdl::sync_to_kdl(&s, q_end);
    KDL::Frame fk_end;
    fk.JntToCart(q_end, fk_end);
    double drift = (fk_initial.p - fk_end.p).Norm();

    std::cout << std::setprecision(3)
              << "Gravity-comp drift after 500 steps: " << drift * 1000.0 << " mm\n";

    if (drift > 0.001) {
        std::cerr << "FAIL: drift " << drift * 1000.0 << " mm > 1 mm\n";
        mj_kdl::cleanup(&s); mj_kdl::destroy_scene(model, data); return 1;
    }
    std::cout << "  OK\n\nOK\n";

    if (gui) {
        if (!mj_kdl::init_window(&s, "Gen3 menagerie — gravity comp")) {
            std::cerr << "No display — skipping GUI\n";
        } else {
            mj_kdl::sync_from_kdl(&s, q_test);
            mj_forward(model, data);
            std::cout << "GUI: close window to exit\n";
            while (mj_kdl::is_running(&s)) {
                KDL::JntArray q(n), g(n);
                mj_kdl::sync_to_kdl(&s, q);
                dyn.JntToGravity(q, g);
                mj_kdl::set_torques(&s, g);
                mj_kdl::step(&s);
                mj_kdl::render(&s);
            }
        }
    }

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
