/* test_init.cpp
 * Loads the Kinova GEN3 URDF, runs 100 simulation steps, and verifies basic
 * model properties are consistent.  Exits with 0 on success, 1 on failure.
 * Usage: test_init [urdf_path] [--gui] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <iostream>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    std::string urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool        gui  = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui")
            gui = true;
        else if (a[0] != '-')
            urdf = a;
    }

    mj_kdl::Config cfg;
    cfg.urdf_path  = urdf.c_str();
    cfg.base_link  = "base_link";
    cfg.tip_link   = "EndEffector_Link";
    cfg.robot_name = "kinova_gen3";
    cfg.headless   = true; // run_simulate_ui opens its own window

    mj_kdl::State s;
    if (!mj_kdl::init(&s, &cfg)) {
        std::cerr << "FAIL: init\n";
        return 1;
    }

    if (s.model->nq != 7 || s.model->nv != 7) {
        std::cerr << "FAIL: expected 7 DOF, got nq=" << s.model->nq << " nv=" << s.model->nv
                  << "\n";
        mj_kdl::cleanup(&s);
        return 1;
    }
    if (s.n_joints != 7) {
        std::cerr << "FAIL: expected 7 KDL joints, got " << s.n_joints << "\n";
        mj_kdl::cleanup(&s);
        return 1;
    }

    // Set home pose
    unsigned      n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(s.model, s.data);

    const double t0 = s.data->time;
    mj_kdl::step_n(&s, 100);
    if (s.data->time <= t0) {
        std::cerr << "FAIL: simulation time did not advance\n";
        mj_kdl::cleanup(&s);
        return 1;
    }

    std::cout << "OK  nq=" << s.model->nq << " nv=" << s.model->nv << " kdl_joints=" << s.n_joints
              << " sim_time=" << s.data->time << "\n";

    if (gui) {
        std::cout << "GUI mode — close window to exit\n";
        // Reset to home pose before opening UI
        mj_kdl::sync_from_kdl(&s, q_home);
        mj_forward(s.model, s.data);
        for (unsigned i = 0; i < n; ++i) s.data->ctrl[i] = s.data->qpos[s.kdl_to_mj_qpos[i]];
        mj_kdl::run_simulate_ui(s.model, s.data, urdf.c_str());
    }

    mj_kdl::cleanup(&s);
    return 0;
}
