/* ex_dual_arm.cpp
 * Two Kinova Gen3 arms, each fitted with a Robotiq 2F-85 gripper,
 * in a shared MuJoCo scene.
 *
 * arm1 at x = -0.5 m, facing +X.
 * arm2 at x = +0.5 m, facing -X (180 deg yaw); all element names prefixed "r2_".
 *
 * Both arms hold the home pose under gravity compensation.
 * Grippers cycle open/closed every 3 s.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_dual_arm [--headless]
 *
 * --headless: run 600 steps and print both EE positions, then exit. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

static const std::vector<std::pair<std::string, std::string>> kGripperExclusions = {
    { "bracelet_link", "g_base" },
    { "bracelet_link", "g_left_pad" },
    { "bracelet_link", "g_right_pad" },
    { "half_arm_2_link", "g_base" },
};

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  git submodule update --init third_party/menagerie\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string a1       = "/tmp/ex_da_arm1.xml";
    const std::string a2       = "/tmp/ex_da_arm2.xml";

    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, a1.c_str())
        || !mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, a2.c_str())) {
        std::cerr << "attach_gripper() failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_contact_exclusions(a1.c_str(), kGripperExclusions)
        || !mj_kdl::patch_mjcf_contact_exclusions(a2.c_str(), kGripperExclusions)) {
        std::cerr << "patch_mjcf_contact_exclusions() failed\n";
        return 1;
    }

    mj_kdl::RobotSpec arms[2];
    arms[0].path     = a1.c_str();
    arms[0].prefix   = "";
    arms[0].pos[0]   = -0.5;
    arms[0].euler[2] = 0.0;
    arms[1].path     = a2.c_str();
    arms[1].prefix   = "r2_";
    arms[1].pos[0]   = 0.5;
    arms[1].euler[2] = 180.0;

    const std::string combined = "/tmp/ex_dual_arm.xml";
    mjModel          *model    = nullptr;
    mjData           *data     = nullptr;
    if (!mj_kdl::build_scene_from_mjcfs(&model, &data, arms, 2, true, true, combined.c_str())) {
        std::cerr << "build_scene_from_mjcfs() failed\n";
        return 1;
    }

    mj_kdl::Robot arm1, arm2;
    if (!mj_kdl::init_from_mjcf(&arm1, model, data, "base_link", "bracelet_link")
        || !mj_kdl::init_from_mjcf(&arm2, model, data, "r2_base_link", "r2_bracelet_link")) {
        std::cerr << "init_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const int n     = arm1.n_joints;
    int       fing1 = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int       fing2 = mj_name2id(model, mjOBJ_ACTUATOR, "r2_g_fingers_actuator");

    KDL::JntArray q_home(n);
    for (int i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::set_joint_pos(&arm1, q_home);
    mj_kdl::set_joint_pos(&arm2, q_home);
    mj_forward(model, data);

    /* Hold position + feed-forward gravity via qfrc_bias (includes gripper mass). */
    auto apply_ctrl = [&](mj_kdl::Robot &arm) {
        for (int i = 0; i < arm.n_joints; ++i) {
            if (arm.kdl_to_mj_ctrl[i] >= 0) arm.cmd(i) = arm.pos(i);
            data->qfrc_applied[arm.kdl_to_mj_dof[i]] = arm.frc(i);
        }
    };

    if (headless) {
        for (int step = 0; step < 600; ++step) {
            apply_ctrl(arm1);
            apply_ctrl(arm2);
            if (fing1 >= 0) data->ctrl[fing1] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
            if (fing2 >= 0) data->ctrl[fing2] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
            mj_kdl::step(&arm1);
        }

        KDL::ChainFkSolverPos_recursive fk1(arm1.chain), fk2(arm2.chain);
        KDL::JntArray                   q1, q2;
        KDL::Frame                      ee1, ee2;
        mj_kdl::get_joint_pos(&arm1, q1);
        mj_kdl::get_joint_pos(&arm2, q2);
        fk1.JntToCart(q1, ee1);
        fk2.JntToCart(q2, ee2);
        std::cout << "arm1 EE: [" << ee1.p.x() << ", " << ee1.p.y() << ", " << ee1.p.z() << "]\n";
        std::cout << "arm2 EE: [" << ee2.p.x() << ", " << ee2.p.y() << ", " << ee2.p.z() << "]\n";
    } else {
        mj_kdl::run_simulate_ui(
          model, data, combined.c_str(), [&](mjModel * /*m*/, mjData * /*d*/) {
              apply_ctrl(arm1);
              apply_ctrl(arm2);
              if (fing1 >= 0) data->ctrl[fing1] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
              if (fing2 >= 0) data->ctrl[fing2] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
          });
    }

    mj_kdl::cleanup(&arm1);
    mj_kdl::cleanup(&arm2);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
