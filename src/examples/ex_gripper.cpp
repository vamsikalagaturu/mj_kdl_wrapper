/* ex_gripper.cpp
 * Attach a Robotiq 2F-85 gripper to the Kinova GEN3 and run a gravity
 * compensation controller.  The gripper cycles open and closed every 3 s.
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_gripper [--headless]
 *
 * With --headless runs 300 steps and prints elapsed sim time. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

static const std::vector<std::pair<std::string, std::string>> kGripperExclusions = {
    { "bracelet_link", "g_base" },
    { "bracelet_link", "g_left_pad" },
    { "bracelet_link", "g_right_pad" },
    { "half_arm_2_link", "g_base" },
};

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found  - run: "
                     "git submodule update --init third_party/menagerie\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string combined = "/tmp/ex_gripper.xml";

    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str())) {
        std::cerr << "attach_gripper() failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_add_skybox(combined.c_str())
        || !mj_kdl::patch_mjcf_add_floor(combined.c_str())) {
        std::cerr << "patch_mjcf visuals failed\n";
        return 1;
    }
    if (!mj_kdl::patch_mjcf_contact_exclusions(combined.c_str(), kGripperExclusions)) {
        std::cerr << "patch_mjcf_contact_exclusions() failed\n";
        return 1;
    }

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
        std::cerr << "load_mjcf() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_from_mjcf(&robot, model, data, "base_link", "bracelet_link")) {
        std::cerr << "init_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n           = robot.chain.getNrOfJoints();
    int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    if (key_id >= 0) {
        mj_resetDataKeyframe(model, data, key_id);
    } else {
        KDL::JntArray q_home(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&robot, q_home);
    }
    mj_forward(model, data);

    /* Pure gravity compensation: hold position + cancel gravity via qfrc_bias.
     * qfrc_bias includes gripper mass - KDL alone would miss ~1.7 kg. */
    auto ctrl_step = [&]() {
        for (unsigned i = 0; i < n; ++i) {
            robot.cmd(i)                               = robot.pos(i);
            data->qfrc_applied[robot.kdl_to_mj_dof[i]] = robot.frc(i);
        }
        data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
    };

    if (headless) {
        for (int step = 0; step < 300; ++step) {
            ctrl_step();
            mj_kdl::step(&robot);
        }
        std::cout << "sim_time=" << data->time << " s\n";
    } else {
        mj_kdl::run_simulate_ui(
          model, data, combined.c_str(), [&](mjModel * /*m*/, mjData * /*d*/) { ctrl_step(); });
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
