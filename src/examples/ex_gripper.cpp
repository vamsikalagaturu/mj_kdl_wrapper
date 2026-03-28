/* ex_gripper.cpp
 * Attach a Robotiq 2F-85 gripper to the Kinova GEN3 and run KDL gravity
 * compensation.  The gripper cycles open and closed every 3 s.
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_gripper [--headless]
 *
 * With --headless runs 300 steps and prints elapsed sim time. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>

#include <cmath>
#include <filesystem>
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
        std::cerr << "third_party/menagerie/ not found  - run: "
                     "git submodule update --init third_party/menagerie\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    mj_kdl::RobotSpec rs;
    rs.path = arm_mjcf.c_str();
    rs.attachments.push_back(gs);

    mj_kdl::SceneSpec sc;
    sc.robots.push_back(rs);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", "g_base"
        )) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned n           = robot.chain.getNrOfJoints();
    int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));
    KDL::JntArray      q_home(n), q(n), g(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    auto reset_to_home = [&]() {
        if (key_id >= 0) {
            mj_resetDataKeyframe(model, data, key_id);
        } else {
            mj_resetData(model, data);
            mj_kdl::set_joint_pos(&robot, q_home, false);
        }
        mj_forward(model, data);
        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_pos_cmd[i]                       = data->qpos[robot.kdl_to_mj_qpos[i]];
            robot.jnt_trq_cmd[i]                       = 0.0;
            data->qfrc_applied[robot.kdl_to_mj_dof[i]] = 0.0;
        }
        mj_kdl::update(&robot);
        if (fingers_act >= 0) data->ctrl[fingers_act] = 255.0;
    };

    reset_to_home();

    auto ctrl_step = [&]() {
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
        data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
    };

    if (headless) {
        for (int step = 0; step < 300; ++step) {
            ctrl_step();
            mj_kdl::step(&robot);
        }
        std::cout << "sim_time=" << data->time << " s\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        double prev_sim_time = data->time;
        while (true) {
            if (data->time < prev_sim_time - 1e-6) reset_to_home();
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::tick(&viewer, model, data)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
