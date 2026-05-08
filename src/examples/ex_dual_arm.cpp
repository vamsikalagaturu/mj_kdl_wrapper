/* ex_dual_arm.cpp
 * Two Kinova Gen3 arms, each fitted with a Robotiq 2F-85 gripper,
 * in a shared MuJoCo scene.
 *
 * arm1 at x = -1.5 m, facing +X.
 * arm2 at x = +1.5 m, facing +X; all element names prefixed "r2_".
 *
 * Both arms hold the home pose via PD + KDL gravity compensation.
 * Grippers cycle open/closed every 3 s.
 *
 * Gravity comp uses KDL::ChainDynParam built from a chain that includes
 * the gripper's lumped inertia (via init_robot_from_mjcf tool_body).
 * jnt_trq_cmd is primed before the loop so the first physics step already
 * gets correct compensation.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_dual_arm [--headless]
 *
 * --headless: run 600 steps and print both EE positions, then exit. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

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

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0; // 180 deg around X to flip gripper

    mj_kdl::RobotSpec arm1_spec;
    arm1_spec.path   = arm_mjcf.c_str();
    arm1_spec.prefix = "";
    arm1_spec.pos[0] = -1.0;
    arm1_spec.attachments.push_back(gs);

    mj_kdl::RobotSpec arm2_spec;
    arm2_spec.path   = arm_mjcf.c_str();
    arm2_spec.prefix = "r2_";
    arm2_spec.pos[0] = 1.0;
    arm2_spec.attachments.push_back(gs);

    mj_kdl::SceneSpec sc;
    sc.robots.push_back(arm1_spec);
    sc.robots.push_back(arm2_spec);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    mj_kdl::ToolFrameSpec tool1, tool2;
    tool1.tool_body = "g_base";
    tool1.tcp_site  = "g_pinch";
    tool2.tool_body = "r2_g_base";
    tool2.tcp_site  = "r2_g_pinch";

    mj_kdl::Robot arm1, arm2;
    if (
      !mj_kdl::init_robot_from_mjcf(&arm1, model, data, "base_link", "bracelet_link", "", &tool1)
      || !mj_kdl::init_robot_from_mjcf(
        &arm2, model, data, "r2_base_link", "r2_bracelet_link", "", &tool2
      )
    ) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const int n     = arm1.n_joints;
    int       fing1 = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int       fing2 = mj_name2id(model, mjOBJ_ACTUATOR, "r2_g_fingers_actuator");

    KDL::ChainDynParam dyn1(arm1.chain, KDL::Vector(0.0, 0.0, -9.81));
    KDL::ChainDynParam dyn2(arm2.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n), q1(n), q2(n), g1(n), g2(n);
    for (int i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    arm1.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    arm2.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    // Prime jnt_trq_cmd so the first physics step already gets gravity compensation.
    auto prime_grav = [&]() {
        dyn1.JntToGravity(q_home, g1);
        dyn2.JntToGravity(q_home, g2);
        for (int i = 0; i < n; ++i) {
            arm1.jnt_trq_cmd[i] = g1(i);
            arm2.jnt_trq_cmd[i] = g2(i);
        }
    };

    auto reset_to_home = [&]() {
        mj_resetData(model, data);
        mj_kdl::set_joint_pos(&arm1, q_home, false);
        mj_kdl::set_joint_pos(&arm2, q_home, false);
        mj_forward(model, data);
        for (int i = 0; i < n; ++i) {
            arm1.jnt_pos_cmd[i]                       = data->qpos[arm1.kdl_to_mj_qpos[i]];
            arm2.jnt_pos_cmd[i]                       = data->qpos[arm2.kdl_to_mj_qpos[i]];
            data->qfrc_applied[arm1.kdl_to_mj_dof[i]] = 0.0;
            data->qfrc_applied[arm2.kdl_to_mj_dof[i]] = 0.0;
        }
        prime_grav();
        mj_kdl::update(&arm1);
        mj_kdl::update(&arm2);
        if (fing1 >= 0) data->ctrl[fing1] = 255.0;
        if (fing2 >= 0) data->ctrl[fing2] = 255.0;
    };

    reset_to_home();

    // Per-step: update() reads sensors and flushes the previous jnt_trq_cmd;
    // then compute PD + KDL gravity for the next step.
    auto ctrl_step = [&]() {
        mj_kdl::update(&arm1);
        mj_kdl::update(&arm2);
        for (int i = 0; i < n; ++i) q1(i) = arm1.jnt_pos_msr[i];
        for (int i = 0; i < n; ++i) q2(i) = arm2.jnt_pos_msr[i];
        dyn1.JntToGravity(q1, g1);
        dyn2.JntToGravity(q2, g2);
        for (int i = 0; i < n; ++i) {
            arm1.jnt_trq_cmd[i] =
              kKp[i] * (kHomePose[i] - arm1.jnt_pos_msr[i]) - kKd[i] * arm1.jnt_vel_msr[i] + g1(i);
            arm2.jnt_trq_cmd[i] =
              kKp[i] * (kHomePose[i] - arm2.jnt_pos_msr[i]) - kKd[i] * arm2.jnt_vel_msr[i] + g2(i);
        }
        if (fing1 >= 0) data->ctrl[fing1] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
        if (fing2 >= 0) data->ctrl[fing2] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
    };

    if (headless) {
        for (int step = 0; step < 600; ++step) {
            ctrl_step();
            mj_kdl::step(&arm1);
        }

        KDL::ChainFkSolverPos_recursive fk1(arm1.chain), fk2(arm2.chain);
        KDL::JntArray                   q1(n), q2(n);
        KDL::Frame                      ee1, ee2;
        for (int i = 0; i < n; ++i) {
            q1(i) = arm1.jnt_pos_msr[i];
            q2(i) = arm2.jnt_pos_msr[i];
        }
        fk1.JntToCart(q1, ee1);
        fk2.JntToCart(q2, ee2);
        std::cout << "arm1 EE: [" << ee1.p.x() << ", " << ee1.p.y() << ", " << ee1.p.z() << "]\n";
        std::cout << "arm2 EE: [" << ee2.p.x() << ", " << ee2.p.y() << ", " << ee2.p.z() << "]\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &arm1)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&arm1);
            mj_kdl::cleanup(&arm2);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        double prev_sim_time = data->time;
        while (true) {
            if (data->time < prev_sim_time - 1e-6) reset_to_home();
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::step(&arm1)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&arm1);
    mj_kdl::cleanup(&arm2);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
