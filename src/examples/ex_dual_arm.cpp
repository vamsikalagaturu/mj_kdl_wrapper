/* ex_dual_arm.cpp
 * Two Kinova Gen3 arms, each fitted with a Robotiq 2F-85 gripper,
 * in a shared MuJoCo scene.
 *
 * arm1 at x = -1.0 m, facing +X.
 * arm2 at x = +1.0 m, facing +X; all element names prefixed "r2_".
 *
 * Both arms hold the home pose via PD + KDL gravity compensation.
 * Grippers cycle open/closed every 3 s.
 *
 * Gravity comp uses KDL::ChainDynParam built from a chain that includes
 * the gripper's lumped inertia (via init_robot_from_mjcf tool_body).
 * jnt_trq_cmd is primed before the loop so the first physics step already
 * gets correct compensation.
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_dual_arm [--headless]
 *
 * Runs 600 steps and exits; --headless skips the viewer and prints both EE positions. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <iostream>
#include <string>

using mj_kdl_examples::kHomePose;

static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };
static constexpr int    kSteps = 600;

using mj_kdl_examples::kGripperClosed;

int main(int argc, char *argv[])
{
    const bool headless = mj_kdl_examples::parse_args(argc, argv).headless;

    const std::string arm_mjcf = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gs.prefix    = "g_";

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
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    sc.robots.push_back(arm1_spec);
    sc.robots.push_back(arm2_spec);

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &sc)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }

    mj_kdl::ToolFrameSpec tool1, tool2;
    tool1.tool_body = "g_base_mount";
    tool1.tcp_site  = "g_pinch";
    tool2.tool_body = "r2_g_base_mount";
    tool2.tcp_site  = "r2_g_pinch";

    mj_kdl::Robot arm1, arm2;
    if (
      !mj_kdl::init_robot_from_mjcf(&arm1, &env, "base_link", "bracelet_link", "", &tool1)
      || !mj_kdl::init_robot_from_mjcf(
        &arm2, &env, "r2_base_link", "r2_bracelet_link", "", &tool2
      )
    ) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    const int                  n     = arm1.n_joints;
    mj_kdl::SceneActuatorSlot *fing1 = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    mj_kdl::SceneActuatorSlot *fing2 =
      mj_kdl::bind_scene_actuator(&env.scene, "r2_g_fingers_actuator");
    if (!fing1 || !fing2) return 1;

    KDL::ChainDynParam dyn1(arm1.chain, KDL::Vector(0.0, 0.0, -9.81));
    KDL::ChainDynParam dyn2(arm2.chain, KDL::Vector(0.0, 0.0, -9.81));

    KDL::JntArray q_home(n), q1(n), q2(n), g1(n), g2(n);
    for (int i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    if (
      !mj_kdl::set_control_mode(&arm1, mj_kdl::CtrlMode::TORQUE)
      || !mj_kdl::set_control_mode(&arm2, mj_kdl::CtrlMode::TORQUE)
    )
        return 1;

    // Prime jnt_trq_cmd so the first physics step already gets gravity compensation.
    auto prime_grav = [&]() {
        dyn1.JntToGravity(q_home, g1);
        dyn2.JntToGravity(q_home, g2);
        for (int i = 0; i < n; ++i) {
            arm1.jnt_trq_cmd[i] = g1(i);
            arm2.jnt_trq_cmd[i] = g2(i);
        }
    };

    // reset() seeds the finger slots from ctrl, so the hook sets ctrl.
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&arm1, q_home);
        mj_kdl::set_joint_pos(&arm2, q_home);
        ctx->data->ctrl[fing1->ctrl_id] = kGripperClosed;
        ctx->data->ctrl[fing2->ctrl_id] = kGripperClosed;
        prime_grav();
    };

    mj_kdl::reset(&env);
    mj_kdl::update(&env);

    // Per-step: update() reads sensors and flushes the previous jnt_trq_cmd;
    // then compute PD + KDL gravity for the next step.
    auto ctrl_step = [&]() {
        mj_kdl::update(&env);
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
        fing1->command = (std::fmod(env.data->time, 6.0) < 3.0) ? kGripperClosed : 0.0;
        fing2->command = fing1->command;
    };

    if (headless) {
        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            mj_kdl::step(&env);
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
        if (!mj_kdl::open_viewer(&env)) {
            std::cerr << "open_viewer() failed\n";
            return 1;
        }
        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            if (!mj_kdl::step(&env)) break;
            mj_kdl::pace_realtime(&env);
        }
    }

    mj_kdl::cleanup(&env);
    return 0;
}
