/* ex_pick.cpp
 * Scripted pick: Kinova GEN3 + Robotiq 2F-85 picks an orange cube from the floor and
 * lifts it.
 *
 * Joint impedance control law applied every step:
 *   tau_i = g_kdl_i + Kp*(q_des_i - q_i) - Kd*dq_i
 * where g_kdl is computed via KDL::ChainDynParam::JntToGravity (includes gripper inertia).
 *
 * Phases: HOME -> PREGRASP -> GRASP -> CLOSE -> LIFT -> HOLD
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_pick [--headless]
 *
 * With --headless runs the full pick sequence and prints final cube height. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using mj_kdl_examples::kGripperClosed;

// scene constants
using mj_kdl_examples::kHomePose;
static constexpr double kCubeX       = 0.4;
static constexpr double kCubeY       = 0.0;
static constexpr double kCubeHS      = 0.02; // half-size of 4 cm cube
static constexpr double kCubeZ       = kCubeHS;

// impedance gains (per joint, matching ex_impedance tuning)
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 }; // Nm/rad
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };        // Nm*s/rad

/*
 * Compute joint impedance torques into robot.jnt_trq_cmd:
 *   tau_i = g_kdl[i] + Kp[i]*(q_des_i - q_msr_i) - Kd[i]*dq_msr_i
 * dyn must have been built from the chain returned by init_robot_from_mjcf with tool_body set.
 */
static void impedance_ctrl(
  mj_kdl::Robot       &robot,
  const KDL::JntArray &q_des,
  unsigned             n,
  KDL::ChainDynParam  &dyn
)
{
    KDL::JntArray q(n), g(n);
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) {
        robot.jnt_trq_cmd[i] =
          g(i) + kKp[i] * (q_des(i) - robot.jnt_pos_msr[i]) - kKd[i] * robot.jnt_vel_msr[i];
    }
}

int main(int argc, char *argv[])
{
    const bool headless = mj_kdl_examples::parse_args(argc, argv).headless;

    // scene setup
    const std::string arm_mjcf = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf;
    gs.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gs.prefix    = "g_";

    mj_kdl::RobotSpec rs;
    rs.path = arm_mjcf;
    rs.attachments.push_back(gs);

    mj_kdl::SceneObject cube{
        .name      = "cube",
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::BOX,
        .size      = { kCubeHS, kCubeHS, kCubeHS },
        .pos       = { kCubeX, kCubeY, kCubeZ },
        .rgba      = { 1.0f, 0.5f, 0.0f, 1.0f },
        .mass      = 0.1,
        .condim    = mj_kdl::Condim::Torsional,
        .friction  = { 0.8, 0.02, 0.001 },
    };

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    sc.robots.push_back(rs);
    sc.objects.push_back(cube);

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &sc)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }
    const mjModel *model = env.model;

    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base_mount";
    tool.tcp_site  = "g_pinch";

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool)) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    unsigned                   n       = robot.chain.getNrOfJoints();
    mj_kdl::SceneActuatorSlot *fingers = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    int                        cube_jnt = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    int                        key_id   = mj_name2id(model, mjOBJ_KEY, "home");
    if (!fingers) return 1;

    KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    // IK setup
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainIkSolverVel_wdls      ik_vel(robot.chain, 1e-5, 150);
    ik_vel.setLambda(0.05);

    KDL::JntArray q_home(n), q_pregrasp(n), q_grasp(n), q_lift(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    const double        kGraspZ    = kCubeZ;
    const double        kPreGraspZ = kGraspZ + 0.20;
    const double        kLiftZ     = kGraspZ + 0.30;
    const KDL::Rotation kGraspRot  = robot.tip_T_tcp.M;

    struct WP
    {
        double               z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    WP wps[] = {
        { kPreGraspZ, &q_pregrasp, &q_home },
        { kGraspZ, &q_grasp, &q_pregrasp },
        { kLiftZ, &q_lift, &q_grasp },
    };
    for (auto &wp : wps) {
        KDL::Frame target(kGraspRot, KDL::Vector(kCubeX, kCubeY, wp.z));
        if (!mj_kdl_examples::solve_near_seed(ik_vel, fk, robot, *wp.seed, target, *wp.out)) {
            std::cerr << "IK failed for z=" << wp.z << "\n";
            return 1;
        }
    }

    const double kHoldDuration = 1.0;

    // clang-format off
    const std::vector<mj_kdl_examples::Phase> phases = {
        { "HOME",     &q_home,     1.0,           2.5,            0.08, 0.0            },
        { "PREGRASP", &q_pregrasp, 5.0,           7.0,            0.08, 0.0            },
        { "GRASP",    &q_grasp,    5.0,           8.0,            0.03, 0.0            },
        { "CLOSE",    &q_grasp,    1.5,           2.5,           -1.0,  kGripperClosed },
        { "LIFT",     &q_lift,     3.0,           5.0,            0.08, kGripperClosed },
        { "HOLD",     &q_lift,     kHoldDuration, kHoldDuration, -1.0,  kGripperClosed },
    };
    // clang-format on

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    mj_kdl::ResetOptions reset_opts;
    reset_opts.use_keyframe = key_id >= 0;
    reset_opts.keyframe     = key_id >= 0 ? key_id : 0;

    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        if (key_id < 0) mj_kdl::set_joint_pos(&robot, q_home);
        double *cube_q = ctx->data->qpos + model->jnt_qposadr[cube_jnt];
        cube_q[0]      = kCubeX;
        cube_q[1]      = kCubeY;
        cube_q[2]      = kCubeZ;
        cube_q[3]      = 1.0;
        cube_q[4] = cube_q[5] = cube_q[6] = 0.0;

        ctx->data->ctrl[fingers->ctrl_id] = 0.0;
        restart                           = true;
    };

    mj_kdl::reset(&env, &reset_opts);

    if (!headless && !mj_kdl::open_viewer(&env)) {
        std::cerr << "open_viewer() failed\n";
        return 1;
    }

    mj_kdl_examples::run_phases(
      env,
      robot,
      fingers,
      phases,
      restart,
      [&](const KDL::JntArray &q_des) { impedance_ctrl(robot, q_des, n, dyn); }
    );

    if (headless) {
        int    qadr   = model->jnt_qposadr[cube_jnt];
        double cube_z = env.data->qpos[qadr + 2];
        std::cout << "cube Z after pick: " << std::fixed << std::setprecision(3) << cube_z
                  << " m\n";
    }

    mj_kdl::cleanup(&env);
    return 0;
}
