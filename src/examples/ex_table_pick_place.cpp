/* ex_table_pick_place.cpp
 * Kinova GEN3 + Robotiq 2F-85 picks a cube from one table location and
 * places it at another.
 *
 * Usage:
 *   ex_table_pick_place [--headless]
 *
 * Runs the full sequence once, prints the final cube position and exits; --headless skips
 * the viewer. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using mj_kdl_examples::kGripperClosed;

using mj_kdl_examples::kHomePose;
static constexpr double kCubeHS      = 0.02;
static constexpr double kPickX       = 0.40;
static constexpr double kPickY       = 0.00;
static constexpr double kPlaceX      = 0.40;
static constexpr double kPlaceY      = 0.24;
static constexpr double kTableZ      = 0.70;

static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

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

static mj_kdl::SceneObject make_cube(double surface_z)
{
    return {
        .name      = "cube",
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::BOX,
        .size      = { kCubeHS, kCubeHS, kCubeHS },
        .pos       = { kPickX, kPickY, surface_z + kCubeHS },
        .rgba      = { 0.1f, 0.35f, 1.0f, 1.0f },
        .mass      = 0.1,
        .condim    = mj_kdl::Condim::Torsional,
        .friction  = { 0.8, 0.02, 0.001 },
    };
}

int main(int argc, char *argv[])
{
    const bool headless = mj_kdl_examples::parse_args(argc, argv).headless;

    const std::string arm_mjcf = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string table_mjcf = mj_kdl_examples::asset("table.xml");

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gripper.prefix    = "g_";

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = arm_mjcf.c_str();
    robot_spec.pos[2] = kTableZ;
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.timestep   = 0.002;
    scene.add_floor  = true;
    scene.add_skybox = true;
    scene.robots.push_back(robot_spec);
    mj_kdl::SceneObject table{
        .name      = "table",
        .mjcf_path = table_mjcf,
        .pos       = { 0.0, 0.0, kTableZ },
        .fixed     = true,
    };
    scene.objects.push_back(table);
    scene.objects.push_back(make_cube(kTableZ));

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &scene)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }
    const mjModel *model = env.model;
    mjData        *data  = env.data;

    KDL::Frame world_T_table_top;
    const std::string table_top_site = mj_kdl::scene_object_site_name(table, "table_top");
    if (!mj_kdl::get_site_frame(&env, table_top_site.c_str(), &world_T_table_top)) {
        std::cerr << "table_top site not found\n";
        return 1;
    }

    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base_mount";
    tool.tcp_site  = "g_pinch";

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool)) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    const unsigned             n        = robot.chain.getNrOfJoints();
    mj_kdl::SceneActuatorSlot *fingers  = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    const int                  cube_jnt = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    if (!fingers || cube_jnt < 0) {
        std::cerr << "required actuator or cube joint not found\n";
        return 1;
    }

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainIkSolverVel_wdls      ik_vel(robot.chain, 1e-5, 150);
    ik_vel.setLambda(0.05);
    KDL::ChainDynParam  dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
    const KDL::Rotation kGraspRot = robot.tip_T_tcp.M;

    const double z_grasp = kCubeHS;
    const double z_above = z_grasp + 0.20;
    const double z_lift  = z_grasp + 0.30;

    KDL::JntArray q_pick_above(n), q_pick(n), q_lift(n), q_place_above(n), q_place(n);
    struct Waypoint
    {
        double               world_x;
        double               world_y;
        double               world_z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    Waypoint waypoints[] = {
        { kPickX, kPickY, kTableZ + z_above, &q_pick_above, &q_home },
        { kPickX, kPickY, kTableZ + z_grasp, &q_pick, &q_pick_above },
        { kPickX, kPickY, kTableZ + z_lift, &q_lift, &q_pick },
        { kPlaceX, kPlaceY, kTableZ + z_above, &q_place_above, &q_lift },
        { kPlaceX, kPlaceY, kTableZ + z_grasp, &q_place, &q_place_above },
    };
    KDL::Frame world_T_base(KDL::Rotation::Identity(), KDL::Vector(0.0, 0.0, kTableZ));
    KDL::Frame base_T_world = world_T_base.Inverse();
    for (const auto &wp : waypoints) {
        KDL::Frame world_target(kGraspRot, KDL::Vector(wp.world_x, wp.world_y, wp.world_z));
        if (!mj_kdl_examples::solve_near_seed(
              ik_vel, fk, robot, *wp.seed, base_T_world * world_target, *wp.out
            )) {
            std::cerr << "IK failed for waypoint at world [" << wp.world_x << ", " << wp.world_y
                      << ", " << wp.world_z << "]\n";
            return 1;
        }
    }
    // clang-format off
    const std::vector<mj_kdl_examples::Phase> phases = {
        { "HOME",        &q_home,        1.0, 2.5,  0.08, 0.0            },
        { "PICK_ABOVE",  &q_pick_above,  5.0, 7.0,  0.08, 0.0            },
        { "PICK",        &q_pick,        5.0, 8.0,  0.03, 0.0            },
        { "CLOSE",       &q_pick,        1.5, 2.5, -1.0,  kGripperClosed },
        { "LIFT",        &q_lift,        3.0, 5.0,  0.08, kGripperClosed },
        { "PLACE_ABOVE", &q_place_above, 3.0, 5.0,  0.08, kGripperClosed },
        { "PLACE",       &q_place,       5.0, 8.0,  0.03, kGripperClosed },
        { "OPEN",        &q_place,       1.0, 2.0, -1.0,  0.0            },
        { "RETREAT",     &q_place_above, 2.0, 4.0,  0.08, 0.0            },
        { "HOLD",        &q_place_above, 1.0, 1.0, -1.0,  0.0            },
    };
    // clang-format on

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;
    int qadr = model->jnt_qposadr[cube_jnt];

    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home);
        data->qpos[qadr]     = kPickX;
        data->qpos[qadr + 1] = kPickY;
        data->qpos[qadr + 2] = kTableZ + kCubeHS;
        data->qpos[qadr + 3] = 1.0;
        data->qpos[qadr + 4] = data->qpos[qadr + 5] = data->qpos[qadr + 6] = 0.0;

        data->ctrl[fingers->ctrl_id] = 0.0;
        restart                      = true;
    };
    mj_kdl::reset(&env);

    if (!headless && !mj_kdl::open_viewer(&env)) {
        std::cerr << "open_viewer() failed\n";
        return 1;
    }

    const bool completed = mj_kdl_examples::run_phases(
      env,
      robot,
      fingers,
      phases,
      restart,
      [&](const KDL::JntArray &q_des) { impedance_ctrl(robot, q_des, n, dyn); }
    );

    int ret = 0;
    if (completed) {
        const bool closed   = phases.back().gripper_cmd > 0.0;
        double cube_x       = data->qpos[qadr];
        double cube_y       = data->qpos[qadr + 1];
        double cube_z       = data->qpos[qadr + 2];
        double place_err_xy = std::hypot(cube_x - kPlaceX, cube_y - kPlaceY);
        std::cout << "cube final position: [" << std::fixed << std::setprecision(3) << cube_x
                  << ", " << cube_y << ", " << cube_z << "]"
                  << " target=[" << kPlaceX << ", " << kPlaceY << ", " << kTableZ + kCubeHS
                  << "] xy_error=" << place_err_xy << " gripper=" << (closed ? "closed" : "open")
                  << "\n";
        if (headless && place_err_xy > 0.08) ret = 1;
    }
    mj_kdl::cleanup(&env);
    return ret;
}
