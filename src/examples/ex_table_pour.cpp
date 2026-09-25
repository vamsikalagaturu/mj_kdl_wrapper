/* ex_table_pour.cpp
 * Kinova GEN3 + Robotiq 2F-85 pours small balls from a small attached bottle into a
 * transparent tabletop receiver.
 *
 * Usage:
 *   ex_table_pour [--headless] [--record output.mp4]
 *
 * Runs the full pour sequence once, prints how many balls ended in the receiver and exits;
 * --headless skips the viewer. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using mj_kdl_examples::kGripperClosed;

using mj_kdl_examples::kHomePose;
static constexpr double kTableZ           = 0.70;
static constexpr double kRobotBackX       = -0.26;
static constexpr double kJugX             = 0.30;
static constexpr double kJugY             = 0.14;
static constexpr double kRetreatX         = kJugX - 0.08;
static constexpr double kRetreatY         = kJugY - 0.08;
static constexpr double kJugRadius        = 0.028;
static constexpr double kJugHeight        = 0.084;
static constexpr int    kNumBallsGui      = 36;
static constexpr int    kNumBallsHeadless = kNumBallsGui;
static constexpr double kBallRadius       = 0.007;
static constexpr double kReceiverFrameZ   = kTableZ;
static constexpr double kIkTol            = 3e-3;
static constexpr double kPourTiltRad      = 1.95;
static constexpr double kTiltOutletZ      = kTableZ + 0.18;

static constexpr double kKp[7] = { 120, 220, 120, 220, 110, 190, 90 };
static constexpr double kKd[7] = { 12, 22, 12, 22, 11, 18, 9 };

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

static mj_kdl::SceneObject make_ball(int idx)
{
    char name[32];
    std::snprintf(name, sizeof(name), "grain_%02d", idx);
    return {
        .name      = name,
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::SPHERE,
        .size      = { kBallRadius, 0.0, 0.0 },
        .pos       = { 0.0, 0.0, kTableZ + 0.40 + idx * 2.0 * kBallRadius },
        .rgba      = { 1.0f, 0.84f, 0.30f, 1.0f },
        .mass      = 0.006,
        .condim    = mj_kdl::Condim::Torsional,
        .friction  = { 0.5, 0.02, 0.001 },
    };
}

static bool inside_jug(const mjData *data, const mjModel *model, int joint_id)
{
    const int     qadr = model->jnt_qposadr[joint_id];
    const double *p    = data->qpos + qadr;
    return std::abs(p[0] - kJugX) < (kJugRadius - 0.012)
           && std::abs(p[1] - kJugY) < (kJugRadius - 0.012) && p[2] > kTableZ + 0.006
           && p[2] < kTableZ + kJugHeight + 0.04;
}

int main(int argc, char *argv[])
{
    const mj_kdl_examples::Args args = mj_kdl_examples::parse_args(argc, argv, "table_pour.mp4");
    const bool                  headless = args.headless;

    const int num_balls = headless ? kNumBallsHeadless : kNumBallsGui;

    const std::string arm_mjcf    = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf    = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string bottle_mjcf = mj_kdl_examples::asset("mug.xml");
    const std::string receiver_mjcf = mj_kdl_examples::asset("mug_table.xml");
    const std::string table_mjcf    = mj_kdl_examples::asset("table.xml");

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gripper.prefix    = "g_";

    mj_kdl::AttachmentSpec bottle;
    bottle.mjcf_path = bottle_mjcf.c_str();
    bottle.attach_to = { mj_kdl::AttachKind::Body, "g_base" };
    bottle.prefix    = "pour_";
    bottle.pos[0]    = 0.0;
    bottle.pos[1]    = 0.0;
    bottle.pos[2]    = 0.0;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = arm_mjcf.c_str();
    robot_spec.pos[0] = kRobotBackX;
    robot_spec.pos[2] = kTableZ;
    robot_spec.attachments.push_back(gripper);
    robot_spec.attachments.push_back(bottle);

    mj_kdl::SceneSpec   scene_cfg;
    scene_cfg.timestep   = 0.002;
    scene_cfg.add_floor  = true;
    scene_cfg.add_skybox = true;
    mj_kdl::SceneObject table{
        .name      = "table",
        .mjcf_path = table_mjcf,
        .pos       = { 0.0, 0.0, kTableZ },
        .fixed     = true,
    };
    scene_cfg.objects.push_back(table);
    for (int i = 0; i < num_balls; ++i) scene_cfg.objects.push_back(make_ball(i));
    scene_cfg.objects.push_back(mj_kdl::SceneObject{
      .name      = "recv",
      .mjcf_path = receiver_mjcf,
      .pos       = { kJugX, kJugY, kReceiverFrameZ },
    });
    scene_cfg.robots.push_back(robot_spec);

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &scene_cfg)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }
    const mjModel *model = env.model;
    const mjData  *data  = env.data;

    KDL::Frame        world_T_table_top;
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

    const unsigned             n       = robot.chain.getNrOfJoints();
    mj_kdl::SceneActuatorSlot *fingers = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    if (!fingers) {
        std::cerr << "g_fingers_actuator not found\n";
        return 1;
    }

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::JntArray                   q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        const auto [lo, hi] = robot.joint_limits[i];
        if (std::isfinite(lo) && std::isfinite(hi)) {
            q_min(i) = lo;
            q_max(i) = hi;
        } else {
            q_min(i) = -2 * M_PI;
            q_max(i) = 2 * M_PI;
        }
    }
    KDL::ChainIkSolverVel_pinv  ik_vel(robot.chain);
    KDL::ChainIkSolverPos_NR_JL ik_nr(robot.chain, q_min, q_max, fk, ik_vel, 2000, 1e-5);
    KDL::ChainIkSolverPos_LMA   ik_lma(robot.chain, 1e-5, 2000);
    KDL::ChainDynParam          dyn(robot.chain, KDL::Vector(0.0, 0.0, scene_cfg.gravity_z));

    KDL::Frame home_fk;
    fk.JntToCart(q_home, home_fk);
    const KDL::Rotation carry_tcp = home_fk.M * KDL::Rotation::RotY(-0.05);

    KDL::JntArray q_pre_pour(n), q_pour(n), q_tilt(n), q_retreat(n);
    struct Waypoint
    {
        const char          *name;
        KDL::Frame           target;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    const KDL::Frame world_T_base(
      KDL::Rotation::Identity(), KDL::Vector(kRobotBackX, 0.0, kTableZ)
    );
    const KDL::Frame base_T_world = world_T_base.Inverse();

    mj_kdl::set_joint_pos(&robot, q_home);
    KDL::Frame world_T_outlet, world_T_tcp;
    mj_kdl::get_site_frame(&env, "pour_outlet", &world_T_outlet);
    mj_kdl::get_site_frame(&env, "g_pinch", &world_T_tcp);
    const KDL::Vector tcp_outlet = world_T_tcp.Inverse() * world_T_outlet.p;

    const auto outlet_target_to_tcp_target =
      [&](const KDL::Rotation &tcp_rot, const KDL::Vector &outlet_pos) {
          return KDL::Frame(tcp_rot, outlet_pos - tcp_rot * tcp_outlet);
      };

    std::array<KDL::Vector, 3> waypoint_pos = {
        KDL::Vector(kJugX, kJugY, kTableZ + 0.27),
        KDL::Vector(kJugX, kJugY, kTableZ + 0.20),
        KDL::Vector(kRetreatX, kRetreatY, kTableZ + 0.27),
    };
    const auto solve_waypoints = [&](const std::array<KDL::Vector, 3> &pos) {
        Waypoint waypoints[] = {
            { "pre-pour",
              base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[0]),
              &q_pre_pour,
              &q_home },
            { "pour",
              base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[1]),
              &q_pour,
              &q_pre_pour },
            { "retreat",
              base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[2]),
              &q_retreat,
              &q_pour },
        };
        for (const auto &wp : waypoints) {
            bool ok = ik_nr.CartToJnt(*wp.seed, wp.target, *wp.out) >= 0;
            if (!ok) ok = ik_lma.CartToJnt(*wp.seed, wp.target, *wp.out) >= 0;
            if (!ok) {
                std::cerr << "IK failed for " << wp.name << "\n";
                return false;
            }
            KDL::Frame fk_out;
            fk.JntToCart(*wp.out, fk_out);
            if ((wp.target.p - fk_out.p).Norm() > kIkTol) {
                std::cerr << "IK pose error for " << wp.name << "\n";
                return false;
            }
        }
        return true;
    };
    if (!solve_waypoints(waypoint_pos)) return 1;
    q_tilt = q_pour;
    q_tilt(n - 1) += kPourTiltRad;
    for (int iter = 0; iter < 4; ++iter) {
        mj_kdl::set_joint_pos(&robot, q_tilt);
        mj_kdl::get_site_frame(&env, "pour_outlet", &world_T_outlet);
        const double dx = kJugX - world_T_outlet.p.x();
        const double dy = kJugY - world_T_outlet.p.y();
        const double dz = kTiltOutletZ - world_T_outlet.p.z();
        if (std::sqrt(dx * dx + dy * dy + dz * dz) < 5e-3) break;

        waypoint_pos[1][0] += dx;
        waypoint_pos[1][1] += dy;
        waypoint_pos[1][2] += dz;
        if (!solve_waypoints(waypoint_pos)) return 1;
        q_tilt = q_pour;
        q_tilt(n - 1) += kPourTiltRad;
    }

    std::vector<int> grain_joints;
    grain_joints.reserve(num_balls);
    for (int i = 0; i < num_balls; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "grain_%02d_joint", i);
        int jid = mj_name2id(model, mjOBJ_JOINT, name);
        if (jid >= 0) grain_joints.push_back(jid);
    }

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    /* Scene-specific reset: place balls inside bottle and close gripper.
     * Env::on_reset runs after mj_resetData and before final mj_forward/robot sync. */
    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, q_home);

        KDL::Frame world_T_center;
        mj_kdl::get_site_frame(&env, "pour_center", &world_T_center);

        const double spacing = 2.00 * kBallRadius;
        for (int i = 0; i < num_balls; ++i) {
            const int    layer = i / 9;
            const int    slot  = i % 9;
            const double ix    = static_cast<double>(slot % 3) - 1.0;
            const double iy    = static_cast<double>(slot / 3) - 1.0;
            KDL::Vector  world_v =
              world_T_center * KDL::Vector(ix * spacing, iy * spacing, -0.026 + layer * spacing);
            const double world[3] = { world_v.x(), world_v.y(), world_v.z() };
            char         body_name[32];
            std::snprintf(body_name, sizeof(body_name), "grain_%02d", i);
            mj_kdl::set_body_pose(&env, body_name, world);
        }
        ctx->data->ctrl[fingers->ctrl_id] = kGripperClosed;
        restart                           = true;
    };

    const std::vector<mj_kdl_examples::Phase> phases = {
        { .name        = "HOME",
          .target      = &q_home,
          .duration    = 1.0,
          .timeout     = 2.5,
          .settle_tol  = 0.08,
          .gripper_cmd = kGripperClosed },
        { .name        = "PRE_POUR",
          .target      = &q_pre_pour,
          .duration    = 4.0,
          .timeout     = 6.5,
          .settle_tol  = 0.08,
          .gripper_cmd = kGripperClosed },
        { .name        = "POUR",
          .target      = &q_pour,
          .duration    = 3.5,
          .timeout     = 5.5,
          .settle_tol  = 0.07,
          .gripper_cmd = kGripperClosed },
        { .name        = "TILT",
          .target      = &q_tilt,
          .duration    = 7.0,
          .timeout     = 10.0,
          .settle_tol  = 0.07,
          .gripper_cmd = kGripperClosed },
        { .name        = "POUR_HOLD",
          .target      = &q_tilt,
          .duration    = headless ? 9.0 : 10.0,
          .timeout     = headless ? 10.0 : 11.0,
          .settle_tol  = -1.0,
          .gripper_cmd = kGripperClosed },
        { .name        = "RETREAT",
          .target      = &q_retreat,
          .duration    = 2.0,
          .timeout     = 4.0,
          .settle_tol  = 0.08,
          .gripper_cmd = kGripperClosed },
        { .name        = "HOLD",
          .target      = &q_retreat,
          .duration    = 1.0,
          .timeout     = 1.0,
          .settle_tol  = -1.0,
          .gripper_cmd = kGripperClosed },
    };

    mj_kdl::VideoRecorder recorder;
    bool                  recorder_ok = false;
    const int             kRecordFps  = 60;
    const int             steps_per_frame =
      std::max(1, static_cast<int>(1.0 / (kRecordFps * model->opt.timestep)));
    int sim_step = 0;
    if (args.record) {
        if (!mj_kdl::init_video_recorder(
              &recorder,
              env.model,
              args.record_path.c_str(),
              mj_kdl::VideoResolution::R1080p,
              kRecordFps
            )) {
            std::cerr << "init_video_recorder() failed -- is EGL available and ffmpeg installed?\n";
            return 1;
        }
        recorder.cam.azimuth   = 145.0;
        recorder.cam.elevation = -22.0;
        recorder.cam.distance  = 1.35;
        recorder.cam.lookat[0] = 0.05;
        recorder.cam.lookat[1] = 0.02;
        recorder.cam.lookat[2] = 0.88;
        recorder_ok            = true;
    }

    if (!headless && !mj_kdl::open_viewer(&env)) {
        std::cerr << "open_viewer() failed\n";
        return 1;
    }

    mj_kdl::reset(&env);

    const bool completed = mj_kdl_examples::run_phases(
      env,
      robot,
      fingers,
      phases,
      restart,
      [&](const KDL::JntArray &q_des) { impedance_ctrl(robot, q_des, n, dyn); },
      [&] {
          ++sim_step;
          if (recorder_ok && sim_step % steps_per_frame == 0) {
              if (!mj_kdl::record_frame(&recorder, &env)) {
                  std::cerr << "record_frame() failed at step " << sim_step << "\n";
                  mj_kdl::cleanup(&recorder);
                  recorder_ok = false;
              }
          }
      }
    );

    if (recorder_ok) {
        mj_kdl::cleanup(&recorder);
        std::cout << "Saved recording: " << args.record_path << "\n";
    }

    int ret = 0;
    if (completed) {
        int    in_jug = 0;
        double avg[3] = {};
        for (int jid : grain_joints)
            if (inside_jug(data, model, jid)) ++in_jug;
        for (int jid : grain_joints) {
            const double *p = data->qpos + model->jnt_qposadr[jid];
            avg[0] += p[0];
            avg[1] += p[1];
            avg[2] += p[2];
        }
        if (!grain_joints.empty()) {
            avg[0] /= static_cast<double>(grain_joints.size());
            avg[1] /= static_cast<double>(grain_joints.size());
            avg[2] /= static_cast<double>(grain_joints.size());
        }

        std::cout << "balls in transparent receiver: " << in_jug << "/" << grain_joints.size()
                  << "\n";
        std::cout << "grain centroid: [" << std::fixed << std::setprecision(3) << avg[0] << ", "
                  << avg[1] << ", " << avg[2] << "] receiver center=[" << kJugX << ", " << kJugY
                  << "]\n";
        if (headless && in_jug < 4) {
            std::cerr << "pour failed: too few balls reached the receiver\n";
            ret = 1;
        }
    }

    mj_kdl::cleanup(&env);
    return ret;
}
