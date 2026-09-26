/* ex_table_pour.cpp
 * Kinova GEN3 + Robotiq 2F-85 pours small balls from a small attached bottle into a
 * transparent tabletop receiver, with joint impedance (PD + KDL gravity) in TORQUE mode.
 *
 * Usage:
 *   ex_table_pour [--headless] [--record output.mp4]
 *
 * Runs the pour sequence once and prints how many balls ended in the receiver; --headless skips
 * the viewer and exits 1 if fewer than kMinInReceiver did; --record writes an MP4 offscreen
 * (EGL + ffmpeg) and implies --headless. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace ex = mj_kdl_examples;

static constexpr double kRobotBackX    = -0.26;
static constexpr double kJugX          = 0.30;
static constexpr double kJugY          = 0.14;
static constexpr double kRetreatX      = kJugX - 0.08;
static constexpr double kRetreatY      = kJugY - 0.08;
static constexpr double kJugRadius     = 0.028;
static constexpr double kJugHeight     = 0.084;
static constexpr int    kNumBalls      = 36;
static constexpr double kBallRadius    = 0.007;
static constexpr double kIkTol         = 3e-3;
static constexpr double kPourTiltRad   = 1.95;
static constexpr double kTiltOutletZ   = ex::kTableZ + 0.18;
static constexpr int    kMinInReceiver = 24;
static constexpr int    kRecordFps     = 60;

static constexpr double kKp[7] = { 120, 220, 120, 220, 110, 190, 90 };
static constexpr double kKd[7] = { 12, 22, 12, 22, 11, 18, 9 };

static mj_kdl::SceneObject make_ball(int idx)
{
    mj_kdl::SceneObject ball;
    char                name[32];
    std::snprintf(name, sizeof(name), "grain_%02d", idx);
    ball.name    = name;
    ball.shape   = mj_kdl::Shape::SPHERE;
    ball.size[0] = kBallRadius;
    ball.size[1] = ball.size[2] = 0.0;
    ball.pos[2]                 = ex::kTableZ + 0.40 + idx * 2.0 * kBallRadius;
    const float rgba[4]         = { 1.0f, 0.84f, 0.30f, 1.0f };
    std::copy(rgba, rgba + 4, ball.rgba);
    ball.mass        = 0.006;
    ball.condim      = mj_kdl::Condim::Torsional;
    ball.friction[0] = 0.5;
    ball.friction[1] = 0.02;
    ball.friction[2] = 0.001;
    return ball;
}

static bool inside_receiver(const KDL::Vector &p)
{
    return std::abs(p.x() - kJugX) < (kJugRadius - 0.012)
           && std::abs(p.y() - kJugY) < (kJugRadius - 0.012) && p.z() > ex::kTableZ + 0.006
           && p.z() < ex::kTableZ + kJugHeight + 0.04;
}

int main(int argc, char *argv[])
{
    const ex::Args args     = ex::parse_args(argc, argv, "table_pour.mp4");
    const bool     headless = args.headless;

    mj_kdl::AttachmentSpec bottle;
    bottle.mjcf_path = ex::asset("mug.xml");
    bottle.attach_to = { mj_kdl::AttachKind::Body, "g_base" };
    bottle.prefix    = "pour_";

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = ex::menagerie_model("kinova_gen3/gen3.xml");
    robot_spec.pos[0] = kRobotBackX;
    robot_spec.pos[2] = ex::kTableZ;
    robot_spec.attachments.push_back(ex::gripper_attachment(ex::asset("robotiq_2f85/2f85.xml")));
    robot_spec.attachments.push_back(bottle);

    mj_kdl::SceneSpec scene_cfg = ex::scene_spec();
    scene_cfg.objects.push_back(ex::table_object(ex::asset("table.xml"), ex::kTableZ));
    for (int i = 0; i < kNumBalls; ++i) scene_cfg.objects.push_back(make_ball(i));
    mj_kdl::SceneObject receiver;
    receiver.name      = "recv";
    receiver.mjcf_path = ex::asset("mug_table.xml");
    receiver.pos[0]    = kJugX;
    receiver.pos[1]    = kJugY;
    receiver.pos[2]    = ex::kTableZ;
    scene_cfg.objects.push_back(receiver);
    scene_cfg.robots.push_back(robot_spec);

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &scene_cfg)) return 1;
    const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool))
        return 1;
    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;
    mj_kdl::SceneActuatorSlot *fingers =
      mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    if (!fingers) return 1;
    std::vector<mj_kdl::SceneFreeBodySlot *> grains;
    for (int i = 0; i < kNumBalls; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "grain_%02d", i);
        grains.push_back(mj_kdl::bind_scene_free_body(&env.scene, name));
        if (!grains.back()) return 1;
    }

    const unsigned                  n      = robot.chain.getNrOfJoints();
    const KDL::JntArray             q_home = ex::home_q(n);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::JntArray                   q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        const auto [lo, hi] = robot.joint_limits[i];
        q_min(i)            = std::isfinite(lo) ? lo : -2 * M_PI;
        q_max(i)            = std::isfinite(hi) ? hi : 2 * M_PI;
    }
    KDL::ChainIkSolverVel_pinv  ik_vel(robot.chain);
    KDL::ChainIkSolverPos_NR_JL ik_nr(robot.chain, q_min, q_max, fk, ik_vel, 2000, 1e-5);
    KDL::ChainIkSolverPos_LMA   ik_lma(robot.chain, 1e-5, 2000);
    KDL::ChainDynParam          dyn(robot.chain, KDL::Vector(0.0, 0.0, scene_cfg.gravity_z));

    KDL::Frame home_fk;
    fk.JntToCart(q_home, home_fk);
    const KDL::Rotation carry_tcp = home_fk.M * KDL::Rotation::RotY(-0.05);
    const KDL::Frame    base_T_world =
      KDL::Frame(KDL::Vector(kRobotBackX, 0.0, ex::kTableZ)).Inverse();

    // The TCP -> outlet offset, measured at home, turns an outlet target into a TCP target.
    mj_kdl::set_joint_pos(&robot, q_home);
    KDL::Frame world_T_outlet, world_T_tcp;
    mj_kdl::get_site_frame(&env, "pour_outlet", &world_T_outlet);
    mj_kdl::get_site_frame(&env, "g_pinch", &world_T_tcp);
    const KDL::Vector tcp_outlet = world_T_tcp.Inverse() * world_T_outlet.p;

    const auto solve =
      [&](
        const char *name, const KDL::Vector &outlet, const KDL::JntArray &seed, KDL::JntArray &out
      ) {
          const KDL::Frame target =
            base_T_world * KDL::Frame(carry_tcp, outlet - carry_tcp * tcp_outlet);
          bool ok = ik_nr.CartToJnt(seed, target, out) >= 0;
          if (!ok) ok = ik_lma.CartToJnt(seed, target, out) >= 0;
          KDL::Frame fk_out;
          fk.JntToCart(out, fk_out);
          if (!ok || (target.p - fk_out.p).Norm() > kIkTol) {
              std::cerr << "IK failed for " << name << "\n";
              return false;
          }
          return true;
      };

    KDL::JntArray q_pre_pour(n), q_pour(n), q_tilt(n), q_retreat(n);
    KDL::Vector   pour_outlet(kJugX, kJugY, ex::kTableZ + 0.20);
    if (!solve("pre-pour", KDL::Vector(kJugX, kJugY, ex::kTableZ + 0.27), q_home, q_pre_pour))
        return 1;
    if (!solve("pour", pour_outlet, q_pre_pour, q_pour)) return 1;
    // Tilting swings the outlet away; shift the pour pose until the tilted outlet is on target.
    for (int iter = 0; iter < 4; ++iter) {
        q_tilt = q_pour;
        q_tilt(n - 1) += kPourTiltRad;
        mj_kdl::set_joint_pos(&robot, q_tilt);
        mj_kdl::get_site_frame(&env, "pour_outlet", &world_T_outlet);
        const KDL::Vector err = KDL::Vector(kJugX, kJugY, kTiltOutletZ) - world_T_outlet.p;
        if (err.Norm() < 5e-3) break;
        pour_outlet += err;
        if (!solve("pour", pour_outlet, q_pre_pour, q_pour)) return 1;
    }
    q_tilt = q_pour;
    q_tilt(n - 1) += kPourTiltRad;
    if (!solve("retreat", KDL::Vector(kRetreatX, kRetreatY, ex::kTableZ + 0.27), q_pour, q_retreat))
        return 1;

    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, q_home);
        KDL::Frame world_T_center;
        mj_kdl::get_site_frame(&env, "pour_center", &world_T_center);
        const double spacing = 2.0 * kBallRadius;
        for (int i = 0; i < kNumBalls; ++i) {
            const int         layer = i / 9, slot = i % 9;
            const KDL::Vector local(
              (slot % 3 - 1) * spacing, (slot / 3 - 1) * spacing, -0.026 + layer * spacing
            );
            const KDL::Vector w      = world_T_center * local;
            const double      pos[3] = { w.x(), w.y(), w.z() };
            mj_kdl::set_body_pose(&env, grains[i]->name.c_str(), pos);
        }
        ctx->data->ctrl[fingers->ctrl_id] = ex::kGripperClosed;
        ex::prime_gravity(robot, dyn, q_home);
        restart = true;
    };

    const double close = ex::kGripperClosed;
    // clang-format off
    const std::vector<ex::Phase> phases = {
        { "HOME",      &q_home,     1.0,  2.5,  0.08, close },
        { "PRE_POUR",  &q_pre_pour, 4.0,  6.5,  0.08, close },
        { "POUR",      &q_pour,     3.5,  5.5,  0.07, close },
        { "TILT",      &q_tilt,     7.0, 10.0,  0.07, close },
        { "POUR_HOLD", &q_tilt,     9.0,  9.0, -1.0,  close },
        { "RETREAT",   &q_retreat,  2.0,  4.0,  0.08, close },
        { "HOLD",      &q_retreat,  1.0,  1.0, -1.0,  close },
    };
    // clang-format on

    mj_kdl::VideoRecorder recorder;
    bool                  recording = false;
    const int             steps_per_frame =
      std::max(1, static_cast<int>(std::lround(1.0 / (kRecordFps * env.model->opt.timestep))));
    if (args.record) {
        const mj_kdl::Status s = mj_kdl::init_video_recorder(
          &recorder,
          env.model,
          args.record_path.c_str(),
          mj_kdl::VideoResolution::R1080p,
          kRecordFps
        );
        if (!s) {
            std::cerr << "init_video_recorder() failed: " << s.error << "\n";
            return 1;
        }
        recorder.cam.azimuth   = 145.0;
        recorder.cam.elevation = -22.0;
        recorder.cam.distance  = 1.35;
        recorder.cam.lookat[0] = 0.05;
        recorder.cam.lookat[1] = 0.02;
        recorder.cam.lookat[2] = 0.88;
        recording              = true;
    }

    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    int        sim_step    = 0;
    bool       record_fail = false;
    const bool completed   = ex::run_phases(
      env,
      { { &robot, fingers } },
      phases,
      restart,
      [&](std::size_t, const KDL::JntArray &q_des) { ex::pd_gravity(robot, dyn, q_des, kKp, kKd); },
      [&](const ex::Phase &, double) {
          if (recording && ++sim_step % steps_per_frame == 0
              && !mj_kdl::record_frame(&recorder, &env)) {
              std::cerr << "record_frame() failed at step " << sim_step << "\n";
              mj_kdl::cleanup(&recorder);
              recording   = false;
              record_fail = true;
          }
      }
    );
    mj_kdl::update(&env);
    if (recording) {
        mj_kdl::cleanup(&recorder);
        std::cout << "Saved recording: " << args.record_path << "\n";
    }

    int         in_receiver = 0;
    KDL::Vector centroid    = KDL::Vector::Zero();
    for (const auto *g : grains) {
        if (inside_receiver(g->pose.p)) ++in_receiver;
        centroid += g->pose.p / kNumBalls;
    }
    std::cout << "balls in transparent receiver: " << in_receiver << "/" << kNumBalls
              << " (at least " << kMinInReceiver << ")\n"
              << std::fixed << std::setprecision(3) << "grain centroid: [" << centroid.x() << ", "
              << centroid.y() << ", " << centroid.z() << "] receiver center=[" << kJugX << ", "
              << kJugY << "]\n";
    mj_kdl::cleanup(&env);
    const bool ok = completed && !record_fail && in_receiver >= kMinInReceiver;
    return headless ? ex::verdict(ok, "the balls were poured into the receiver") : 0;
}
