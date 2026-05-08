/* ex_table_pour.cpp
 * Kinova GEN3 + Robotiq 2F-85 pours small balls from a small attached bottle into a
 * transparent tabletop receiver.
 *
 * Usage:
 *   ex_table_pour [--headless] [--record output.mp4]
 *
 * With --headless runs the full pour sequence and prints how many balls ended
 * in the receiver. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTableZ      = 0.70;
static constexpr double kRobotBackX  = -0.26;
static constexpr double kJugX        = 0.30;
static constexpr double kJugY        = 0.14;
static constexpr double kRetreatX    = kJugX - 0.08;
static constexpr double kRetreatY    = kJugY - 0.08;
static constexpr double kJugRadius   = 0.028;
static constexpr double kJugHeight   = 0.084;
static constexpr int    kNumBallsGui = 36;
static constexpr int    kNumBallsHeadless = kNumBallsGui;
static constexpr double kBallRadius  = 0.007;
static constexpr double kReceiverFrameZ = kTableZ;
static constexpr double kIkTol       = 3e-3;
static constexpr double kPourTiltRad = 1.95;
static constexpr double kTiltOutletZ = kTableZ + 0.18;

static constexpr double kKp[7] = { 120, 220, 120, 220, 110, 190, 90 };
static constexpr double kKd[7] = { 12, 22, 12, 22, 11, 18, 9 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double   clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

static void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

static void snapshot_q(const mj_kdl::Robot &robot, unsigned n, KDL::JntArray &q)
{
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
}

static double max_abs_joint_err(const mj_kdl::Robot &robot, const KDL::JntArray &q, unsigned n)
{
    double max_err = 0.0;
    for (unsigned i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(q(i) - robot.jnt_pos_msr[i]));
    return max_err;
}

static void
  impedance_ctrl(mj_kdl::Robot &robot, const KDL::JntArray &q_des, unsigned n, KDL::ChainDynParam &dyn)
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
    char                name[32];
    std::snprintf(name, sizeof(name), "grain_%02d", idx);
    return {
        .name      = name,
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::SPHERE,
        .size      = { kBallRadius, 0.0, 0.0 },
        .pos       = { 0.0, 0.0, kTableZ + 0.40 + idx * 2.0 * kBallRadius },
        .rgba      = { 1.0f, 0.84f, 0.30f, 1.0f },
        .mass      = 0.006,
        .condim    = 4,
        .friction  = { 0.5, 0.02, 0.001 },
    };
}

static bool inside_jug(const mjData *data, const mjModel *model, int joint_id)
{
    const int     qadr = model->jnt_qposadr[joint_id];
    const double *p    = data->qpos + qadr;
    return std::abs(p[0] - kJugX) < (kJugRadius - 0.012) &&
           std::abs(p[1] - kJugY) < (kJugRadius - 0.012) &&
           p[2] > kTableZ + 0.006 && p[2] < kTableZ + kJugHeight + 0.04;
}

struct Phase
{
    const char          *name;
    const KDL::JntArray *target;
    double               duration;
    double               timeout;
    double               settle_tol;
    double               gripper_cmd;
};

int main(int argc, char *argv[])
{
    bool headless = false;
    bool do_record = false;
    std::string record_path = "table_pour.mp4";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--record") {
            do_record = true;
            headless = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') record_path = argv[++i];
        }
    }
    const int num_balls = headless ? kNumBallsHeadless : kNumBallsGui;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string bottle_mjcf = (root / "src/examples/assets/mug.xml").string();
    const std::string receiver_mjcf = (root / "src/examples/assets/mug_table.xml").string();
    const std::string table_mjcf = (root / "src/examples/assets/table.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = "bracelet_link";
    gripper.prefix    = "g_";
    gripper.pos[2]    = -0.061525;
    gripper.euler[0]  = 180.0;

    mj_kdl::AttachmentSpec bottle;
    bottle.mjcf_path = bottle_mjcf.c_str();
    bottle.attach_to = "g_base";
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

    mj_kdl::SceneSpec scene_cfg;
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

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene_cfg)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    KDL::Frame world_T_table_top;
    const std::string table_top_site = mj_kdl::scene_object_site_name(table, "table_top");
    if (!mj_kdl::get_site_frame(model, data, table_top_site.c_str(), &world_T_table_top)) {
        std::cerr << "table_top site not found\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base";
    tool.tcp_site  = "g_pinch";

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", &tool
        )) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const unsigned n           = robot.chain.getNrOfJoints();
    const int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    if (fingers_act < 0) {
        std::cerr << "g_fingers_actuator not found\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::JntArray                   q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        int jid = model->dof_jntid[robot.kdl_to_mj_dof[i]];
        if (model->jnt_limited[jid]) {
            q_min(i) = model->jnt_range[2 * jid];
            q_max(i) = model->jnt_range[2 * jid + 1];
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

    mj_kdl::set_joint_pos(&robot, q_home, false);
    KDL::Frame world_T_outlet, world_T_tcp;
    mj_kdl::get_site_frame(model, data, "pour_outlet", &world_T_outlet);
    mj_kdl::get_site_frame(model, data, "g_pinch", &world_T_tcp);
    const KDL::Vector tcp_outlet = world_T_tcp.Inverse() * world_T_outlet.p;

    const auto outlet_target_to_tcp_target = [&](const KDL::Rotation &tcp_rot, const KDL::Vector &outlet_pos) {
        return KDL::Frame(tcp_rot, outlet_pos - tcp_rot * tcp_outlet);
    };

    std::array<KDL::Vector, 3> waypoint_pos = {
        KDL::Vector(kJugX,     kJugY,     kTableZ + 0.27),
        KDL::Vector(kJugX,     kJugY,     kTableZ + 0.20),
        KDL::Vector(kRetreatX, kRetreatY, kTableZ + 0.27),
    };
    const auto solve_waypoints = [&](const std::array<KDL::Vector, 3> &pos) {
        Waypoint waypoints[] = {
            { "pre-pour", base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[0]), &q_pre_pour, &q_home },
            { "pour",     base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[1]), &q_pour,     &q_pre_pour },
            { "retreat",  base_T_world * outlet_target_to_tcp_target(carry_tcp, pos[2]), &q_retreat,  &q_pour },
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
    if (!solve_waypoints(waypoint_pos)) {
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    q_tilt = q_pour;
    q_tilt(n - 1) += kPourTiltRad;
    for (int iter = 0; iter < 4; ++iter) {
        mj_kdl::set_joint_pos(&robot, q_tilt, false);
        mj_kdl::get_site_frame(model, data, "pour_outlet", &world_T_outlet);
        const double dx = kJugX - world_T_outlet.p.x();
        const double dy = kJugY - world_T_outlet.p.y();
        const double dz = kTiltOutletZ - world_T_outlet.p.z();
        if (std::sqrt(dx * dx + dy * dy + dz * dz) < 5e-3) break;

        waypoint_pos[1][0] += dx;
        waypoint_pos[1][1] += dy;
        waypoint_pos[1][2] += dz;
        if (!solve_waypoints(waypoint_pos)) {
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
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

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    /* Scene-specific reset: place balls inside bottle and close gripper.
     * Env::on_reset runs after mj_resetData and before final mj_forward/robot sync. */
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mjModel *m = ctx->model;
        mjData  *d = ctx->data;
        mj_kdl::set_joint_pos(&robot, q_home, false);

        KDL::Frame world_T_center;
        mj_kdl::get_site_frame(m, d, "pour_center", &world_T_center);

        const double spacing = 2.00 * kBallRadius;
        for (int i = 0; i < num_balls; ++i) {
            const int    layer = i / 16;
            const int    slot  = i % 16;
            const double ix    = static_cast<double>(slot % 4) - 1.5;
            const double iy    = static_cast<double>(slot) / 4 - 1.5;
            KDL::Vector  world_v =
              world_T_center * KDL::Vector(ix * spacing, iy * spacing, -0.026 + layer * spacing);
            const double world[3] = { world_v.x(), world_v.y(), world_v.z() };
            char body_name[32];
            std::snprintf(body_name, sizeof(body_name), "grain_%02d", i);
            mj_kdl::set_body_pose(m, d, body_name, world);
        }
        d->ctrl[fingers_act] = 255.0;
    };

    mj_kdl::reset(&env);

    const std::vector<Phase> phases = {
        { .name = "HOME",      .target = &q_home,     .duration = 1.0,                  .timeout = 2.5,                   .settle_tol =  0.08, .gripper_cmd = 255.0 },
        { .name = "PRE_POUR",  .target = &q_pre_pour, .duration = 4.0,                  .timeout = 6.5,                   .settle_tol =  0.08, .gripper_cmd = 255.0 },
        { .name = "POUR",      .target = &q_pour,     .duration = 3.5,                  .timeout = 5.5,                   .settle_tol =  0.07, .gripper_cmd = 255.0 },
        { .name = "TILT",      .target = &q_tilt,     .duration = 7.0,                  .timeout = 10.0,                  .settle_tol =  0.07, .gripper_cmd = 255.0 },
        { .name = "POUR_HOLD", .target = &q_tilt,     .duration = headless ? 9.0 : 10.0, .timeout = headless ? 10.0 : 11.0, .settle_tol = -1.0,  .gripper_cmd = 255.0 },
        { .name = "RETREAT",   .target = &q_retreat,  .duration = 2.0,                  .timeout = 4.0,                   .settle_tol =  0.08, .gripper_cmd = 255.0 },
        { .name = "HOLD",      .target = &q_retreat,  .duration = headless ? 1.0 : 1e9,  .timeout = headless ? 1.0 : 1e9,  .settle_tol = -1.0,  .gripper_cmd = 255.0 },
    };

    mj_kdl::VideoRecorder recorder;
    bool                  recorder_ok = false;
    const int             kRecordFps  = 60;
    const int             steps_per_frame = std::max(1, static_cast<int>(1.0 / (kRecordFps * model->opt.timestep)));
    int                   sim_step = 0;
    if (do_record) {
        if (!mj_kdl::init_video_recorder(
              &recorder, model, record_path.c_str(), mj_kdl::VideoResolution::R1080p, kRecordFps
            )) {
            std::cerr << "init_video_recorder() failed -- is EGL available and ffmpeg installed?\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        recorder.cam.azimuth   = 145.0;
        recorder.cam.elevation = -22.0;
        recorder.cam.distance  = 1.35;
        recorder.cam.lookat[0] = 0.05;
        recorder.cam.lookat[1] = 0.02;
        recorder.cam.lookat[2] = 0.88;
        recorder_ok = true;
    }

    mj_kdl::Viewer viewer;
    if (!headless && !mj_kdl::init_window_sim(&viewer, &robot)) {
        std::cerr << "init_window_sim() failed\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    KDL::JntArray q_enter(n), q_des(n);
    for (const Phase &phase : phases) {
        std::cout << "State: " << phase.name << "\n";
        const double t_enter = data->time;
        snapshot_q(robot, n, q_enter);
        while (true) {
            mj_kdl::update(&robot);
            const double alpha = phase.duration > 0.0 ? clamp01((data->time - t_enter) / phase.duration)
                                                       : 1.0;
            lerp_q(q_enter, *phase.target, alpha, q_des);
            impedance_ctrl(robot, q_des, n, dyn);
            data->ctrl[fingers_act] = phase.gripper_cmd;
            mj_kdl::update(&robot);

            const double t_rel        = data->time - t_enter;
            const bool   done_time    = t_rel >= phase.duration;
            const bool   done_pose    = phase.settle_tol < 0.0 ||
                                      max_abs_joint_err(robot, *phase.target, n) <= phase.settle_tol;
            const bool done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
            if ((done_time && done_pose) || done_timeout) break;

            if (!mj_kdl::step(&robot)) {
                mj_kdl::cleanup(&viewer);
                mj_kdl::cleanup(&robot);
                mj_kdl::destroy_scene(model, data);
                return 0;
            }
            ++sim_step;
            if (recorder_ok && sim_step % steps_per_frame == 0) {
                if (!mj_kdl::record_frame(&recorder, model, data)) {
                    std::cerr << "record_frame() failed at step " << sim_step << "\n";
                    mj_kdl::cleanup(&recorder);
                    recorder_ok = false;
                }
            }
        }
    }

    int in_jug = 0;
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

    if (recorder_ok) {
        mj_kdl::cleanup(&recorder);
        std::cout << "Saved recording: " << record_path << "\n";
    }

    std::cout << "balls in transparent receiver: " << in_jug << "/" << grain_joints.size() << "\n";
    std::cout << "grain centroid: [" << std::fixed << std::setprecision(3) << avg[0] << ", "
              << avg[1] << ", " << avg[2] << "] receiver center=[" << kJugX << ", " << kJugY
              << "]\n";
    if (headless && in_jug < 4) {
        std::cerr << "pour failed: too few balls reached the receiver\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    if (!headless) mj_kdl::cleanup(&viewer);
    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
