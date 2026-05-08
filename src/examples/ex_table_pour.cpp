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
#include <cstring>
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
static constexpr double kPrePourX    = kJugX - 0.05;
static constexpr double kPrePourY    = kJugY - 0.02;
static constexpr double kPourX       = kJugX - 0.03;
static constexpr double kPourY       = kJugY;
static constexpr double kTiltX       = kJugX - 0.02;
static constexpr double kTiltY       = kJugY;
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
    mj_kdl::SceneObject o;
    char                name[32];
    std::snprintf(name, sizeof(name), "grain_%02d", idx);
    o.name    = name;
    o.shape   = mj_kdl::Shape::SPHERE;
    o.size[0] = kBallRadius;
    o.pos[0]  = 0.0;
    o.pos[1]  = 0.0;
    o.pos[2]  = kTableZ + 0.40 + idx * 2.0 * kBallRadius;
    o.rgba[0] = 1.0f;
    o.rgba[1] = 0.84f;
    o.rgba[2] = 0.30f;
    o.rgba[3] = 1.0f;
    o.mass    = 0.006;
    o.condim  = 4;
    o.friction[0] = 0.5;
    o.friction[1] = 0.02;
    o.friction[2] = 0.001;
    return o;
}

static bool add_fixed_attachment_to_world(
  mjSpec *scene,
  const char *mjcf_path,
  const char *prefix,
  double x,
  double y,
  double z,
  double roll_deg = 0.0,
  double pitch_deg = 0.0,
  double yaw_deg = 0.0
)
{
    char err[2048] = {};
    mjSpec *att = mj_parseXML(mjcf_path, nullptr, err, sizeof(err));
    if (!att) {
        std::cerr << "mj_parseXML failed for '" << mjcf_path << "': " << err << "\n";
        return false;
    }

    mjsBody *world = mjs_findBody(scene, "world");
    mjsBody *att_world = mjs_findBody(att, "world");
    mjsElement *first = att_world ? mjs_firstChild(att_world, mjOBJ_BODY, 0) : nullptr;
    mjsBody *att_root = first ? mjs_asBody(first) : nullptr;
    if (!world || !att_root) {
        mj_deleteSpec(att);
        std::cerr << "failed to find world or root body for fixed attachment\n";
        return false;
    }

    mjsFrame *place = mjs_addFrame(world, nullptr);
    place->pos[0] = x;
    place->pos[1] = y;
    place->pos[2] = z;
    if (roll_deg || pitch_deg || yaw_deg) {
        const double d2r = M_PI / 180.0;
        KDL::Rotation rot = KDL::Rotation::RPY(roll_deg * d2r, pitch_deg * d2r, yaw_deg * d2r);
        double qx, qy, qz, qw;
        rot.GetQuaternion(qx, qy, qz, qw);
        place->quat[0] = qw;
        place->quat[1] = qx;
        place->quat[2] = qy;
        place->quat[3] = qz;
    }

    if (!mjs_attach(place->element, att_root->element, prefix ? prefix : "", "")) {
        std::cerr << "mjs_attach failed for fixed attachment: " << mjs_getError(scene) << "\n";
        mj_deleteSpec(att);
        return false;
    }
    mj_deleteSpec(att);
    return true;
}

static void set_free_body_pose(mjModel *model, mjData *data, const char *joint_name, const double pos[3])
{
    int jid = mj_name2id(model, mjOBJ_JOINT, joint_name);
    if (jid < 0) return;
    int qadr          = model->jnt_qposadr[jid];
    int dadr          = model->jnt_dofadr[jid];
    data->qpos[qadr]     = pos[0];
    data->qpos[qadr + 1] = pos[1];
    data->qpos[qadr + 2] = pos[2];
    data->qpos[qadr + 3] = 1.0;
    data->qpos[qadr + 4] = 0.0;
    data->qpos[qadr + 5] = 0.0;
    data->qpos[qadr + 6] = 0.0;
    for (int k = 0; k < 6; ++k) data->qvel[dadr + k] = 0.0;
}

static void body_local_to_world(const mjData *data, int body_id, const double local[3], double world[3])
{
    const double *p = data->xpos + 3 * body_id;
    const double *r = data->xmat + 9 * body_id;
    for (int i = 0; i < 3; ++i) {
        world[i] = p[i] + r[3 * i + 0] * local[0] + r[3 * i + 1] * local[1] + r[3 * i + 2] * local[2];
    }
}

static KDL::Vector site_point_to_local(
  const mjData *data,
  int           site_id,
  const double  point_world[3]
)
{
    const double *p = data->site_xpos + 3 * site_id;
    const double *r = data->site_xmat + 9 * site_id;
    const double  d[3] = {
        point_world[0] - p[0],
        point_world[1] - p[1],
        point_world[2] - p[2],
    };
    return KDL::Vector(
      r[0] * d[0] + r[3] * d[1] + r[6] * d[2],
      r[1] * d[0] + r[4] * d[1] + r[7] * d[2],
      r[2] * d[0] + r[5] * d[1] + r[8] * d[2]
    );
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
    scene_cfg.table.enabled     = true;
    scene_cfg.table.pos[2]      = kTableZ;
    scene_cfg.table.top_size[0] = 0.8;
    scene_cfg.table.top_size[1] = 0.6;
    scene_cfg.table.thickness   = 0.04;
    scene_cfg.table.leg_radius  = 0.03;
    for (int i = 0; i < num_balls; ++i) scene_cfg.objects.push_back(make_ball(i));

    mj_kdl::ensure_plugins_loaded();
    mjSpec *scene = mj_makeSpec();
    if (!scene) {
        std::cerr << "mj_makeSpec() failed\n";
        return 1;
    }

    char err[2048] = {};
    mjSpec *arm = mj_parseXML(arm_mjcf.c_str(), nullptr, err, sizeof(err));
    if (!arm) {
        std::cerr << "mj_parseXML failed for arm: " << err << "\n";
        mj_deleteSpec(scene);
        return 1;
    }
    scene->option = arm->option;
    if (!mj_kdl::attach_to_spec(arm, &gripper) || !mj_kdl::attach_to_spec(arm, &bottle)) {
        mj_deleteSpec(arm);
        mj_deleteSpec(scene);
        return 1;
    }
    mjsBody *world = mjs_findBody(scene, "world");
    mjsBody *arm_world = mjs_findBody(arm, "world");
    mjsElement *first = arm_world ? mjs_firstChild(arm_world, mjOBJ_BODY, 0) : nullptr;
    mjsBody *arm_root = first ? mjs_asBody(first) : nullptr;
    if (!world || !arm_root) {
        std::cerr << "no root body found in arm spec\n";
        mj_deleteSpec(arm);
        mj_deleteSpec(scene);
        return 1;
    }
    mjsFrame *place = mjs_addFrame(world, nullptr);
    place->pos[0] = kRobotBackX;
    place->pos[2] = kTableZ;
    if (!mjs_attach(place->element, arm_root->element, "", "")) {
        std::cerr << "mjs_attach failed for arm: " << mjs_getError(scene) << "\n";
        mj_deleteSpec(arm);
        mj_deleteSpec(scene);
        return 1;
    }
    mj_deleteSpec(arm);

    mj_kdl::configure_spec(scene, &scene_cfg);
    if (!add_fixed_attachment_to_world(scene, receiver_mjcf.c_str(), "recv_", kJugX, kJugY, kReceiverFrameZ)) {
        mj_deleteSpec(scene);
        return 1;
    }

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::compile_and_make_data(scene, &model, &data)) {
        std::cerr << "compile_and_make_data() failed\n";
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
    const int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    const int      bottle_body = mj_name2id(model, mjOBJ_BODY, "pour_mug");
    const int      tcp_site    = mj_name2id(model, mjOBJ_SITE, "g_pinch");
    if (fingers_act < 0 || bottle_body < 0 || tcp_site < 0) {
        std::cerr << "required gripper actuator, pouring bottle body, or tcp site not found\n";
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
    mj_forward(model, data);
    double outlet_home[3] = {};
    body_local_to_world(data, bottle_body, (const double[3]){ 0.0, 0.0, 0.070 }, outlet_home);
    const KDL::Vector tcp_outlet = site_point_to_local(data, tcp_site, outlet_home);

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
        mj_forward(model, data);

        double outlet_world[3] = {};
        body_local_to_world(data, bottle_body, (const double[3]){ 0.0, 0.0, 0.070 }, outlet_world);
        const double dx = kJugX - outlet_world[0];
        const double dy = kJugY - outlet_world[1];
        const double dz = kTiltOutletZ - outlet_world[2];
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

    /* Scene-specific reset: place balls inside bottle, close gripper.
     * Called both at startup (after mj_resetData) and by the Simulate UI
     * reset callback (on_reset), where mj_resetData has already been done. */
    auto reset_scene_objects = [&](mjModel *m, mjData *d) {
        mj_kdl::set_joint_pos(&robot, q_home, false);
        mj_forward(m, d);

        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_pos_cmd[i]                      = d->qpos[robot.kdl_to_mj_qpos[i]];
            robot.jnt_trq_cmd[i]                      = 0.0;
            d->qfrc_applied[robot.kdl_to_mj_dof[i]]  = 0.0;
        }

        const double spacing = 2.00 * kBallRadius;
        for (int i = 0; i < num_balls; ++i) {
            const int layer = i / 16;
            const int slot  = i % 16;
            const double ix = static_cast<double>(slot % 4) - 1.5;
            const double iy = static_cast<double>(slot / 4) - 1.5;
            const double local[3] = {
                ix * spacing,
                iy * spacing,
                -0.026 + layer * spacing,
            };
            double world[3];
            body_local_to_world(d, bottle_body, local, world);
            char joint_name[32];
            std::snprintf(joint_name, sizeof(joint_name), "grain_%02d_joint", i);
            set_free_body_pose(m, d, joint_name, world);
        }
        mj_forward(m, d);
        mj_kdl::update(&robot);
        d->ctrl[fingers_act] = 255.0;
    };

    /* Full reset: data wipe + scene objects. Used at startup. */
    auto reset_scene = [&]() {
        mj_resetData(model, data);
        reset_scene_objects(model, data);
    };

    /* Hook into the Simulate UI Reset button so the scene is fully restored. */
    robot.on_reset = reset_scene_objects;

    reset_scene();

    const std::vector<Phase> phases = {
        { "HOME", &q_home, 1.0, 2.5, 0.08, 255.0 },
        { "PRE_POUR", &q_pre_pour, 4.0, 6.5, 0.08, 255.0 },
        { "POUR", &q_pour, 3.5, 5.5, 0.07, 255.0 },
        { "TILT", &q_tilt, 7.0, 10.0, 0.07, 255.0 },
        { "POUR_HOLD", &q_tilt, headless ? 9.0 : 10.0, headless ? 10.0 : 11.0, -1.0, 255.0 },
        { "RETREAT", &q_retreat, 2.0, 4.0, 0.08, 255.0 },
        { "HOLD", &q_retreat, headless ? 1.0 : 1e9, headless ? 1.0 : 1e9, -1.0, 255.0 },
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
