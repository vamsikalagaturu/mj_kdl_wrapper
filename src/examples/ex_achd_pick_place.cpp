/* ex_achd_pick_place.cpp
 * Kinova GEN3 + Robotiq 2F-85 picks a cube from one table location and places
 * it at another using Cartesian acceleration-constrained hybrid dynamics
 * via KDL ChainHdSolver_Vereshchagin_Fixed_Joint + RNEA:
 *
 *   Xddot_des = Kp * diff(T_tcp, T_target) + Kd * d/dt(diff)
 *   beta      = alpha^T * Xddot_des, with alpha = I_6
 *   qddot     = ACHD(q, qdot, alpha, beta)   [constrained joint accelerations]
 *   tau       = RNEA(q, qdot, qddot)         [full inverse dynamics for MuJoCo]
 *
 * mj_kdl_wrapper's TORQUE mode fully nulls the MuJoCo position actuators
 * (ctrl = qpos + kv/kp * qvel), so qfrc_applied is the sole torque source.
 *
 * Usage:
 *   ex_achd_pick_place [--headless] [--record output.mp4]
 *
 * With --headless runs the full sequence and prints final cube position. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include "chainhdsolver_vereshchagin_fixed_joint.hpp"
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kCubeHS      = 0.02;
static constexpr double kSceneBaseZ = 0.70;
static constexpr double kPickX      = 0.40;
static constexpr double kPickY      = 0.00;
static constexpr double kPlaceX     = 0.40;
static constexpr double kPlaceY     = 0.24;
static constexpr double kTableZ     = 0.0;

// Cartesian acceleration gains. ACHD consumes acceleration-energy setpoints; with
// alpha = I_6 these are the desired TCP linear/angular accelerations.
static constexpr double kKpLin   = 200.0;
static constexpr double kKiLin   = 100.0;
static constexpr double kKdLin   = 40.0;
static constexpr double kKpRot   = 120.0;
static constexpr double kKiRot   = 50.0;
static constexpr double kKdRot   = 80.0;
static constexpr double kBetaMaxLin = 120.0;
static constexpr double kBetaMaxRot = 80.0;
static constexpr double kTauMax  = 59.0;
static constexpr double kIntegralMax = 0.5;
static constexpr int    kRecordFps = 30;
static constexpr const char *kSupportLink = "half_arm_2_link";
static constexpr double kSupportKp   = 800.0;
static constexpr double kSupportKd   = 80.0;
static constexpr double kSupportFMax = 45.0;
static constexpr double kSupportLift = 0.06;

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double   clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }
static double   clamp_abs(double v, double limit) { return std::max(-limit, std::min(limit, v)); }
static double   smoothstep(double t)
{
    t = clamp01(t);
    return t * t * (3.0 - 2.0 * t);
}

static KDL::Frame lerp_frame(const KDL::Frame &a, const KDL::Frame &b, double t)
{
    return KDL::addDelta(a, KDL::diff(a, b), clamp01(t));
}

static void fill_q_state(
  const mj_kdl::Robot &robot,
  unsigned             n,
  KDL::JntArray       &q,
  KDL::JntArray       &qdot
)
{
    for (unsigned i = 0; i < n; ++i) {
        q(i)    = robot.jnt_pos_msr[i];
        qdot(i) = robot.jnt_vel_msr[i];
    }
}

static int find_segment_index(const KDL::Chain &chain, const std::string &name)
{
    for (unsigned i = 0; i < chain.getNrOfSegments(); ++i) {
        if (chain.getSegment(i).getName() == name) return static_cast<int>(i);
    }
    return -1;
}

static bool support_phase(const std::string &name)
{
    return name == "PLACE_ABOVE" || name == "PLACE" || name == "OPEN" || name == "RETREAT"
           || name == "HOLD";
}

static void clear_wrenches(KDL::Wrenches &wrenches)
{
    std::fill(wrenches.begin(), wrenches.end(), KDL::Wrench::Zero());
}

static double link_world_z(
  KDL::ChainFkSolverPos_recursive &fk_pos,
  const KDL::JntArray             &q,
  int                              segment_index
)
{
    KDL::Frame base_T_link;
    fk_pos.JntToCart(q, base_T_link, segment_index + 1);
    return kSceneBaseZ + base_T_link.p.z();
}

static double apply_support_wrench(
  KDL::ChainFkSolverPos_recursive &fk_pos,
  const KDL::JntArray             &q,
  int                              segment_index,
  double                           z_ref,
  double                           dt,
  double                          &prev_z,
  bool                            &prev_z_valid,
  KDL::Wrenches                   &f_ext_achd
)
{
    KDL::Frame base_T_link;
    fk_pos.JntToCart(q, base_T_link, segment_index + 1);

    const double z_world = kSceneBaseZ + base_T_link.p.z();
    const double vz = prev_z_valid ? (z_world - prev_z) / dt : 0.0;
    prev_z = z_world;
    prev_z_valid = true;

    const double fz = std::min(
      kSupportFMax,
      std::max(0.0, kSupportKp * (z_ref - z_world) - kSupportKd * vz)
    );
    const KDL::Wrench wrench_base(KDL::Vector(0.0, 0.0, fz), KDL::Vector::Zero());
    f_ext_achd[segment_index] = wrench_base;
    return fz;
}

static void achd_cartesian_ctrl(
  mj_kdl::Robot                               &robot,
  const KDL::Frame                            &target,
  unsigned                                     n,
  KDL::ChainFkSolverPos_recursive             &fk_pos,
  KDL::ChainFkSolverVel_recursive             &fk_vel,
  KDL::ChainHdSolver_Vereshchagin_Fixed_Joint &achd,
  KDL::ChainIdSolver_RNE                      &rnea,
  KDL::JntArray                               &q,
  KDL::JntArray                               &qdot,
  KDL::JntArray                               &qddot,
  KDL::Jacobian                               &alpha,
  KDL::JntArray                               &beta,
  KDL::Wrenches                               &f_ext_achd,
  KDL::Wrenches                               &f_ext_rnea_zero,
  KDL::JntArray                               &ff_torques,
  KDL::JntArray                               &constraint_torques,
  KDL::JntArray                               &tau_cmd,
  const KDL::Twist                            &target_twist,
  std::array<double, 6>                       &err_i,
  std::array<double, 6>                       &err_prev,
  bool                                        &first_pid
)
{
    fill_q_state(robot, n, q, qdot);

    KDL::Frame current;
    fk_pos.JntToCart(q, current);
    KDL::FrameVel current_vel;
    fk_vel.JntToCart(KDL::JntArrayVel(q, qdot), current_vel);
    const KDL::Twist tcp_twist = current_vel.deriv();

    const KDL::Twist err = KDL::diff(current, target);
    const double dt = robot.model ? robot.model->opt.timestep : 0.002;
    const double e[6] = { err.vel.x(), err.vel.y(), err.vel.z(),
                          err.rot.x(), err.rot.y(), err.rot.z() };
    if (first_pid) {
        for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];
        first_pid = false;
    }
    for (unsigned i = 0; i < 6; ++i) err_i[i] = clamp_abs(err_i[i] + e[i] * dt, kIntegralMax);
    const double de[6] = {
        (e[0] - err_prev[0]) / dt,
        (e[1] - err_prev[1]) / dt,
        (e[2] - err_prev[2]) / dt,
        (e[3] - err_prev[3]) / dt,
        (e[4] - err_prev[4]) / dt,
        (e[5] - err_prev[5]) / dt,
    };

    beta(0) = clamp_abs(kKpLin * e[0] + kKiLin * err_i[0] + kKdLin * de[0], kBetaMaxLin);
    beta(1) = clamp_abs(kKpLin * e[1] + kKiLin * err_i[1] + kKdLin * de[1], kBetaMaxLin);
    beta(2) = clamp_abs(kKpLin * e[2] + kKiLin * err_i[2] + kKdLin * de[2], kBetaMaxLin);
    beta(3) = clamp_abs(kKpRot * e[3] + kKiRot * err_i[3] + kKdRot * (target_twist.rot.x() - tcp_twist.rot.x()), kBetaMaxRot);
    beta(4) = clamp_abs(kKpRot * e[4] + kKiRot * err_i[4] + kKdRot * (target_twist.rot.y() - tcp_twist.rot.y()), kBetaMaxRot);
    beta(5) = clamp_abs(kKpRot * e[5] + kKiRot * err_i[5] + kKdRot * (target_twist.rot.z() - tcp_twist.rot.z()), kBetaMaxRot);
    for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];
    KDL::SetToZero(ff_torques);
    if (achd.CartToJnt(q, qdot, qddot, alpha, beta, f_ext_achd, ff_torques, constraint_torques) < 0) {
        std::cerr << "ACHD CartToJnt() failed\n";
        std::fill(robot.jnt_trq_cmd.begin(), robot.jnt_trq_cmd.end(), 0.0);
        return;
    }
    if (rnea.CartToJnt(q, qdot, qddot, f_ext_rnea_zero, tau_cmd) < 0) {
        std::cerr << "RNEA CartToJnt() failed\n";
        std::fill(robot.jnt_trq_cmd.begin(), robot.jnt_trq_cmd.end(), 0.0);
        return;
    }
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = clamp_abs(tau_cmd(i), kTauMax);
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
        .condim    = 4,
        .friction  = { 0.8, 0.02, 0.001 },
    };
}

struct Phase
{
    const char *name;
    KDL::Frame  target;
    double      duration;
    double      timeout;
    double      settle_pos_tol;
    double      settle_rot_tol;
    double      gripper_cmd;
};

int main(int argc, char *argv[])
{
    bool        headless    = false;
    bool        do_record   = false;
    bool        print_ee_angular_vel = false;
    std::string record_path = "achd_pick_place.mp4";
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--record") {
            do_record = true;
            headless  = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') record_path = argv[++i];
        } else if (arg == "--print-ee-angular-vel") {
            print_ee_angular_vel = true;
        }
    }

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string arm_mjcf   = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf   = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string table_mjcf = (root / "src/examples/assets/table.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gripper.prefix    = "g_";

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = arm_mjcf.c_str();
    robot_spec.pos[2] = kSceneBaseZ;
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.robots.push_back(robot_spec);
    mj_kdl::SceneObject table{
        .name      = "table",
        .mjcf_path = table_mjcf,
        .pos       = { 0.0, 0.0, kSceneBaseZ },
        .fixed     = true,
    };
    scene.objects.push_back(table);
    scene.objects.push_back(make_cube(kSceneBaseZ));

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene)) {
        std::cerr << "build_scene() failed\n";
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
    const unsigned ns          = robot.chain.getNrOfSegments();
    const int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    const int      cube_jnt    = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    if (fingers_act < 0 || cube_jnt < 0) {
        std::cerr << "required actuator or cube joint not found\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::ChainFkSolverPos_recursive fk_pos(robot.chain);
    KDL::ChainFkSolverVel_recursive fk_vel(robot.chain);
    KDL::Twist                      root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd(robot.chain, root_acc, 6);
    KDL::ChainIdSolver_RNE rnea(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
    KDL::JntArray q_buf(n), qdot_buf(n), qddot(n), ff_torques(n), constraint_torques(n), tau_cmd(n);
    KDL::Wrenches f_ext_achd(ns, KDL::Wrench::Zero());
    KDL::Wrenches f_ext_rnea_zero(ns, KDL::Wrench::Zero());
    KDL::Jacobian alpha(6);
    KDL::JntArray beta(6);
    for (unsigned i = 0; i < 6; ++i) alpha(i, i) = 1.0;

    const int support_segment = find_segment_index(robot.chain, kSupportLink);
    if (support_segment < 0) {
        std::cerr << "support segment not found: " << kSupportLink << "\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    std::cout << "support segment: " << kSupportLink << " index=" << support_segment << "\n";

    const double z_grasp = kCubeHS;
    const double z_above = z_grasp + 0.20;
    const double z_lift  = z_grasp + 0.30;

    mj_kdl::set_joint_pos(&robot, q_home, false);
    mj_kdl::update(&robot);
    fill_q_state(robot, n, q_buf, qdot_buf);
    KDL::Frame home_tcp;
    fk_pos.JntToCart(q_buf, home_tcp);
    const KDL::Rotation grasp_rot = robot.tip_T_tcp.M;

    auto target_frame = [&](double base_x, double base_y, double base_z) {
        return KDL::Frame(grasp_rot, KDL::Vector(base_x, base_y, base_z));
    };

    const std::vector<Phase> phases = {
        { "HOME",        home_tcp,                                      1.0,                 2.5, 0.03, 0.05, 0.0   },
        { "PICK_ABOVE",  target_frame(kPickX,  kPickY,  kTableZ + z_above), 8.0,            14.0, 0.04, 0.03, 0.0   },
        { "PICK",        target_frame(kPickX,  kPickY,  kTableZ + z_grasp), 5.0,            12.0, 0.02, 0.03, 0.0   },
        { "CLOSE",       target_frame(kPickX,  kPickY,  kTableZ + z_grasp), 1.5,             2.5, -1.0, -1.0, 255.0 },
        { "LIFT",        target_frame(kPickX,  kPickY,  kTableZ + z_lift),  3.0,             8.0, 0.04, 0.03, 255.0 },
        { "PLACE_ABOVE", target_frame(kPlaceX, kPlaceY, kTableZ + z_above), 5.0,            12.0, 0.04, 0.03, 255.0 },
        { "PLACE",       target_frame(kPlaceX, kPlaceY, kTableZ + z_grasp), 5.0,            14.0, 0.02, 0.03, 255.0 },
        { "OPEN",        target_frame(kPlaceX, kPlaceY, kTableZ + z_grasp), 1.0,             2.0, -1.0, -1.0, 0.0   },
        { "RETREAT",     target_frame(kPlaceX, kPlaceY, kTableZ + z_above), 3.0,             6.0, 0.04, 0.08, 0.0   },
        { "HOLD",        target_frame(kPlaceX, kPlaceY, kTableZ + z_above), headless ? 4.0 : 1e9, headless ? 4.0 : 1e9, -1.0, -1.0, 0.0 },
    };

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    int qadr = model->jnt_qposadr[cube_jnt];

    auto reset_cube = [&]() {
        data->qpos[qadr]     = kPickX;
        data->qpos[qadr + 1] = kPickY;
        data->qpos[qadr + 2] = kSceneBaseZ + kCubeHS;
        data->qpos[qadr + 3] = 1.0;
        data->qpos[qadr + 4] = data->qpos[qadr + 5] = data->qpos[qadr + 6] = 0.0;
    };

    mj_kdl::Env env;
    env.spec  = scene;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home, false);
        reset_cube();
        data->ctrl[fingers_act] = 0.0;
    };

    double prev_sim_time = data->time;
    bool   aborted       = false;
    bool   restart       = false;

    auto reset_scene = [&]() {
        mj_kdl::reset(&env);
        prev_sim_time = data->time;
        restart       = true;
    };

    reset_scene();

    mj_kdl::Viewer viewer;
    if (!headless && !mj_kdl::init_window_sim(&viewer, &robot)) {
        std::cerr << "init_window_sim() failed\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    mj_kdl::VideoRecorder recorder;
    bool recorder_ok = false;
    if (do_record) {
        recorder_ok = mj_kdl::init_video_recorder(
          &recorder, model, record_path.c_str(), mj_kdl::VideoResolution::R720p, kRecordFps
        );
        if (!recorder_ok)
            std::cerr << "init_video_recorder() failed -- is EGL available and ffmpeg installed?\n";
        else {
            recorder.cam.azimuth   = 145.0;
            recorder.cam.elevation = -22.0;
            recorder.cam.distance  = 1.35;
            recorder.cam.lookat[0] = 0.12;
            recorder.cam.lookat[1] = 0.12;
            recorder.cam.lookat[2] = 0.90;
        }
    }
    const int steps_per_record_frame =
      std::max(1, static_cast<int>(1.0 / (kRecordFps * model->opt.timestep)));
    int sim_step = 0;

    do {
        restart = false;
        double support_z_ref = 0.0;
        bool   support_z_ref_valid = false;
        double support_prev_z = 0.0;
        bool   support_prev_z_valid = false;
        for (const Phase &phase : phases) {
            if (aborted || restart) break;
            std::cout << "State: " << phase.name << "\n";
            double t_enter = data->time;
            mj_kdl::update(&robot);
            fill_q_state(robot, n, q_buf, qdot_buf);
            KDL::Frame phase_start;
            fk_pos.JntToCart(q_buf, phase_start);
            if (std::string(phase.name) == "PLACE_ABOVE" || (support_phase(phase.name) && !support_z_ref_valid)) {
                support_z_ref = link_world_z(fk_pos, q_buf, support_segment) + kSupportLift;
                support_z_ref_valid = true;
                support_prev_z = support_z_ref;
                support_prev_z_valid = true;
                std::cout << "  support_z_ref=" << std::fixed << std::setprecision(3)
                          << support_z_ref << "\n";
            }

            std::array<double, 6> err_i{}, err_prev{};
            bool first_pid = true;
            KDL::Frame prev_target = phase_start;
            bool first_target = true;
            while (true) {
                if (data->time < prev_sim_time - 1e-6) {
                    reset_scene();
                    break;
                }
                prev_sim_time = data->time;

                double phase_alpha =
                  phase.duration > 0.0 ? smoothstep((data->time - t_enter) / phase.duration) : 1.0;
                KDL::Frame target = lerp_frame(phase_start, phase.target, phase_alpha);
                const double dt = model->opt.timestep;
                KDL::Twist target_twist = KDL::Twist::Zero();
                if (!first_target) target_twist = KDL::diff(prev_target, target, dt);
                prev_target = target;
                first_target = false;
                mj_kdl::update(&robot);
                fill_q_state(robot, n, q_buf, qdot_buf);
                clear_wrenches(f_ext_achd);
                clear_wrenches(f_ext_rnea_zero);
                double support_force = 0.0;
                if (support_phase(phase.name) && support_z_ref_valid) {
                    support_force = apply_support_wrench(
                      fk_pos,
                      q_buf,
                      support_segment,
                      support_z_ref,
                      dt,
                      support_prev_z,
                      support_prev_z_valid,
                      f_ext_achd
                    );
                }
                achd_cartesian_ctrl(
                  robot,
                  target,
                  n,
                  fk_pos,
                  fk_vel,
                  achd,
                  rnea,
                  q_buf,
                  qdot_buf,
                  qddot,
                  alpha,
                  beta,
                  f_ext_achd,
                  f_ext_rnea_zero,
                  ff_torques,
                  constraint_torques,
                  tau_cmd,
                  target_twist,
                  err_i,
                  err_prev,
                  first_pid
                );
                mj_kdl::update(&robot);
                data->ctrl[fingers_act] = phase.gripper_cmd;

                fill_q_state(robot, n, q_buf, qdot_buf);
                KDL::Frame current;
                fk_pos.JntToCart(q_buf, current);
                if (print_ee_angular_vel) {
                    KDL::FrameVel current_vel;
                    fk_vel.JntToCart(KDL::JntArrayVel(q_buf, qdot_buf), current_vel);
                    const KDL::Twist ee_twist = current_vel.deriv();
                    std::cout << "ee_ang_vel"
                              << " phase=" << phase.name
                              << " t=" << std::fixed << std::setprecision(4) << (data->time - t_enter)
                              << " wx=" << ee_twist.rot.x()
                              << " wy=" << ee_twist.rot.y()
                              << " wz=" << ee_twist.rot.z()
                              << " target_wx=" << target_twist.rot.x()
                              << " target_wy=" << target_twist.rot.y()
                              << " target_wz=" << target_twist.rot.z()
                              << " support_fz=" << support_force
                              << "\n";
                }
                KDL::Twist err = KDL::diff(current, phase.target);

                double t_rel     = data->time - t_enter;
                bool   done_time = t_rel >= phase.duration;
                bool done_pose = phase.settle_pos_tol < 0.0
                                 || (err.vel.Norm() <= phase.settle_pos_tol
                                     && err.rot.Norm() <= phase.settle_rot_tol);
                bool   done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
                if ((done_time && done_pose) || done_timeout) break;

                if (!mj_kdl::step(&robot)) {
                    aborted = true;
                    break;
                }
                if (recorder_ok && sim_step % steps_per_record_frame == 0) {
                    if (!mj_kdl::record_frame(&recorder, model, data)) {
                        std::cerr << "record_frame() failed at sim step " << sim_step << "\n";
                        mj_kdl::cleanup(&recorder);
                        recorder_ok = false;
                    }
                }
                ++sim_step;
            }
            fill_q_state(robot, n, q_buf, qdot_buf);
            KDL::Frame phase_end;
            fk_pos.JntToCart(q_buf, phase_end);
            KDL::Twist phase_err = KDL::diff(phase_end, phase.target);
            std::cout << "  pos_err=" << std::fixed << std::setprecision(3)
                      << phase_err.vel.Norm() << " rot_err=" << phase_err.rot.Norm()
                      << " t=" << data->time - t_enter << "\n";
        }
    } while (restart);

    int ret = 0;
    if (!aborted) {
        double cube_x       = data->qpos[qadr];
        double cube_y       = data->qpos[qadr + 1];
        double cube_z       = data->qpos[qadr + 2];
        double place_err_xy = std::hypot(cube_x - kPlaceX, cube_y - kPlaceY);
        std::cout << "cube final position: [" << std::fixed << std::setprecision(3) << cube_x
                  << ", " << cube_y << ", " << cube_z << "]"
                  << " target=[" << kPlaceX << ", " << kPlaceY << ", " << kSceneBaseZ + kCubeHS
                  << "] xy_error=" << place_err_xy << "\n";
        if (headless && place_err_xy > 0.08) ret = 1;
    }
    if (!headless) mj_kdl::cleanup(&viewer);
    if (recorder_ok) {
        mj_kdl::cleanup(&recorder);
        std::cout << "Saved recording: " << record_path << "\n";
    }
    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return ret;
}
