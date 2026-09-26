/* ex_achd_pick_place.cpp
 * The table pick-place with Cartesian acceleration-constrained hybrid dynamics:
 * KDL ChainHdSolver_Vereshchagin + RNEA in TORQUE mode:
 *
 *   Xddot_des = Kp * diff(T_tcp, T_target) + Ki * integral + Kd * d/dt(diff)
 *   beta      = alpha^T * Xddot_des, with alpha = I_6
 *   qddot     = ACHD(q, qdot, alpha, beta, f_ext)
 *   tau       = RNEA(q, qdot, qddot)
 *
 * In every phase an ACHD-only upward wrench on half_arm_2_link keeps the elbow from dropping
 * while the TCP task holds.
 *
 * Usage:
 *   ex_achd_pick_place [--headless] [--record output.mp4]
 *
 * Runs the sequence once and prints the cube's final position; --headless skips the viewer and
 * exits 1 unless the cube rests on the table within kMaxPlaceErr of the place spot and the
 * elbow stayed within kMaxElbowDrop of its reference; --record
 * writes an MP4 offscreen (EGL + ffmpeg) and implies --headless. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl/chainhdsolver_vereshchagin.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace ex = mj_kdl_examples;

static constexpr double kMaxPlaceErr  = 0.005; // [m] in the table plane
static constexpr double kMaxElbowDrop = 0.10;  // [m] below the support reference

// With alpha = I_6 these are the desired TCP linear/angular accelerations.
static constexpr double kKpLin       = 200.0;
static constexpr double kKiLin       = 100.0;
static constexpr double kKdLin       = 40.0;
static constexpr double kKpRot       = 120.0;
static constexpr double kKiRot       = 50.0;
static constexpr double kKdRot       = 80.0;
static constexpr double kBetaMaxLin  = 120.0;
static constexpr double kBetaMaxRot  = 80.0;
static constexpr double kIntegralMax = 0.5;
static constexpr int    kRecordFps   = 30;

static constexpr const char *kSupportLink = "half_arm_2_link";
static constexpr double      kSupportKp   = 800.0;
static constexpr double      kSupportKd   = 80.0;
static constexpr double      kSupportFMax = 45.0;
static constexpr double      kSupportLift = 0.06;

static double smoothstep(double t)
{
    t = ex::clamp01(t);
    return t * t * (3.0 - 2.0 * t);
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

struct AchdController
{
    explicit AchdController(const mj_kdl::Robot &r, double gravity_z)
      : robot(r), n(r.chain.getNrOfJoints()), fk_pos(r.chain), fk_vel(r.chain),
        achd(r.chain, KDL::Twist(KDL::Vector(0.0, 0.0, -gravity_z), KDL::Vector::Zero()), 6),
        rnea(r.chain, KDL::Vector(0.0, 0.0, gravity_z)), q(n), qdot(n), qddot(n), ff_tau(n),
        constraint_tau(n), tau(n), beta(6), alpha(6),
        f_ext(r.chain.getNrOfSegments(), KDL::Wrench::Zero()),
        f_zero(r.chain.getNrOfSegments(), KDL::Wrench::Zero())
    {
        for (unsigned i = 0; i < 6; ++i) alpha(i, i) = 1.0;
    }

    void new_phase()
    {
        err_i.fill(0.0);
        first = true;
    }

    // The TCP frame of the measured state; call once per cycle before control().
    const KDL::Frame &measure()
    {
        ex::read_q(robot, q, qdot);
        fk_pos.JntToCart(q, tcp);
        return tcp;
    }

    // ACHD qddot for the TCP target, priced by RNEA into jnt_trq_cmd; false if a solver failed.
    bool control(mj_kdl::Robot &out, const KDL::Frame &target, const KDL::Twist &target_twist)
    {
        KDL::FrameVel tcp_vel;
        fk_vel.JntToCart(KDL::JntArrayVel(q, qdot), tcp_vel);
        const KDL::Twist w    = tcp_vel.deriv();
        const KDL::Twist err  = KDL::diff(tcp, target);
        const double     dt   = robot.model->opt.timestep;
        const double     e[6] = { err.vel.x(), err.vel.y(), err.vel.z(),
                                  err.rot.x(), err.rot.y(), err.rot.z() };
        if (first) std::copy(e, e + 6, err_prev.begin());
        first = false;
        // The rotation is damped against the measured rate: the target's own turn is no error.
        const double d_rot[3] = { target_twist.rot.x() - w.rot.x(),
                                  target_twist.rot.y() - w.rot.y(),
                                  target_twist.rot.z() - w.rot.z() };
        for (unsigned i = 0; i < 6; ++i) {
            err_i[i]           = ex::clamp_abs(err_i[i] + e[i] * dt, kIntegralMax);
            const bool   lin   = i < 3;
            const double d     = lin ? (e[i] - err_prev[i]) / dt : d_rot[i - 3];
            const double accel = lin ? kKpLin * e[i] + kKiLin * err_i[i] + kKdLin * d
                                     : kKpRot * e[i] + kKiRot * err_i[i] + kKdRot * d;
            beta(i)            = ex::clamp_abs(accel, lin ? kBetaMaxLin : kBetaMaxRot);
            err_prev[i]        = e[i];
        }
        KDL::SetToZero(ff_tau);
        if (achd.CartToJnt(q, qdot, qddot, alpha, beta, f_ext, ff_tau, constraint_tau) < 0)
            return false;
        if (rnea.CartToJnt(q, qdot, qddot, f_zero, tau) < 0) return false;
        // update() clamps each torque to its joint's limit and reports it in jnt_saturated.
        for (unsigned i = 0; i < n; ++i) out.jnt_trq_cmd[i] = tau(i);
        return true;
    }

    const mj_kdl::Robot            &robot;
    unsigned                        n;
    KDL::ChainFkSolverPos_recursive fk_pos;
    KDL::ChainFkSolverVel_recursive fk_vel;
    KDL::ChainHdSolver_Vereshchagin achd;
    KDL::ChainIdSolver_RNE          rnea;
    KDL::JntArray                   q, qdot, qddot, ff_tau, constraint_tau, tau, beta;
    KDL::Jacobian                   alpha;
    KDL::Wrenches                   f_ext, f_zero;
    KDL::Frame                      tcp;
    std::array<double, 6>           err_i{}, err_prev{};
    bool                            first = true;
};

int main(int argc, char *argv[])
{
    const ex::Args args     = ex::parse_args(argc, argv, "achd_pick_place.mp4");
    const bool     headless = args.headless;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = ex::menagerie_model("kinova_gen3/gen3.xml");
    robot_spec.pos[2] = ex::kTableZ;
    robot_spec.attachments.push_back(ex::gripper_attachment(ex::asset("robotiq_2f85/2f85.xml")));

    mj_kdl::SceneSpec scene = ex::scene_spec();
    scene.robots.push_back(robot_spec);
    scene.objects.push_back(ex::table_object(ex::asset("table.xml"), ex::kTableZ));
    scene.objects.push_back(ex::cube_object(ex::kPickXY[0], ex::kPickXY[1], ex::kTableZ));

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &scene)) return 1;
    const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool))
        return 1;
    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;
    mj_kdl::SceneActuatorSlot *fingers =
      mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    mj_kdl::SceneFreeBodySlot *cube = mj_kdl::bind_scene_free_body(&env.scene, "cube");
    if (!fingers || !cube) return 1;

    int support_segment = -1;
    for (unsigned i = 0; i < robot.chain.getNrOfSegments(); ++i)
        if (robot.chain.getSegment(i).getName() == kSupportLink)
            support_segment = static_cast<int>(i);
    if (support_segment < 0) {
        std::cerr << "support segment not found: " << kSupportLink << "\n";
        return 1;
    }

    const unsigned      n      = robot.chain.getNrOfJoints();
    const KDL::JntArray q_home = ex::home_q(n);
    AchdController      ctrl(robot, scene.gravity_z);
    KDL::ChainDynParam  dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));

    KDL::Frame home_tcp;
    ctrl.fk_pos.JntToCart(q_home, home_tcp);
    const auto at = [&](const double xy[2], double z) {
        return KDL::Frame(robot.tip_T_tcp.M, KDL::Vector(xy[0], xy[1], z));
    };
    const double z_grasp = ex::kCubeHS, z_above = z_grasp + 0.20, z_lift = z_grasp + 0.30;
    const double close = ex::kGripperClosed;
    const auto  &pick = ex::kPickXY, &place = ex::kPlaceXY;
    // clang-format off
    const std::vector<Phase> phases = {
        { "HOME",        home_tcp,              1.0,  2.5,  0.03,  0.05, 0.0   },
        { "PICK_ABOVE",  at(pick, z_above),     8.0, 14.0,  0.04,  0.03, 0.0   },
        { "PICK",        at(pick, z_grasp),     5.0, 12.0,  0.02,  0.03, 0.0   },
        { "CLOSE",       at(pick, z_grasp),     1.5,  2.5, -1.0,  -1.0,  close },
        { "LIFT",        at(pick, z_lift),      3.0,  8.0,  0.04,  0.03, close },
        { "PLACE_ABOVE", at(place, z_above),    5.0, 12.0,  0.04,  0.03, close },
        { "PLACE",       at(place, z_grasp),    5.0, 14.0,  0.02,  0.03, close },
        { "OPEN",        at(place, z_grasp),    1.0,  2.0, -1.0,  -1.0,  0.0   },
        { "RETREAT",     at(place, z_above),    3.0,  6.0,  0.04,  0.08, 0.0   },
        { "HOLD",        at(place, z_above),    4.0,  4.0, -1.0,  -1.0,  0.0   },
    };
    // clang-format on

    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, q_home);
        ex::place_cube(env);
        ctx->data->ctrl[fingers->ctrl_id] = 0.0;
        ex::prime_gravity(robot, dyn, q_home);
        restart = true;
    };
    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    mj_kdl::VideoRecorder recorder;
    bool                  recording = false;
    if (args.record) {
        const mj_kdl::Status s = mj_kdl::init_video_recorder(
          &recorder, env.model, args.record_path.c_str(), mj_kdl::VideoResolution::R720p, kRecordFps
        );
        if (!s) {
            std::cerr << "init_video_recorder() failed: " << s.error << "\n";
            return 1;
        }
        recorder.cam.azimuth   = 145.0;
        recorder.cam.elevation = -22.0;
        recorder.cam.distance  = 1.35;
        recorder.cam.lookat[0] = 0.12;
        recorder.cam.lookat[1] = 0.12;
        recorder.cam.lookat[2] = 0.90;
        recording              = true;
    }
    const int steps_per_frame =
      std::max(1, static_cast<int>(1.0 / (kRecordFps * env.model->opt.timestep)));

    bool   aborted = false, solver_failed = false, record_failed = false;
    int    sim_step   = 0;
    double elbow_drop = 0.0; // [m] below the support reference, the worst over the run
    do {
        restart               = false;
        elbow_drop            = 0.0;
        bool   support_valid  = false;
        double support_z_ref  = 0.0;
        double support_prev_z = 0.0;
        for (const Phase &phase : phases) {
            if (aborted || restart) break;
            std::cout << "State: " << phase.name << "\n";
            const double     t_enter     = env.data->time;
            const KDL::Frame phase_start = ctrl.measure();
            KDL::Frame       link;
            ctrl.fk_pos.JntToCart(ctrl.q, link, support_segment + 1);
            if (!support_valid || std::string(phase.name) == "PLACE_ABOVE") {
                support_z_ref  = ex::kTableZ + link.p.z() + kSupportLift;
                support_prev_z = support_z_ref;
                support_valid  = true;
            }
            ctrl.new_phase();
            KDL::Frame prev_target = phase_start;
            while (true) {
                mj_kdl::update(&env);
                const KDL::Frame &tcp   = ctrl.measure();
                const double      t_rel = env.data->time - t_enter;
                const double      dt    = env.model->opt.timestep;
                const double s = phase.duration > 0.0 ? smoothstep(t_rel / phase.duration) : 1.0;
                const KDL::Frame target =
                  KDL::addDelta(phase_start, KDL::diff(phase_start, phase.target), s);
                const KDL::Twist target_twist = KDL::diff(prev_target, target, dt);
                prev_target                   = target;

                std::fill(ctrl.f_ext.begin(), ctrl.f_ext.end(), KDL::Wrench::Zero());
                ctrl.fk_pos.JntToCart(ctrl.q, link, support_segment + 1);
                const double z  = ex::kTableZ + link.p.z();
                const double vz = (z - support_prev_z) / dt;
                support_prev_z  = z;
                elbow_drop      = std::max(elbow_drop, support_z_ref - kSupportLift - z);
                const double fz =
                  std::clamp(kSupportKp * (support_z_ref - z) - kSupportKd * vz, 0.0, kSupportFMax);
                ctrl.f_ext[support_segment] =
                  KDL::Wrench(KDL::Vector(0.0, 0.0, fz), KDL::Vector::Zero());
                if (!ctrl.control(robot, target, target_twist)) {
                    std::cerr << "ACHD/RNEA failed in " << phase.name << "\n";
                    solver_failed = aborted = true;
                    break;
                }
                fingers->command = phase.gripper_cmd;

                const KDL::Twist err = KDL::diff(tcp, phase.target);
                const bool       settled =
                  err.vel.Norm() <= phase.settle_pos_tol && err.rot.Norm() <= phase.settle_rot_tol;
                const bool done_pose    = phase.settle_pos_tol < 0.0 || settled;
                const bool done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
                if ((t_rel >= phase.duration && done_pose) || done_timeout) {
                    std::cout << "  pos_err=" << std::fixed << std::setprecision(3)
                              << err.vel.Norm() << " rot_err=" << err.rot.Norm() << " t=" << t_rel
                              << "\n";
                    break;
                }

                if (!mj_kdl::step(&env)) {
                    aborted = true;
                    break;
                }
                if (restart) break;
                mj_kdl::pace_realtime(&env);
                if (recording && ++sim_step % steps_per_frame == 0
                    && !mj_kdl::record_frame(&recorder, &env)) {
                    std::cerr << "record_frame() failed at step " << sim_step << "\n";
                    mj_kdl::cleanup(&recorder);
                    recording     = false;
                    record_failed = true;
                }
            }
        }
    } while (restart);
    mj_kdl::update(&env);
    if (recording) {
        mj_kdl::cleanup(&recorder);
        std::cout << "Saved recording: " << args.record_path << "\n";
    }

    const KDL::Vector c        = cube->pose.p;
    const double      place_xy = std::hypot(c.x() - ex::kPlaceXY[0], c.y() - ex::kPlaceXY[1]);
    const bool        on_table = std::abs(c.z() - (ex::kTableZ + ex::kCubeHS)) < 0.002;
    std::cout << std::fixed << std::setprecision(4) << "cube final position: [" << c.x() << ", "
              << c.y() << ", " << c.z() << "] place error " << place_xy * 1000.0 << " mm (limit "
              << kMaxPlaceErr * 1000.0 << " mm)" << (on_table ? "" : ", not on the table") << "\n";
    std::cout << "elbow drop: " << elbow_drop * 1000.0 << " mm (limit " << kMaxElbowDrop * 1000.0
              << " mm)\n";
    mj_kdl::cleanup(&env);
    const bool ok = !aborted && !solver_failed && !record_failed && on_table
                    && place_xy <= kMaxPlaceErr && elbow_drop <= kMaxElbowDrop;
    return headless ? ex::verdict(ok, "the cube was placed with the elbow held up") : 0;
}
