/* ex_admittance_ft.cpp
 * Kinova Gen3 + wrist F/T sensor + Robotiq 2F-85 on a table: admittance control driven by the
 * named F/T sensor, with an RNEA computed-torque inner loop in task space (TORQUE mode).
 *
 * Outer admittance law per Cartesian axis (K = 0: hand-guiding, the pose holds on release):
 *   M * a = F_ext - D * v,  v += a * dt,  offset += v * dt
 * Inner loop:
 *   beta      = Cartesian PD on the TCP pose error    (desired TCP acceleration)
 *   qddot_des = WDLS(beta) + null-space posture        (resolved acceleration, holds the elbow)
 *   tau       = RNEA(q, qdot, qddot_des)               (KDL ChainIdSolver_RNE)
 *
 * The run: a scripted helix force drives the admittance, the F/T sensor is tared, then the
 * controller follows the measured force (a scripted push headless, the mouse in the viewer).
 *
 * Usage:
 *   ex_admittance_ft [--headless]
 *
 * --headless runs the self-check and exits non-zero on failure; with the viewer the same
 * sequence runs once and exits. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace ex = mj_kdl_examples;

// Admittance: virtual mass, damping, stiffness (isotropic).
static constexpr double kMAdm          = 8.0;
static constexpr double kDAdm          = 80.0;
static constexpr double kKAdm          = 0.0;
static constexpr double kForceDeadband = 2.5;  // [N]
static constexpr double kMaxOffset     = 0.20; // [m]
static constexpr double kMaxVel        = 0.25; // [m/s]

// Helix intro.
static constexpr double kTeachTime   = 16.0; // [s]
static constexpr double kTeachRadius = 0.04; // [m]
static constexpr double kTeachRise   = 0.10; // [m]
static constexpr double kTeachTurns  = 5.0;

// Sequence and self-check.
static constexpr int    kSettleSteps      = 300;
static constexpr double kHandoffTareTime  = 1.0; // [s]
static constexpr int    kHandoffSteps     = 100;
static constexpr double kSettleTime       = 0.5;                // [s]
static constexpr double kPushTime         = 4.0;                // [s]
static constexpr double kSelfcheckPush[3] = { 8.0, 12.0, 6.0 }; // [N]
static constexpr double kMinElbowHeight   = 0.64;               // [m] above the table

// Inner loop: Cartesian PD gains and acceleration limits.
static constexpr double kKpLin      = 2500.0;
static constexpr double kKdLin      = 100.0;
static constexpr double kKpRot      = 2500.0;
static constexpr double kKdRot      = 100.0;
static constexpr double kBetaLinMax = 300.0;
static constexpr double kBetaRotMax = 300.0;
static constexpr double kKpNull     = 100.0; // posture gains in the task's null space
static constexpr double kKdNull     = 20.0;

static constexpr const char *kFtSensor        = "wrist_ft";
static constexpr const char *kToolBody        = "g_base"; // where the self-check pushes
static constexpr const char *kGripperActuator = "g_fingers_actuator";
static constexpr const char *kElbowBody       = "forearm_link"; // its origin is the elbow joint

static KDL::Vector vclamp(const KDL::Vector &v, double limit)
{
    return KDL::Vector(
      std::clamp(v.x(), -limit, limit),
      std::clamp(v.y(), -limit, limit),
      std::clamp(v.z(), -limit, limit)
    );
}

static double norm3(const KDL::Vector &v)
{
    return std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z());
}

// Scripted force whose admittance response is a helix of kTeachTurns turns.
static KDL::Vector spiral_force(double t)
{
    if (t < 0.0 || t > kTeachTime) return KDL::Vector::Zero();
    const double theta     = 2.0 * M_PI * kTeachTurns * t / kTeachTime;
    const double theta_dot = 2.0 * M_PI * kTeachTurns / kTeachTime;
    return KDL::Vector(
      kDAdm * (-kTeachRadius * theta_dot * std::sin(theta)),
      kDAdm * (kTeachRadius * theta_dot * std::cos(theta)),
      kDAdm * (kTeachRise / kTeachTime)
    );
}

struct Admittance
{
    KDL::Vector bias   = KDL::Vector::Zero(); // tared F/T force, world frame
    KDL::Vector offset = KDL::Vector::Zero();
    KDL::Vector vel    = KDL::Vector::Zero();
};

static void admittance_update(Admittance &a, const KDL::Vector &force, double dt)
{
    if (norm3(force) == 0.0) {
        a.vel = KDL::Vector::Zero();
        return;
    }
    const KDL::Vector acc = (force - kDAdm * a.vel - kKAdm * a.offset) / kMAdm;
    a.vel                 = vclamp(a.vel + acc * dt, kMaxVel);
    a.offset              = vclamp(a.offset + a.vel * dt, kMaxOffset);
}

// The sensor's force rotated into the world frame.
static KDL::Vector ft_force_world(const mj_kdl::Robot &robot)
{
    const auto ft =
      std::find_if(robot.ft_sensors.begin(), robot.ft_sensors.end(), [](const auto &s) {
          return s.name == kFtSensor;
      });
    if (ft == robot.ft_sensors.end() || ft->frame_site_id < 0) return KDL::Vector::Zero();
    const double       *m = robot.data->site_xmat + 9 * ft->frame_site_id;
    const KDL::Rotation R(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
    return R * ft->wrench.force;
}

static KDL::Vector external_force(const mj_kdl::Robot &robot, const Admittance &a)
{
    const KDL::Vector f = a.bias - ft_force_world(robot);
    return norm3(f) < kForceDeadband ? KDL::Vector::Zero() : f;
}

struct Scene
{
    mj_kdl::SceneSpec          spec;
    mj_kdl::Env                env;
    mj_kdl::Robot              robot;
    mj_kdl::SceneActuatorSlot *gripper   = nullptr;
    mj_kdl::SceneWrenchSlot   *push      = nullptr; // the self-check's hand on the tool
    bool                       restarted = false;   // set by env.on_reset (the viewer's reset)
};

static bool build_scene(Scene &s)
{
    mj_kdl::AttachmentSpec ft_spec;
    ft_spec.mjcf_path = ex::asset("ft_sensor.xml");
    ft_spec.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path      = ex::menagerie_model("kinova_gen3/gen3.xml");
    robot_spec.attach_to = { mj_kdl::AttachKind::Site, "table_top" };
    robot_spec.attachments.push_back(ft_spec);
    robot_spec.attachments.push_back(
      ex::gripper_attachment(ex::asset("robotiq_2f85/2f85.xml"), "wrist_ft_site")
    );

    s.spec = ex::scene_spec();
    s.spec.objects.push_back(ex::table_object(ex::asset("table.xml"), ex::kTableZ));
    s.spec.robots.push_back(robot_spec);

    if (!mj_kdl::init_env(&s.env, &s.spec)) return false;

    mj_kdl::ForceTorqueSensorSpec ft_sensor;
    ft_sensor.name             = kFtSensor;
    ft_sensor.frame_site       = "wrist_ft_site";
    mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
    tool.ft_sensors.push_back(ft_sensor);

    if (!mj_kdl::init_robot_from_mjcf(&s.robot, &s.env, "base_link", "bracelet_link", "", &tool))
        return false;
    s.gripper = mj_kdl::bind_scene_actuator(&s.env.scene, kGripperActuator);
    s.push    = mj_kdl::bind_scene_wrench(&s.env.scene, kToolBody);
    return s.gripper && s.push;
}

// Task-space computed torque: Cartesian PD -> WDLS acceleration IK + null-space posture -> RNEA.
struct RneaController
{
    RneaController(Scene &scene, const KDL::JntArray &q_posture)
      : s(scene), fk(s.robot.chain), jac_solver(s.robot.chain), ik_acc(s.robot.chain),
        rnea(s.robot.chain, KDL::Vector(0.0, 0.0, s.spec.gravity_z)), posture(q_posture),
        q(s.robot.n_joints), qd(s.robot.n_joints), qdd(s.robot.n_joints), z(s.robot.n_joints),
        tau(s.robot.n_joints), jac(s.robot.n_joints),
        f_ext(s.robot.chain.getNrOfSegments(), KDL::Wrench::Zero())
    {
        ik_acc.setLambda(0.10);
    }

    KDL::Frame tcp()
    {
        for (int i = 0; i < s.robot.n_joints; ++i) q(i) = s.robot.jnt_pos_msr[i];
        KDL::Frame out;
        fk.JntToCart(q, out);
        return out;
    }

    void track(const KDL::Frame &target)
    {
        for (int i = 0; i < s.robot.n_joints; ++i) qd(i) = s.robot.jnt_vel_msr[i];
        const KDL::Twist err = KDL::diff(tcp(), target);

        jac_solver.JntToJac(q, jac);
        KDL::Twist tcp_vel = KDL::Twist::Zero();
        for (unsigned j = 0; j < q.rows(); ++j) tcp_vel += jac.getColumn(j) * qd(j);

        const auto pd = [](double kp, double kd, double e, double v, double lim) {
            return std::clamp(kp * e - kd * v, -lim, lim);
        };
        const KDL::Twist beta(
          KDL::Vector(
            pd(kKpLin, kKdLin, err.vel.x(), tcp_vel.vel.x(), kBetaLinMax),
            pd(kKpLin, kKdLin, err.vel.y(), tcp_vel.vel.y(), kBetaLinMax),
            pd(kKpLin, kKdLin, err.vel.z(), tcp_vel.vel.z(), kBetaLinMax)
          ),
          KDL::Vector(
            pd(kKpRot, kKdRot, err.rot.x(), tcp_vel.rot.x(), kBetaRotMax),
            pd(kKpRot, kKdRot, err.rot.y(), tcp_vel.rot.y(), kBetaRotMax),
            pd(kKpRot, kKdRot, err.rot.z(), tcp_vel.rot.z(), kBetaRotMax)
          )
        );

        // qdd = J#(beta - J z) + z = J# beta + (I - J# J) z (Siciliano et al. 2009, Sec. 3.5.1).
        KDL::Twist jz = KDL::Twist::Zero();
        for (unsigned j = 0; j < q.rows(); ++j) {
            z(j) = kKpNull * (posture(j) - q(j)) - kKdNull * qd(j);
            jz += jac.getColumn(j) * z(j);
        }
        if (ik_acc.CartToJnt(q, beta - jz, qdd) < 0) return;
        for (unsigned j = 0; j < q.rows(); ++j) qdd(j) += z(j);
        if (rnea.CartToJnt(q, qd, qdd, f_ext, tau) < 0) return;
        // update() clamps each torque to its joint's limit and reports it in jnt_saturated.
        for (int i = 0; i < s.robot.n_joints; ++i) s.robot.jnt_trq_cmd[i] = tau(i);
    }

    Scene                          &s;
    KDL::ChainFkSolverPos_recursive fk;
    KDL::ChainJntToJacSolver        jac_solver;
    KDL::ChainIkSolverVel_wdls      ik_acc;
    KDL::ChainIdSolver_RNE          rnea;
    const KDL::JntArray             posture;
    KDL::JntArray                   q, qd, qdd, z, tau;
    KDL::Jacobian                   jac;
    KDL::Wrenches                   f_ext;
};

// Admits the force into the offset and tracks the offset target; returns that target.
static KDL::Frame control(
  Scene             &s,
  RneaController    &ctrl,
  Admittance        &a,
  const KDL::Frame  &nominal,
  const KDL::Vector &force
)
{
    admittance_update(a, force, s.spec.timestep);
    const KDL::Frame target(nominal.M, nominal.p + a.offset);
    ctrl.track(target);
    return target;
}

static void close_gripper(Scene &s) { s.gripper->command = ex::kGripperClosed; }

// Hold home with the gripper closed until the wrist load settles, then tare the sensor.
static void settle_and_tare(Scene &s, RneaController &ctrl, Admittance &a)
{
    mj_kdl::update(&s.env);
    const KDL::Frame home = ctrl.tcp();
    for (int i = 0; i < kSettleSteps; ++i) {
        mj_kdl::update(&s.env);
        close_gripper(s);
        ctrl.track(home);
        if (!mj_kdl::step(&s.env)) break;
        mj_kdl::pace_realtime(&s.env);
    }
    mj_kdl::update(&s.env);
    a.bias = ft_force_world(s.robot);
}

struct Metrics
{
    double helix_react       = 0.0;
    double helix_track_err   = 0.0;
    double helix_settle_err  = 0.0;
    double handoff_force     = 0.0;
    double push_response     = 0.0;
    double push_dy           = 0.0;
    double push_recovery_err = 0.0;
    double hold_drift        = 0.0;
    double elbow_start       = 0.0; // [m] elbow height above the table
    double elbow_min         = 0.0;
};

static double elbow_height(Scene &s)
{
    KDL::Frame elbow;
    mj_kdl::get_body_frame(&s.env, kElbowBody, &elbow);
    return elbow.p.z() - ex::kTableZ;
}

static Metrics
  run_selfcheck(Scene &s, RneaController &ctrl, Admittance &a, const KDL::Frame &nominal)
{
    Metrics m;
    m.elbow_start = elbow_height(s);
    m.elbow_min   = m.elbow_start;

    const double t0 = s.env.data->time;
    while (s.env.data->time - t0 < kTeachTime) {
        const double t = s.env.data->time - t0;
        mj_kdl::update(&s.env);
        close_gripper(s);
        const KDL::Frame target = control(s, ctrl, a, nominal, spiral_force(t));
        m.helix_react           = std::max(m.helix_react, norm3(a.offset));
        m.helix_track_err       = std::max(m.helix_track_err, norm3(ctrl.tcp().p - target.p));
        if (!mj_kdl::step(&s.env)) break;
        m.elbow_min = std::min(m.elbow_min, elbow_height(s));
        mj_kdl::pace_realtime(&s.env);
    }

    const double th = s.env.data->time;
    while (s.env.data->time - th < kHandoffTareTime) {
        mj_kdl::update(&s.env);
        close_gripper(s);
        const KDL::Frame target = control(s, ctrl, a, nominal, KDL::Vector::Zero());
        m.helix_track_err       = std::max(m.helix_track_err, norm3(ctrl.tcp().p - target.p));
        if (!mj_kdl::step(&s.env)) break;
        m.elbow_min = std::min(m.elbow_min, elbow_height(s));
        mj_kdl::pace_realtime(&s.env);
    }

    mj_kdl::update(&s.env);
    a.bias = ft_force_world(s.robot);
    for (int i = 0; i < kHandoffSteps; ++i) {
        mj_kdl::update(&s.env);
        close_gripper(s);
        const KDL::Vector f = external_force(s.robot, a);
        m.handoff_force     = std::max(m.handoff_force, norm3(f));
        control(s, ctrl, a, nominal, f);
        if (!mj_kdl::step(&s.env)) break;
        m.elbow_min = std::min(m.elbow_min, elbow_height(s));
        mj_kdl::pace_realtime(&s.env);
    }

    const double ts = s.env.data->time;
    while (s.env.data->time - ts < kSettleTime) {
        mj_kdl::update(&s.env);
        close_gripper(s);
        const KDL::Frame target = control(s, ctrl, a, nominal, KDL::Vector::Zero());
        m.helix_settle_err      = std::max(m.helix_settle_err, norm3(ctrl.tcp().p - target.p));
        if (!mj_kdl::step(&s.env)) break;
        m.elbow_min = std::min(m.elbow_min, elbow_height(s));
        mj_kdl::pace_realtime(&s.env);
    }

    const KDL::Vector push(kSelfcheckPush[0], kSelfcheckPush[1], kSelfcheckPush[2]);
    const KDL::Vector pre_push      = a.offset;
    KDL::Vector       settled       = pre_push;
    bool              have_recovery = false;
    bool              have_settled  = false;
    const double      tp            = s.env.data->time;
    while (s.env.data->time - tp < kPushTime) {
        const double t = s.env.data->time - tp;
        s.push->wrench = KDL::Wrench(t < 1.0 ? push : KDL::Vector::Zero(), KDL::Vector::Zero());
        mj_kdl::update(&s.env);
        close_gripper(s);
        const KDL::Frame target = control(s, ctrl, a, nominal, external_force(s.robot, a));
        if (!have_recovery && t >= 2.0) {
            m.push_recovery_err = norm3(ctrl.tcp().p - target.p);
            have_recovery       = true;
        }
        // Once the release transient (~1.5 s) has died; the drift is judged from here to the end.
        if (!have_settled && t >= 2.5) {
            settled      = a.offset;
            have_settled = true;
        }
        if (!mj_kdl::step(&s.env)) break;
        m.elbow_min = std::min(m.elbow_min, elbow_height(s));
        mj_kdl::pace_realtime(&s.env);
    }
    s.push->wrench = KDL::Wrench::Zero();

    const KDL::Vector response = settled - pre_push;
    m.push_response            = norm3(response);
    m.push_dy                  = response.y();
    m.hold_drift               = norm3(a.offset - settled);
    return m;
}

static bool report(const Metrics &m)
{
    std::cout << std::fixed << std::setprecision(4)
              << "helix force response (max offset): " << m.helix_react << " m\n"
              << "helix TCP tracking error:          " << m.helix_track_err << " m\n"
              << "helix settle error:                " << m.helix_settle_err << " m\n"
              << "FT handoff residual force:         " << m.handoff_force << " N\n"
              << "FT push response (offset norm):    " << m.push_response << " m\n"
              << "FT push response (offset dY):      " << m.push_dy << " m\n"
              << "push release recovery error:       " << m.push_recovery_err << " m\n"
              << "hold drift after push released:    " << m.hold_drift << " m\n"
              << "elbow height start / lowest:       " << m.elbow_start << " / " << m.elbow_min
              << " m (limit " << kMinElbowHeight << " m)\n";
    return m.helix_react > 0.10 && m.helix_track_err < 0.006 && m.helix_settle_err < 0.006
           && m.handoff_force == 0.0 && m.push_response > 0.12 && m.push_recovery_err < 0.002
           && m.hold_drift < 0.001 && m.elbow_min >= kMinElbowHeight;
}

// The same sequence with the viewer: helix, tare, then the mouse pushes the tool; ends on its own.
static void run_gui(Scene &s, RneaController &ctrl, Admittance &a, const KDL::Frame &nominal)
{
    mj_kdl::Viewer *viewer = &s.env.viewer;
    mj_kdl::set_free_camera(viewer, 1.55, 145.0, -24.0, { 0.05, 0.0, ex::kTableZ + 0.35 });
    if (!mj_kdl::open_viewer(&s.env, "ex_admittance_ft")) return;

    const double run_time =
      kTeachTime + kHandoffTareTime + kHandoffSteps * s.spec.timestep + kSettleTime + kPushTime;
    double      start         = s.env.data->time;
    bool        handoff_tared = false;
    bool        have_prev     = false;
    KDL::Vector target_prev, tcp_prev;
    int         trace_step = 0;

    s.restarted = false;
    while (mj_kdl::is_running(viewer)) {
        if (s.restarted) {
            s.restarted   = false;
            a             = Admittance{};
            start         = s.env.data->time;
            handoff_tared = false;
            have_prev     = false;
            mj_kdl::clear_trace(viewer);
        }
        const double t = s.env.data->time - start;
        if (t >= run_time) break;
        mj_kdl::update(&s.env);
        close_gripper(s);

        KDL::Vector force = KDL::Vector::Zero();
        if (t < kTeachTime) {
            force = spiral_force(t);
        } else if (t >= kTeachTime + kHandoffTareTime) {
            if (!handoff_tared) {
                a.bias        = ft_force_world(s.robot);
                handoff_tared = true;
            }
            force = external_force(s.robot, a);
        }
        const KDL::Frame target = control(s, ctrl, a, nominal, force);

        KDL::Frame world_base;
        mj_kdl::get_body_frame(&s.env, "base_link", &world_base);
        const KDL::Vector target_xyz = world_base * target.p;
        const KDL::Vector tcp_xyz    = world_base * ctrl.tcp().p;
        ++trace_step;
        if (have_prev && trace_step % 5 == 0) {
            const float yellow[4] = { 1.0f, 0.95f, 0.0f, 1.0f };
            const float green[4]  = { 0.0f, 1.0f, 0.2f, 1.0f };
            mj_kdl::add_trace_segment(viewer, target_prev, target_xyz, yellow);
            mj_kdl::add_trace_segment(viewer, tcp_prev, tcp_xyz, green);
        }
        target_prev = target_xyz;
        tcp_prev    = tcp_xyz;
        have_prev   = true;

        if (!mj_kdl::step(&s.env)) break;
        mj_kdl::pace_realtime(&s.env);
    }
}

int main(int argc, char **argv)
{
    const bool headless = ex::parse_args(argc, argv).headless;

    Scene s;
    if (!build_scene(s)) {
        std::cerr << "failed to build admittance FT scene\n";
        return 1;
    }
    if (!mj_kdl::set_control_mode(&s.robot, mj_kdl::CtrlMode::TORQUE)) return 1;
    const KDL::JntArray q_home = ex::home_q(s.robot.chain.getNrOfJoints());
    RneaController      ctrl(s, q_home);

    s.env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&s.robot, q_home);
        s.restarted = true;
    };
    mj_kdl::reset(&s.env);

    Admittance a;
    settle_and_tare(s, ctrl, a);
    const KDL::Frame nominal = ctrl.tcp();

    std::cout << std::fixed << std::setprecision(3) << "FT bias: [" << a.bias.x() << ", "
              << a.bias.y() << ", " << a.bias.z() << "] N\n";
    int rc = 0;
    if (headless) {
        rc = ex::verdict(
          report(run_selfcheck(s, ctrl, a, nominal)),
          "the admittance responded "
          "to the helix and the FT push and held on release"
        );
    } else {
        run_gui(s, ctrl, a, nominal);
        std::cout << std::fixed << std::setprecision(4) << "final offset: [" << a.offset.x() << ", "
                  << a.offset.y() << ", " << a.offset.z() << "] m\n";
    }
    mj_kdl::cleanup(&s.env);
    return rc;
}
