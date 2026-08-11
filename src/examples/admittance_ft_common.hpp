#pragma once

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainhdsolver_vereshchagin_fixed_joint.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/chainjnttojacsolver.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace admittance_ft
{

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTableZ      = 0.70;
static constexpr double kMAdm        = 8.0;
static constexpr double kDAdm        = 80.0;
static constexpr double kKAdm        = 0.0;
static constexpr double kForceDeadband = 2.5;
static constexpr double kMaxOffset     = 0.20;
static constexpr double kMaxVel        = 0.25;
static constexpr double kTeachTime     = 16.0;
static constexpr double kTeachRadius   = 0.04;
static constexpr double kTeachRise     = 0.10;
static constexpr double kTeachTurns    = 5.0;
static constexpr int    kSettleSteps   = 300;
static constexpr double kHandoffTareTime = 1.0;
static constexpr const char *kToolBody = "g_base";
static constexpr const char *kGripperActuator = "g_fingers_actuator";
static constexpr double kSelfcheckPush[3] = { 8.0, 12.0, 6.0 };

inline double clamp(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

inline KDL::JntArray home_q(unsigned n)
{
    KDL::JntArray q(n);
    for (unsigned i = 0; i < n; ++i) q(i) = kHomePose[i];
    return q;
}

inline KDL::Vector vclamp(const KDL::Vector &v, double limit)
{
    return KDL::Vector(
      clamp(v.x(), -limit, limit),
      clamp(v.y(), -limit, limit),
      clamp(v.z(), -limit, limit)
    );
}

inline double norm3(const KDL::Vector &v) { return std::sqrt(v.x() * v.x() + v.y() * v.y() + v.z() * v.z()); }

inline KDL::Rotation mj_xmat_to_kdl_rot(const double *m)
{
    return KDL::Rotation(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}

inline KDL::Frame site_frame_by_id(const mjModel *model, const mjData *data, int id)
{
    (void)model;
    const double *p = data->site_xpos + 3 * id;
    const double *R = data->site_xmat + 9 * id;
    return KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
}

inline KDL::Frame current_tcp(KDL::ChainFkSolverPos_recursive &fk, const mj_kdl::Robot &robot)
{
    KDL::JntArray q(robot.n_joints);
    for (int i = 0; i < robot.n_joints; ++i) q(i) = robot.jnt_pos_msr[i];
    KDL::Frame out;
    fk.JntToCart(q, out);
    return out;
}

struct AdmState
{
    KDL::Vector bias = KDL::Vector::Zero();
    KDL::Vector offset = KDL::Vector::Zero();
    KDL::Vector vel = KDL::Vector::Zero();
};

inline KDL::Vector spiral_force(double t)
{
    if (t < 0.0 || t > kTeachTime) return KDL::Vector::Zero();
    const double theta = 2.0 * M_PI * kTeachTurns * t / kTeachTime;
    const double theta_dot = 2.0 * M_PI * kTeachTurns / kTeachTime;
    return KDL::Vector(
      kDAdm * (-kTeachRadius * theta_dot * std::sin(theta)),
      kDAdm * ( kTeachRadius * theta_dot * std::cos(theta)),
      kDAdm * ( kTeachRise / kTeachTime)
    );
}

inline void admittance_update(AdmState &s, const KDL::Vector &force, double dt)
{
    if (norm3(force) == 0.0) {
        s.vel = KDL::Vector::Zero();
        return;
    }
    KDL::Vector acc(
      (force.x() - kDAdm * s.vel.x() - kKAdm * s.offset.x()) / kMAdm,
      (force.y() - kDAdm * s.vel.y() - kKAdm * s.offset.y()) / kMAdm,
      (force.z() - kDAdm * s.vel.z() - kKAdm * s.offset.z()) / kMAdm
    );
    s.vel = vclamp(s.vel + acc * dt, kMaxVel);
    s.offset = vclamp(s.offset + s.vel * dt, kMaxOffset);
}

inline void set_body_wrench(mjModel *model, mjData *data, const char *body, const KDL::Vector &force)
{
    const int id = mj_name2id(model, mjOBJ_BODY, body);
    if (id < 0) return;
    data->xfrc_applied[6 * id + 0] = force.x();
    data->xfrc_applied[6 * id + 1] = force.y();
    data->xfrc_applied[6 * id + 2] = force.z();
    data->xfrc_applied[6 * id + 3] = 0.0;
    data->xfrc_applied[6 * id + 4] = 0.0;
    data->xfrc_applied[6 * id + 5] = 0.0;
}

inline void close_gripper(mjModel *model, mjData *data)
{
    const int id = mj_name2id(model, mjOBJ_ACTUATOR, kGripperActuator);
    if (id >= 0) data->ctrl[id] = 255.0;
}

inline KDL::Vector tare_force(const mj_kdl::Robot &robot)
{
    const mj_kdl::ForceTorqueSensor *ft = mj_kdl::find_ft_sensor(&robot, "wrist_ft");
    if (!ft || ft->frame_site_id < 0) return KDL::Vector::Zero();
    return site_frame_by_id(robot.model, robot.data, ft->frame_site_id).M * ft->wrench.force;
}

inline KDL::Vector measured_force(const mj_kdl::Robot &robot, const AdmState &s)
{
    const mj_kdl::ForceTorqueSensor *ft = mj_kdl::find_ft_sensor(&robot, "wrist_ft");
    if (!ft || ft->frame_site_id < 0) return KDL::Vector::Zero();
    const KDL::Vector f_world = site_frame_by_id(robot.model, robot.data, ft->frame_site_id).M * ft->wrench.force;
    const KDL::Vector f_ext = s.bias - f_world;
    return norm3(f_ext) < kForceDeadband ? KDL::Vector::Zero() : f_ext;
}

struct SceneHandles
{
    mjModel *model = nullptr;
    mjData *data = nullptr;
    mj_kdl::SceneSpec scene;
    mj_kdl::Env env;
    mj_kdl::Robot robot;
    int tool_body_id = -1;

    void cleanup()
    {
        mj_kdl::cleanup(&robot);
        if (model && data) mj_kdl::destroy_scene(model, data);
        model = nullptr;
        data = nullptr;
    }
};

inline bool build_scene(SceneHandles &h)
{
    const std::string arm = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string ft = mj_kdl_examples::asset("ft_sensor.xml");
    const std::string gripper = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string table_path = mj_kdl_examples::asset("table.xml");

    mj_kdl::SceneObject table;
    table.name = "table";
    table.mjcf_path = table_path;
    table.pos[0] = 0.0;
    table.pos[1] = 0.0;
    table.pos[2] = kTableZ;
    table.fixed = true;

    mj_kdl::AttachmentSpec ft_spec;
    ft_spec.mjcf_path = ft.c_str();
    ft_spec.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };

    mj_kdl::AttachmentSpec gripper_spec;
    gripper_spec.mjcf_path = gripper.c_str();
    gripper_spec.attach_to = { mj_kdl::AttachKind::Site, "wrist_ft_site" };
    gripper_spec.prefix = "g_";

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path = arm.c_str();
    const std::string table_top = mj_kdl::scene_object_site_name(table, "table_top");
    robot_spec.attach_to = { mj_kdl::AttachKind::Site, table_top.c_str() };
    robot_spec.attachments.push_back(ft_spec);
    robot_spec.attachments.push_back(gripper_spec);

    h.scene.timestep = 0.002;
    h.scene.add_floor = true;
    h.scene.add_skybox = true;
    h.scene.objects.push_back(table);
    h.scene.robots.push_back(robot_spec);

    if (!mj_kdl::build_scene(&h.model, &h.data, &h.scene)) return false;

    mj_kdl::ForceTorqueSensorSpec ft_sensor;
    ft_sensor.name = "wrist_ft";
    ft_sensor.frame_site = "wrist_ft_site";

    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = "g_base";
    tool.tcp_site = "g_pinch";
    tool.ft_sensors.push_back(ft_sensor);

    if (!mj_kdl::init_robot_from_mjcf(&h.robot, h.model, h.data, "base_link", "bracelet_link", "", &tool)) return false;
    h.env.spec = h.scene;
    h.env.model = h.model;
    h.env.data = h.data;
    mj_kdl::env_add_robot(&h.env, &h.robot);
    h.tool_body_id = mj_name2id(h.model, mjOBJ_BODY, kToolBody);
    return h.tool_body_id >= 0;
}

class Controller
{
public:
    virtual ~Controller() = default;
    virtual const char *name() const = 0;
    virtual mj_kdl::CtrlMode mode() const = 0;
    virtual void reset() = 0;
    virtual void track(const KDL::Frame &target) = 0;
};

inline KDL::Frame admittance_step(
  mj_kdl::Robot &robot,
  Controller &ctrl,
  AdmState &state,
  const KDL::Frame &nominal,
  const KDL::Vector &force,
  double dt
)
{
    (void)robot;
    admittance_update(state, force, dt);
    KDL::Frame target(nominal.M, nominal.p + state.offset);
    ctrl.track(target);
    return target;
}

inline void settle_and_tare(SceneHandles &h, Controller &ctrl, AdmState &state)
{
    mj_kdl::update(&h.robot);
    KDL::ChainFkSolverPos_recursive fk(h.robot.chain);
    const KDL::Frame home = current_tcp(fk, h.robot);
    for (int i = 0; i < kSettleSteps; ++i) {
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        ctrl.track(home);
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }
    mj_kdl::update(&h.robot);
    state.bias = tare_force(h.robot);
}

struct Metrics
{
    double helix_react = 0.0;
    double helix_track_err = 0.0;
    double helix_settle_err = 0.0;
    double handoff_force = 0.0;
    double push_response = 0.0;
    double push_dy = 0.0;
    double push_recovery_err = 0.0;
    double hold_drift = 0.0;
};

inline Metrics run_selfcheck(SceneHandles &h, Controller &ctrl, AdmState &state, const KDL::Frame &nominal)
{
    KDL::ChainFkSolverPos_recursive fk(h.robot.chain);
    Metrics m;

    const double t0 = h.data->time;
    while (h.data->time - t0 < kTeachTime) {
        const double t = h.data->time - t0;
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        KDL::Frame target = admittance_step(h.robot, ctrl, state, nominal, spiral_force(t), h.scene.timestep);
        KDL::Frame tcp = current_tcp(fk, h.robot);
        m.helix_react = std::max(m.helix_react, norm3(state.offset));
        m.helix_track_err = std::max(m.helix_track_err, norm3(tcp.p - target.p));
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }

    const double th = h.data->time;
    while (h.data->time - th < kHandoffTareTime) {
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        KDL::Frame target = admittance_step(h.robot, ctrl, state, nominal, KDL::Vector::Zero(), h.scene.timestep);
        KDL::Frame tcp = current_tcp(fk, h.robot);
        m.helix_track_err = std::max(m.helix_track_err, norm3(tcp.p - target.p));
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }

    mj_kdl::update(&h.robot);
    state.bias = tare_force(h.robot);
    for (int i = 0; i < 100; ++i) {
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        KDL::Vector f = measured_force(h.robot, state);
        m.handoff_force = std::max(m.handoff_force, norm3(f));
        admittance_step(h.robot, ctrl, state, nominal, f, h.scene.timestep);
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }

    const double ts = h.data->time;
    while (h.data->time - ts < 0.5) {
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        KDL::Frame target = admittance_step(h.robot, ctrl, state, nominal, KDL::Vector::Zero(), h.scene.timestep);
        KDL::Frame tcp = current_tcp(fk, h.robot);
        m.helix_settle_err = std::max(m.helix_settle_err, norm3(tcp.p - target.p));
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }

    const KDL::Vector pre_push = state.offset;
    const double tp = h.data->time;
    KDL::Vector settled = pre_push;
    bool have_recovery = false;
    while (h.data->time - tp < 4.0) {
        const double t = h.data->time - tp;
        set_body_wrench(
          h.model, h.data, kToolBody,
          t < 1.0 ? KDL::Vector(kSelfcheckPush[0], kSelfcheckPush[1], kSelfcheckPush[2]) : KDL::Vector::Zero()
        );
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);
        KDL::Frame target = admittance_step(h.robot, ctrl, state, nominal, measured_force(h.robot, state), h.scene.timestep);
        KDL::Frame tcp = current_tcp(fk, h.robot);
        if (!have_recovery && t >= 2.0) {
            m.push_recovery_err = norm3(tcp.p - target.p);
            have_recovery = true;
        }
        if (t >= 2.5) settled = state.offset;
        if (!mj_kdl::step(&h.robot)) break;
        mj_kdl::pace_realtime(&h.robot);
    }
    set_body_wrench(h.model, h.data, kToolBody, KDL::Vector::Zero());

    const KDL::Vector response = settled - pre_push;
    m.push_response = norm3(response);
    m.push_dy = response.y();
    m.hold_drift = norm3(state.offset - settled);
    return m;
}

inline int finish_headless(const Metrics &m)
{
    std::cout << std::fixed << std::setprecision(4)
              << "helix force response (max offset): " << m.helix_react << " m\n"
              << "helix TCP tracking error:          " << m.helix_track_err << " m\n"
              << "helix settle error:                " << m.helix_settle_err << " m\n"
              << "FT handoff residual force:         " << m.handoff_force << " N\n"
              << "FT push response (offset norm):    " << m.push_response << " m\n"
              << "FT push response (offset dY):      " << m.push_dy << " m\n"
              << "push release recovery error:       " << m.push_recovery_err << " m\n"
              << "hold drift after push released:    " << m.hold_drift << " m\n";
    if (m.helix_react <= 0.05 || m.helix_track_err >= 0.006 || m.helix_settle_err >= 0.004
        || m.handoff_force != 0.0 || m.push_response <= 0.05 || m.push_recovery_err >= 0.006
        || m.hold_drift >= 0.01) {
        return 1;
    }
    std::cout << "OK: admittance responded to helix + FT push and held on release\n";
    return 0;
}

inline void run_gui(SceneHandles &h, Controller &ctrl, AdmState &state, const KDL::Frame &nominal)
{
    mj_kdl::Viewer viewer{};
    mj_kdl::set_free_camera(&viewer, 1.55, 145.0, -24.0, { 0.05, 0.0, kTableZ + 0.35 });
    if (!mj_kdl::init_window_sim(&viewer, &h.robot, ctrl.name())) return;

    KDL::ChainFkSolverPos_recursive fk(h.robot.chain);
    double start = h.data->time;
    double prev = h.data->time;
    bool handoff_tared = false;
    bool have_prev = false;
    KDL::Vector target_prev, tcp_prev;
    int trace_step = 0;

    while (mj_kdl::is_running(&viewer)) {
        if (h.data->time < prev - 1e-6) {
            mj_kdl::reset(&h.env);
            ctrl.reset();
            state = AdmState{};
            start = h.data->time;
            handoff_tared = false;
            have_prev = false;
        }
        prev = h.data->time;
        const double t = h.data->time - start;
        mj_kdl::update(&h.robot);
        close_gripper(h.model, h.data);

        KDL::Vector force = KDL::Vector::Zero();
        if (t < kTeachTime) {
            force = spiral_force(t);
        } else if (t >= kTeachTime + kHandoffTareTime) {
            if (!handoff_tared) {
                state.bias = tare_force(h.robot);
                handoff_tared = true;
            }
            force = measured_force(h.robot, state);
        }
        KDL::Frame target = admittance_step(h.robot, ctrl, state, nominal, force, h.scene.timestep);
        KDL::Frame tcp = current_tcp(fk, h.robot);
        KDL::Frame world_base;
        mj_kdl::get_body_frame(h.model, h.data, "base_link", &world_base);
        KDL::Vector target_xyz = world_base * target.p;
        KDL::Vector tcp_xyz = world_base * tcp.p;
        ++trace_step;
        if (have_prev && trace_step % 5 == 0) {
            const float yellow[4] = { 1.0f, 0.95f, 0.0f, 1.0f };
            const float green[4] = { 0.0f, 1.0f, 0.2f, 1.0f };
            mj_kdl::add_trace_segment(&viewer, target_prev, target_xyz, yellow);
            mj_kdl::add_trace_segment(&viewer, tcp_prev, tcp_xyz, green);
        }
        target_prev = target_xyz;
        tcp_prev = tcp_xyz;
        have_prev = true;

        if (!mj_kdl::step(&viewer, h.model, h.data)) break;

        mj_kdl::pace_realtime(&viewer, h.model);
    }
    set_body_wrench(h.model, h.data, kToolBody, KDL::Vector::Zero());
    mj_kdl::cleanup(&viewer);
}

inline int run(int argc, char **argv, std::unique_ptr<Controller> (*make_controller)(SceneHandles &))
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    SceneHandles h;
    if (!build_scene(h)) {
        std::cerr << "failed to build admittance FT scene\n";
        h.cleanup();
        return 1;
    }
    std::unique_ptr<Controller> ctrl = make_controller(h);
    h.robot.ctrl_mode = ctrl->mode();

    const KDL::JntArray q_home = home_q(h.robot.n_joints);
    h.env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&h.robot, q_home, false);
        ctrl->reset();
        set_body_wrench(h.model, h.data, kToolBody, KDL::Vector::Zero());
    };
    mj_kdl::reset(&h.env);

    AdmState state;
    settle_and_tare(h, *ctrl, state);
    KDL::ChainFkSolverPos_recursive fk(h.robot.chain);
    KDL::Frame nominal = current_tcp(fk, h.robot);

    std::cout << std::fixed << std::setprecision(3)
              << "FT bias: [" << state.bias.x() << ", " << state.bias.y() << ", " << state.bias.z() << "] N\n";
    int rc = 0;
    if (headless) {
        rc = finish_headless(run_selfcheck(h, *ctrl, state, nominal));
    } else {
        run_gui(h, *ctrl, state, nominal);
        std::cout << std::fixed << std::setprecision(4)
                  << "final offset: [" << state.offset.x() << ", " << state.offset.y() << ", "
                  << state.offset.z() << "] m\n";
    }
    h.cleanup();
    return rc;
}

} // namespace admittance_ft
