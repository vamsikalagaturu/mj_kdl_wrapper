/* ex_table_pick_place.cpp
 * Kinova GEN3 + Robotiq 2F-85 on a table picks a cube and places it 0.24 m to the side, with
 * joint impedance (PD + KDL gravity) in TORQUE mode tracking IK waypoints. While the arm carries
 * the cube (PLACE_ABOVE), a scripted 20 N push across the motion on bracelet_link makes the arm
 * give way and spring back, as the joint stiffness Kp predicts.
 *
 * Usage:
 *   ex_table_pick_place [--headless]
 *
 * --headless skips the viewer and exits 1 unless the cube rests on the table within kMaxPlaceErr
 * of the place spot, the push deflected the TCP by at least kMinDeflection and within
 * kDeflectionRatio of J K^-1 J^T F, the arm sprang back, the cube stayed in the gripper and the
 * elbow stayed above kMinElbowHeight. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainjnttojacsolver.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

namespace ex = mj_kdl_examples;

static constexpr double kMaxPlaceErr = 0.005; // [m] in the table plane

static constexpr const char *kPushBody  = "bracelet_link";
static constexpr const char *kPushPhase = "PLACE_ABOVE";
static constexpr double      kPushForce = 20.0; // [N] along world +x, across the carry along +y
static constexpr double      kPushOn    = 0.8;  // [s] into the phase
static constexpr double      kPushOff   = 2.0;  // [s] into the phase
static constexpr double      kPushRamp  = 0.2;  // [s] rise and fall
static constexpr double      kSettle    = 0.5;  // [s] after the push, when the residual is taken

static constexpr double kMinDeflection     = 0.010; // [m]
static constexpr double kDeflectionRatio[] = { 0.85, 1.15 }; // measured / predicted
static constexpr double kMaxResidual       = 0.004;          // [m]
static constexpr double kMaxCubeSlip       = 0.010; // [m] cube centre from the TCP while pushed
static constexpr double kMinElbowHeight    = 0.45;  // [m] forearm_link origin above the table

struct Metrics
{
    KDL::Vector lag0;                 // TCP minus its reference just before the push
    double      peak      = 0.0;      // [m] along the push
    double      predicted = 0.0;      // [m] J K^-1 J^T F at the peak
    double      residual  = -1.0;     // [m] along the push, kSettle after it; < 0 until taken
    double      cube_slip = 0.0;      // [m]
    double      elbow_min = INFINITY; // [m] world z
};

static double push_scale(double t_rel)
{
    return std::min(
      ex::clamp01((t_rel - kPushOn) / kPushRamp), ex::clamp01((kPushOff - t_rel) / kPushRamp)
    );
}

int main(int argc, char *argv[])
{
    const bool headless = ex::parse_args(argc, argv).headless;

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
    mj_kdl::SceneWrenchSlot   *push = mj_kdl::bind_scene_wrench(&env.scene, kPushBody);
    if (!fingers || !cube || !push) return 1;

    ex::PickPlaceWaypoints wp;
    if (!ex::solve_pick_place(robot, wp)) return 1;
    const unsigned                  n = robot.chain.getNrOfJoints();
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainJntToJacSolver        jac_solver(robot.chain);
    KDL::Jacobian                   jac(n), jac_push(n);
    KDL::JntArray                   q(n), q_ref(n);
    const KDL::Vector               push_dir(1.0, 0.0, 0.0);
    const KDL::Vector               base(0.0, 0.0, ex::kTableZ);
    Metrics                         m;

    bool restart = false;
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, wp.home);
        ex::place_cube(env);
        ctx->data->ctrl[fingers->ctrl_id] = 0.0;
        ex::prime_gravity(robot, dyn, wp.home);
        m       = {};
        restart = true;
    };
    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    const auto control = [&](std::size_t, const KDL::JntArray &q_des) {
        q_ref = q_des;
        ex::pd_gravity(robot, dyn, q_des, ex::kKp, ex::kKd);
    };
    // Measured against the reference of the same cycle; the push lands at the next update().
    const auto after_step = [&](const ex::Phase &phase, double t_rel) {
        KDL::Frame elbow;
        mj_kdl::get_body_frame(&env, "forearm_link", &elbow);
        m.elbow_min               = std::min(m.elbow_min, elbow.p.z());
        const bool        pushing = std::string(phase.name) == kPushPhase;
        const KDL::Vector force   = push_dir * (pushing ? kPushForce * push_scale(t_rel) : 0.0);
        push->wrench              = KDL::Wrench(force, KDL::Vector::Zero());
        if (!pushing) return;

        KDL::Frame tcp, ref;
        ex::read_q(robot, q);
        fk.JntToCart(q, tcp);
        fk.JntToCart(q_ref, ref);
        m.cube_slip           = std::max(m.cube_slip, (cube->pose.p - base - tcp.p).Norm());
        const KDL::Vector lag = tcp.p - ref.p;
        if (t_rel < kPushOn) m.lag0 = lag;
        const double deflection = KDL::dot(lag - m.lag0, push_dir);
        if (t_rel >= kPushOff + kSettle && m.residual < 0.0) m.residual = std::abs(deflection);
        // Only at full force: on the ramps the arm lags the changing force.
        const bool full = t_rel >= kPushOn + kPushRamp && t_rel <= kPushOff - kPushRamp;
        if (!full || deflection <= m.peak) return;

        // dq = K^-1 J_push^T F at the pushed body's centre of mass, seen at the TCP through J.
        jac_solver.JntToJac(q_ref, jac);
        const double *com = env.data->xipos + 3 * push->body_id;
        jac_push          = jac;
        jac_push.changeRefPoint(KDL::Vector(com[0], com[1], com[2]) - base - ref.p);
        KDL::Vector dx = KDL::Vector::Zero();
        for (unsigned i = 0; i < n; ++i)
            dx += jac.getColumn(i).vel * (KDL::dot(jac_push.getColumn(i).vel, force) / ex::kKp[i]);
        m.peak      = deflection;
        m.predicted = KDL::dot(dx, push_dir);
    };
    const bool completed = ex::run_phases(
      env, { { &robot, fingers } }, ex::pick_place_phases(wp), restart, control, after_step
    );
    mj_kdl::update(&env);

    const KDL::Vector c        = cube->pose.p;
    const double      place_xy = std::hypot(c.x() - ex::kPlaceXY[0], c.y() - ex::kPlaceXY[1]);
    const bool        on_table = std::abs(c.z() - (ex::kTableZ + ex::kCubeHS)) < 0.002;
    const double      ratio    = m.predicted > 0.0 ? m.peak / m.predicted : 0.0;
    const double      elbow_h  = m.elbow_min - ex::kTableZ;
    std::cout << std::fixed << std::setprecision(4) << "cube final position: [" << c.x() << ", "
              << c.y() << ", " << c.z() << "] place error " << place_xy * 1000.0 << " mm (limit "
              << kMaxPlaceErr * 1000.0 << " mm)" << (on_table ? "" : ", not on the table") << "\n"
              << "push deflection: " << m.peak * 1000.0 << " mm (at least "
              << kMinDeflection * 1000.0 << " mm), stiffness predicts " << m.predicted * 1000.0
              << " mm, ratio " << ratio << " (limits " << kDeflectionRatio[0] << ".."
              << kDeflectionRatio[1] << ")\n"
              << "deflection " << kSettle << " s after the push: " << m.residual * 1000.0
              << " mm (limit " << kMaxResidual * 1000.0 << " mm)\n"
              << "cube from the TCP while pushed: " << m.cube_slip * 1000.0 << " mm (limit "
              << kMaxCubeSlip * 1000.0 << " mm)\n"
              << "lowest elbow above the table: " << elbow_h * 1000.0 << " mm (limit "
              << kMinElbowHeight * 1000.0 << " mm)\n";
    mj_kdl::cleanup(&env);
    const bool ok = completed && on_table && place_xy <= kMaxPlaceErr && m.peak >= kMinDeflection
                    && ratio >= kDeflectionRatio[0] && ratio <= kDeflectionRatio[1]
                    && m.residual >= 0.0 && m.residual <= kMaxResidual
                    && m.cube_slip <= kMaxCubeSlip && elbow_h >= kMinElbowHeight;
    return headless ? ex::verdict(ok, "the cube was placed through the push") : 0;
}
