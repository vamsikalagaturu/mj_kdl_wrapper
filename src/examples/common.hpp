#pragma once

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace mj_kdl_examples {

// The bundled 2F-85's ctrl is its driver joint angle: ctrlrange 0..0.82, joint range 0..0.8.
constexpr double kGripperClosed = 0.82;

// Kinova Gen3 home pose [rad].
constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

// Gen3 joint impedance gains [Nm/rad], [Nm s/rad].
constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

constexpr double kTableZ     = 0.70; // tabletop height of the bundled table [m]
constexpr double kCubeHS     = 0.02; // pick cube half-size [m]
constexpr double kPickXY[2]  = { 0.40, 0.00 };
constexpr double kPlaceXY[2] = { 0.40, 0.24 };

struct Args
{
    bool        headless = false;
    bool        record   = false;
    std::string record_path;
};

// --headless skips the viewer; --record [out.mp4] records offscreen and implies --headless.
inline Args parse_args(int argc, char **argv, const std::string &default_record_path = "")
{
    Args args;
    args.record_path = default_record_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--headless") {
            args.headless = true;
        } else if (arg == "--record") {
            args.record   = true;
            args.headless = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') args.record_path = argv[++i];
        }
    }
    return args;
}

inline int verdict(bool ok, const std::string &goal)
{
    std::cout << (ok ? "PASS: " : "FAIL: ") << goal << "\n";
    return ok ? 0 : 1;
}

inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

inline double clamp_abs(double v, double limit) { return std::clamp(v, -limit, limit); }

inline KDL::JntArray home_q(unsigned n)
{
    KDL::JntArray q(n);
    for (unsigned i = 0; i < n; ++i) q(i) = kHomePose[i];
    return q;
}

inline void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

inline void read_q(const mj_kdl::Robot &robot, KDL::JntArray &q)
{
    for (unsigned i = 0; i < q.rows(); ++i) q(i) = robot.jnt_pos_msr[i];
}

inline void read_q(const mj_kdl::Robot &robot, KDL::JntArray &q, KDL::JntArray &qd)
{
    for (unsigned i = 0; i < q.rows(); ++i) {
        q(i)  = robot.jnt_pos_msr[i];
        qd(i) = robot.jnt_vel_msr[i];
    }
}

inline double max_abs_joint_err(const mj_kdl::Robot &robot, const KDL::JntArray &q)
{
    double max_err = 0.0;
    for (unsigned i = 0; i < q.rows(); ++i)
        max_err = std::max(max_err, std::abs(q(i) - robot.jnt_pos_msr[i]));
    return max_err;
}

// TCP of the chain at the measured joint positions, in the robot base frame.
inline KDL::Frame tcp_frame(KDL::ChainFkSolverPos_recursive &fk, const mj_kdl::Robot &robot)
{
    KDL::JntArray q(robot.chain.getNrOfJoints());
    read_q(robot, q);
    KDL::Frame out;
    fk.JntToCart(q, out);
    return out;
}

// jnt_trq_cmd = g(q) + kp * (q_des - q) - kd * qdot; kp/kd null for gravity alone.
inline void pd_gravity(
  mj_kdl::Robot       &robot,
  KDL::ChainDynParam  &dyn,
  const KDL::JntArray &q_des,
  const double        *kp = nullptr,
  const double        *kd = nullptr
)
{
    const unsigned n = robot.chain.getNrOfJoints();
    KDL::JntArray  q(n), g(n);
    read_q(robot, q);
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) {
        robot.jnt_trq_cmd[i] = g(i);
        if (kp) robot.jnt_trq_cmd[i] += kp[i] * (q_des(i) - q(i)) - kd[i] * robot.jnt_vel_msr[i];
    }
}

// Gravity at q as the first torque command, so the first step after a reset holds the arm.
inline void prime_gravity(mj_kdl::Robot &robot, KDL::ChainDynParam &dyn, const KDL::JntArray &q)
{
    KDL::JntArray g(q.rows());
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < q.rows(); ++i) robot.jnt_trq_cmd[i] = g(i);
}

inline mj_kdl::SceneSpec scene_spec()
{
    mj_kdl::SceneSpec spec;
    spec.timestep   = 0.002;
    spec.add_floor  = true;
    spec.add_skybox = true;
    return spec;
}

// The bundled 2F-85 on the arm's pinch site (or another site), element names prefixed "g_".
inline mj_kdl::AttachmentSpec
  gripper_attachment(const std::string &path, const std::string &site = "pinch_site")
{
    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = path;
    gripper.attach_to = { mj_kdl::AttachKind::Site, site };
    gripper.prefix    = "g_";
    return gripper;
}

// The 2F-85's mount carries mass too, so the tool starts there; the TCP is its pinch site.
inline mj_kdl::ToolFrameSpec gripper_tool(const std::string &prefix = "")
{
    mj_kdl::ToolFrameSpec tool;
    tool.tool_body = prefix + "g_base_mount";
    tool.tcp_site  = prefix + "g_pinch";
    return tool;
}

// The bundled table, its top (site "table_top") at surface_z.
inline mj_kdl::SceneObject table_object(const std::string &path, double surface_z)
{
    mj_kdl::SceneObject table;
    table.name      = "table";
    table.mjcf_path = path;
    table.pos[2]    = surface_z;
    table.fixed     = true;
    return table;
}

inline mj_kdl::SceneObject cube_object(double x, double y, double surface_z)
{
    mj_kdl::SceneObject cube;
    cube.name   = "cube";
    cube.shape  = mj_kdl::Shape::BOX;
    cube.pos[0] = x;
    cube.pos[1] = y;
    cube.pos[2] = surface_z + kCubeHS;
    for (int k = 0; k < 3; ++k) cube.size[k] = kCubeHS;
    const float rgba[4] = { 0.1f, 0.35f, 1.0f, 1.0f };
    std::copy(rgba, rgba + 4, cube.rgba);
    cube.mass        = 0.1;
    cube.condim      = mj_kdl::Condim::Torsional;
    cube.friction[0] = 0.8;
    cube.friction[1] = 0.02;
    cube.friction[2] = 0.001;
    return cube;
}

inline void place_cube(mj_kdl::Env &env)
{
    const double pos[3] = { kPickXY[0], kPickXY[1], kTableZ + kCubeHS };
    mj_kdl::set_body_pose(&env, "cube", pos);
}

// Damped-least-squares IK stepped from seed, so the answer stays on the seed's branch.
inline bool solve_near_seed(
  KDL::ChainIkSolverVel_wdls      &ik_vel,
  KDL::ChainFkSolverPos_recursive &fk,
  const mj_kdl::Robot             &robot,
  const KDL::JntArray             &seed,
  const KDL::Frame                &target,
  KDL::JntArray                   &out
)
{
    const auto error = [&] {
        KDL::Frame fk_out;
        fk.JntToCart(out, fk_out);
        return KDL::diff(fk_out, target);
    };
    const auto converged = [](const KDL::Twist &dx) {
        return dx.vel.Norm() <= 2e-3 && dx.rot.Norm() <= 2e-2;
    };
    out = seed;
    KDL::JntArray dq(out.rows());
    for (int iter = 0; iter < 300; ++iter) {
        KDL::Twist dx = error();
        if (converged(dx)) return true;

        const double vel_norm = dx.vel.Norm();
        if (vel_norm > 0.05) dx.vel = dx.vel * (0.05 / vel_norm);
        const double rot_norm = dx.rot.Norm();
        if (rot_norm > 0.20) dx.rot = dx.rot * (0.20 / rot_norm);

        if (ik_vel.CartToJnt(out, dx, dq) < 0) return false;
        for (unsigned i = 0; i < out.rows(); ++i) {
            // joint_limits are +-inf for an unlimited joint, so the clamp leaves it alone.
            const auto [lo, hi] = robot.joint_limits[i];
            out(i)              = std::clamp(out(i) + dq(i), lo, hi);
        }
    }
    return converged(error());
}

struct PickPlaceWaypoints
{
    KDL::JntArray home, pick_above, pick, lift, place_above, place;
};

// Joint waypoints of the table pick-place, for an arm standing on the table at the world origin.
inline bool solve_pick_place(const mj_kdl::Robot &robot, PickPlaceWaypoints &w)
{
    const unsigned                  n = robot.chain.getNrOfJoints();
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainIkSolverVel_wdls      ik_vel(robot.chain, 1e-5, 150);
    ik_vel.setLambda(0.05);

    w.home = home_q(n);
    // Base-frame heights: the TCP at the cube centre, then above it.
    const double z_grasp = kCubeHS;
    struct Target
    {
        const double        *xy;
        double               z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    const Target targets[] = {
        { kPickXY, z_grasp + 0.20, &w.pick_above, &w.home },
        { kPickXY, z_grasp, &w.pick, &w.pick_above },
        { kPickXY, z_grasp + 0.30, &w.lift, &w.pick },
        { kPlaceXY, z_grasp + 0.20, &w.place_above, &w.lift },
        { kPlaceXY, z_grasp, &w.place, &w.place_above },
    };
    for (const Target &t : targets) {
        const KDL::Frame target(robot.tip_T_tcp.M, KDL::Vector(t.xy[0], t.xy[1], t.z));
        if (!solve_near_seed(ik_vel, fk, robot, *t.seed, target, *t.out)) {
            std::cerr << "IK failed for [" << t.xy[0] << ", " << t.xy[1] << ", " << t.z << "]\n";
            return false;
        }
    }
    return true;
}

struct Phase
{
    const char          *name;
    const KDL::JntArray *target;
    double               duration;   // [s] to interpolate from the entry pose to target
    double               timeout;    // [s] ends the phase even if not settled
    double               settle_tol; // [rad]; < 0 ends on duration alone
    double               gripper_cmd;
};

inline std::vector<Phase> pick_place_phases(const PickPlaceWaypoints &w)
{
    // clang-format off
    return {
        { "HOME",        &w.home,        1.0, 2.5,  0.08, 0.0            },
        { "PICK_ABOVE",  &w.pick_above,  5.0, 7.0,  0.08, 0.0            },
        { "PICK",        &w.pick,        5.0, 8.0,  0.03, 0.0            },
        { "CLOSE",       &w.pick,        1.5, 2.5, -1.0,  kGripperClosed },
        { "LIFT",        &w.lift,        3.0, 5.0,  0.08, kGripperClosed },
        { "PLACE_ABOVE", &w.place_above, 3.0, 5.0,  0.08, kGripperClosed },
        { "PLACE",       &w.place,       5.0, 8.0,  0.03, kGripperClosed },
        { "OPEN",        &w.place,       1.0, 2.0, -1.0,  0.0            },
        { "RETREAT",     &w.place_above, 2.0, 4.0,  0.08, 0.0            },
        { "HOLD",        &w.place_above, 1.0, 1.0, -1.0,  0.0            },
    };
    // clang-format on
}

// An arm the phases drive; each follows the same joint waypoints in its own base frame.
struct PhaseArm
{
    mj_kdl::Robot             *robot;
    mj_kdl::SceneActuatorSlot *gripper;
};

// One update() per step for all arms, so a command lands one cycle later; control(i, q_des)
// commands arm i. False if the viewer was closed.
inline bool run_phases(
  mj_kdl::Env                                                   &env,
  const std::vector<PhaseArm>                                   &arms,
  const std::vector<Phase>                                      &phases,
  bool                                                          &restart,
  const std::function<void(std::size_t, const KDL::JntArray &)> &control,
  const std::function<void(const Phase &, double t_rel)>        &after_step = {}
)
{
    const unsigned             n = arms.front().robot->chain.getNrOfJoints();
    std::vector<KDL::JntArray> q_enter(arms.size(), KDL::JntArray(n));
    KDL::JntArray              q_des(n);
    do {
        restart = false;
        for (const Phase &phase : phases) {
            std::cout << "State: " << phase.name << "\n";
            const double t_enter = env.data->time;
            for (std::size_t i = 0; i < arms.size(); ++i) read_q(*arms[i].robot, q_enter[i]);
            while (true) {
                mj_kdl::update(&env);
                const double t_rel = env.data->time - t_enter;
                const double alpha = phase.duration > 0.0 ? clamp01(t_rel / phase.duration) : 1.0;
                double       err   = 0.0;
                for (std::size_t i = 0; i < arms.size(); ++i) {
                    lerp_q(q_enter[i], *phase.target, alpha, q_des);
                    control(i, q_des);
                    arms[i].gripper->command = phase.gripper_cmd;
                    err = std::max(err, max_abs_joint_err(*arms[i].robot, *phase.target));
                }

                const bool done_pose    = phase.settle_tol < 0.0 || err <= phase.settle_tol;
                const bool done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
                if ((t_rel >= phase.duration && done_pose) || done_timeout) break;

                if (!mj_kdl::step(&env)) return false;
                if (restart) break;
                mj_kdl::pace_realtime(&env);
                if (after_step) after_step(phase, t_rel);
            }
            if (restart) break;
        }
    } while (restart);
    return true;
}

} // namespace mj_kdl_examples
