#pragma once

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace mj_kdl_examples {

// The bundled 2F-85's ctrl is its driver joint angle; 0.82 rad is the joint's upper limit.
constexpr double kGripperClosed = 0.82;

// Kinova Gen3 home pose [rad].
constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

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

inline double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

inline void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

inline void snapshot_q(const mj_kdl::Robot &robot, KDL::JntArray &q)
{
    for (unsigned i = 0; i < q.rows(); ++i) q(i) = robot.jnt_pos_msr[i];
}

inline double max_abs_joint_err(const mj_kdl::Robot &robot, const KDL::JntArray &q)
{
    double max_err = 0.0;
    for (unsigned i = 0; i < q.rows(); ++i)
        max_err = std::max(max_err, std::abs(q(i) - robot.jnt_pos_msr[i]));
    return max_err;
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

struct Phase
{
    const char          *name;
    const KDL::JntArray *target;
    double               duration;   // [s] to interpolate from the entry pose to target
    double               timeout;    // [s] ends the phase even if not settled
    double               settle_tol; // [rad]; < 0 ends on duration alone
    double               gripper_cmd;
};

/* Runs the phases once. control writes the robot's commands for the interpolated setpoint;
 * after_step runs after each physics step. env.on_reset setting restart (the UI's reset button)
 * starts the phases over. Returns false if the viewer was closed. */
inline bool run_phases(
  mj_kdl::Env                                      &env,
  mj_kdl::Robot                                    &robot,
  mj_kdl::SceneActuatorSlot                        *gripper,
  const std::vector<Phase>                         &phases,
  bool                                             &restart,
  const std::function<void(const KDL::JntArray &)> &control,
  const std::function<void()>                      &after_step = {}
)
{
    const unsigned n = robot.chain.getNrOfJoints();
    KDL::JntArray  q_enter(n), q_des(n);
    do {
        restart = false;
        for (const Phase &phase : phases) {
            std::cout << "State: " << phase.name << "\n";
            const double t_enter = env.data->time;
            snapshot_q(robot, q_enter);
            while (true) {
                mj_kdl::update(&env); // the setpoint is computed from this cycle's state
                const double t_rel = env.data->time - t_enter;
                const double alpha = phase.duration > 0.0 ? clamp01(t_rel / phase.duration) : 1.0;
                lerp_q(q_enter, *phase.target, alpha, q_des);
                control(q_des);
                gripper->command = phase.gripper_cmd;
                mj_kdl::update(&env);

                const bool done_pose =
                  phase.settle_tol < 0.0
                  || max_abs_joint_err(robot, *phase.target) <= phase.settle_tol;
                const bool done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
                if ((t_rel >= phase.duration && done_pose) || done_timeout) break;

                if (!mj_kdl::step(&env)) return false;
                if (restart) break;
                mj_kdl::pace_realtime(&env);
                if (after_step) after_step();
            }
            if (restart) break;
        }
    } while (restart);
    return true;
}

} // namespace mj_kdl_examples
