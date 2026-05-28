/* ex_pick.cpp
 * Scripted pick-and-place: Kinova GEN3 + Robotiq 2F-85 picks an orange cube
 * from the floor and lifts it.
 *
 * Joint impedance control law applied every step:
 *   tau_i = g_kdl_i + Kp*(q_des_i - q_i) - Kd*dq_i
 * where g_kdl is computed via KDL::ChainDynParam::JntToGravity (includes gripper inertia).
 *
 * A state machine drives the arm through six states, each with its own
 * while control loop:
 *   HOME -> PREGRASP -> GRASP -> CLOSE -> LIFT -> HOLD
 *
 * Requires third_party/menagerie (MuJoCo Menagerie submodule).
 *
 * Usage:
 *   ex_pick [--headless]
 *
 * With --headless runs the full pick sequence and prints final cube height. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// scene constants
static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kCubeX       = 0.4;
static constexpr double kCubeY       = 0.0;
static constexpr double kCubeHS      = 0.02; // half-size of 4 cm cube
static constexpr double kCubeZ       = kCubeHS;
static constexpr double kIkTol       = 2e-3;

// impedance gains (per joint, matching ex_impedance tuning)
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 }; // Nm/rad
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };        // Nm*s/rad

// helpers
namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double   clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

static void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

static bool solve_near_seed(
  KDL::ChainIkSolverVel_wdls      &ik_vel,
  KDL::ChainFkSolverPos_recursive &fk,
  const KDL::JntArray             &seed,
  const KDL::Frame                &target,
  const std::vector<bool>         &joint_limited,
  const KDL::JntArray             &q_min,
  const KDL::JntArray             &q_max,
  KDL::JntArray                   &out
)
{
    out = seed;
    KDL::JntArray dq(out.rows());
    for (int iter = 0; iter < 300; ++iter) {
        KDL::Frame fk_out;
        fk.JntToCart(out, fk_out);
        KDL::Twist dx = KDL::diff(fk_out, target);
        if (dx.vel.Norm() <= kIkTol && dx.rot.Norm() <= 2e-2) return true;

        double vel_norm = dx.vel.Norm();
        if (vel_norm > 0.05) dx.vel = dx.vel * (0.05 / vel_norm);
        double rot_norm = dx.rot.Norm();
        if (rot_norm > 0.20) dx.rot = dx.rot * (0.20 / rot_norm);

        if (ik_vel.CartToJnt(out, dx, dq) < 0) return false;
        for (unsigned i = 0; i < out.rows(); ++i) {
            out(i) += dq(i);
            if (joint_limited[i]) out(i) = std::max(q_min(i), std::min(q_max(i), out(i)));
        }
    }

    KDL::Frame fk_out;
    fk.JntToCart(out, fk_out);
    KDL::Twist dx = KDL::diff(fk_out, target);
    return dx.vel.Norm() <= kIkTol && dx.rot.Norm() <= 2e-2;
}

/*
 * Compute joint impedance torques into robot.jnt_trq_cmd:
 *   tau_i = g_kdl[i] + Kp[i]*(q_des_i - q_msr_i) - Kd[i]*dq_msr_i
 * dyn must have been built from the chain returned by init_robot_from_mjcf with tool_body set.
 */
static void impedance_ctrl(
  mj_kdl::Robot       &robot,
  const KDL::JntArray &q_des,
  unsigned             n,
  KDL::ChainDynParam  &dyn
)
{
    KDL::JntArray q(n), g(n);
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) {
        robot.jnt_trq_cmd[i] =
          g(i) + kKp[i] * (q_des(i) - robot.jnt_pos_msr[i]) - kKd[i] * robot.jnt_vel_msr[i];
    }
}

// Copy the current measured joint positions into q.
static void snapshot_q(const mj_kdl::Robot &robot, unsigned n, KDL::JntArray &q)
{
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
}

static double max_abs_joint_err(const mj_kdl::Robot &robot, const KDL::JntArray &q_des, unsigned n)
{
    double max_err = 0.0;
    for (unsigned i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(q_des(i) - robot.jnt_pos_msr[i]));
    return max_err;
}

// state machine
enum class PickState { HOME, PREGRASP, GRASP, CLOSE, LIFT, HOLD, DONE };

/*
 * Static description of one state: how long to stay, which joint target to
 * track (pointer into the IK-solved arrays), gripper command, and successor.
 */
struct StateConfig
{
    double               duration;    // nominal interpolation time
    double               timeout;     // force transition if settling runs long
    double               settle_tol;  // rad, < 0 disables pose check
    const KDL::JntArray *q_target;    // interpolation goal (set after IK)
    double               gripper_cmd; // 0=open, 255=fully closed
    PickState            next;
};

/*
 * Mutable context updated on every state transition.
 * q_enter is the measured joint position when the state was entered;
 * the setpoint is linearly interpolated from q_enter to q_target over duration.
 */
struct StateMachine
{
    PickState     state   = PickState::HOME;
    double        t_enter = 0.0;
    KDL::JntArray q_enter;
};

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found  - run: "
                     "git submodule update --init third_party/menagerie\n";
        return 1;
    }

    // scene setup
    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gs.prefix    = "g_";

    mj_kdl::RobotSpec rs;
    rs.path = arm_mjcf.c_str();
    rs.attachments.push_back(gs);

    mj_kdl::SceneObject cube{
        .name      = "cube",
        .mjcf_path = "",
        .shape     = mj_kdl::Shape::BOX,
        .size      = { kCubeHS, kCubeHS, kCubeHS },
        .pos       = { kCubeX, kCubeY, kCubeZ },
        .rgba      = { 1.0f, 0.5f, 0.0f, 1.0f },
        .mass      = 0.1,
        .condim    = mj_kdl::Condim::Torsional,
        .friction  = { 0.8, 0.02, 0.001 },
    };

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    sc.robots.push_back(rs);
    sc.objects.push_back(cube);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
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

    unsigned n           = robot.chain.getNrOfJoints();
    int      fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int      cube_jnt    = mj_name2id(model, mjOBJ_JOINT, "cube_joint");
    int      key_id      = mj_name2id(model, mjOBJ_KEY, "home");

    KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, -9.81));

    // IK setup
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::JntArray                   q_min(n), q_max(n);
    std::vector<bool>               joint_limited(n, false);
    for (unsigned i = 0; i < n; ++i) {
        int jid = model->dof_jntid[robot.kdl_to_mj_dof[i]];
        if (model->jnt_limited[jid]) {
            joint_limited[i] = true;
            q_min(i)         = model->jnt_range[2 * jid];
            q_max(i)         = model->jnt_range[2 * jid + 1];
        } else {
            q_min(i) = -2 * M_PI;
            q_max(i) = 2 * M_PI;
        }
    }
    KDL::ChainIkSolverVel_wdls ik_vel(robot.chain, 1e-5, 150);
    ik_vel.setLambda(0.05);

    KDL::JntArray q_home(n), q_pregrasp(n), q_grasp(n), q_lift(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    const double        kGraspZ    = kCubeZ;
    const double        kPreGraspZ = kGraspZ + 0.20;
    const double        kLiftZ     = kGraspZ + 0.30;
    const KDL::Rotation kGraspRot  = robot.tip_T_tcp.M;

    struct WP
    {
        double               z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    WP wps[] = {
        { kPreGraspZ, &q_pregrasp, &q_home },
        { kGraspZ, &q_grasp, &q_pregrasp },
        { kLiftZ, &q_lift, &q_grasp },
    };
    for (auto &wp : wps) {
        KDL::Frame target(kGraspRot, KDL::Vector(kCubeX, kCubeY, wp.z));
        if (!solve_near_seed(ik_vel, fk, *wp.seed, target, joint_limited, q_min, q_max, *wp.out)) {
            std::cerr << "IK failed for z=" << wp.z << "\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        KDL::Frame fk_out;
        fk.JntToCart(*wp.out, fk_out);
        double pos_err = (target.p - fk_out.p).Norm();
        if (pos_err > kIkTol) {
            std::cerr << "IK pose error " << pos_err << " exceeds tolerance for z=" << wp.z << "\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    }
    // state configuration table
    /*
     * HOLD duration is 1 s in headless mode (enough to verify the lift),
     * and 10 s in GUI mode so the user has time to inspect the scene before
     * the window auto-closes.
     */
    const double kHoldDuration = headless ? 1.0 : 10.0;

    // clang-format off
    const StateConfig state_cfg[] = {
        { .duration = 1.0,           .timeout = 2.5,           .settle_tol =  0.08, .q_target = &q_home,     .gripper_cmd =   0.0, .next = PickState::PREGRASP },
        { .duration = 5.0,           .timeout = 7.0,           .settle_tol =  0.08, .q_target = &q_pregrasp, .gripper_cmd =   0.0, .next = PickState::GRASP    },
        { .duration = 5.0,           .timeout = 8.0,           .settle_tol =  0.03, .q_target = &q_grasp,    .gripper_cmd =   0.0, .next = PickState::CLOSE    },
        { .duration = 1.5,           .timeout = 2.5,           .settle_tol = -1.0,  .q_target = &q_grasp,    .gripper_cmd = 255.0, .next = PickState::LIFT     },
        { .duration = 3.0,           .timeout = 5.0,           .settle_tol =  0.08, .q_target = &q_lift,     .gripper_cmd = 255.0, .next = PickState::HOLD     },
        { .duration = kHoldDuration, .timeout = kHoldDuration, .settle_tol = -1.0,  .q_target = &q_lift,     .gripper_cmd = 255.0, .next = PickState::DONE     },
        { .duration = 0.0,           .timeout = 0.0,           .settle_tol = -1.0,  .q_target = &q_lift,     .gripper_cmd = 255.0, .next = PickState::DONE     },
    };
    // clang-format on

    auto cfg = [&](PickState s) -> const StateConfig & { return state_cfg[static_cast<int>(s)]; };

    // initial conditions
    auto reset_cube = [&](mjData *d) {
        int qadr          = model->jnt_qposadr[cube_jnt];
        d->qpos[qadr + 0] = kCubeX;
        d->qpos[qadr + 1] = kCubeY;
        d->qpos[qadr + 2] = kCubeZ;
        d->qpos[qadr + 3] = 1.0;
        d->qpos[qadr + 4] = d->qpos[qadr + 5] = d->qpos[qadr + 6] = 0.0;
    };

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.spec  = sc;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    mj_kdl::ResetOptions reset_opts;
    reset_opts.use_keyframe = key_id >= 0;
    reset_opts.keyframe     = key_id >= 0 ? key_id : 0;

    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        if (key_id < 0) mj_kdl::set_joint_pos(&robot, q_home, false);
        reset_cube(ctx->data);
        if (fingers_act >= 0) ctx->data->ctrl[fingers_act] = 0.0;
    };

    mj_kdl::reset(&env, &reset_opts);

    // GUI init (non-headless only)
    mj_kdl::Viewer viewer;
    if (!headless) {
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    }

    StateMachine sm;
    sm.q_enter = KDL::JntArray(n);

    KDL::JntArray q_des(n);

    auto begin_state = [&](PickState next) {
        sm.state   = next;
        sm.t_enter = data->time;
        snapshot_q(robot, n, sm.q_enter);
    };

    begin_state(PickState::HOME);

    double prev_sim_time = data->time;

    while (sm.state != PickState::DONE) {
        switch (sm.state) {
        case PickState::HOME:
            std::cout << "State: HOME\n";
            while (sm.state == PickState::HOME) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    continue;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // HOME uses the default joint-impedance controller.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::PREGRASP:
            std::cout << "State: PREGRASP\n";
            while (sm.state == PickState::PREGRASP) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    break;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // PREGRASP could swap in a different approach controller later.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::GRASP:
            std::cout << "State: GRASP\n";
            while (sm.state == PickState::GRASP) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    break;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // GRASP keeps the same structure but can use contact-aware logic later.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::CLOSE:
            std::cout << "State: CLOSE\n";
            while (sm.state == PickState::CLOSE) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    break;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // CLOSE could switch to a grasp-force controller instead of pure impedance.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::LIFT:
            std::cout << "State: LIFT\n";
            while (sm.state == PickState::LIFT) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    break;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // LIFT could use a stiffer transport controller if needed.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::HOLD:
            std::cout << "State: HOLD\n";
            while (sm.state == PickState::HOLD) {
                if (data->time < prev_sim_time - 1e-6) {
                    mj_kdl::reset(&env, &reset_opts);
                    begin_state(PickState::HOME);
                    prev_sim_time = data->time;
                    break;
                }
                prev_sim_time = data->time;

                const StateConfig &state = cfg(sm.state);
                double             alpha = (state.duration > 0.0)
                                             ? clamp01((data->time - sm.t_enter) / state.duration)
                                             : 1.0;
                lerp_q(sm.q_enter, *state.q_target, alpha, q_des);

                // HOLD can later become a dedicated grasp-maintenance controller.
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                if (fingers_act >= 0) data->ctrl[fingers_act] = state.gripper_cmd;

                double t_rel        = data->time - sm.t_enter;
                double pose_err     = max_abs_joint_err(robot, *state.q_target, n);
                bool   done_time    = t_rel >= state.duration;
                bool   done_pose    = (state.settle_tol < 0.0) || (pose_err <= state.settle_tol);
                bool   done_timeout = (state.timeout > 0.0) && (t_rel >= state.timeout);
                if ((done_time && done_pose) || done_timeout) {
                    begin_state(state.next);
                    break;
                }

                if (!mj_kdl::step(&robot)) {
                    sm.state = PickState::DONE;
                    break;
                }
            }
            break;

        case PickState::DONE:
            break;
        }
    }

    if (headless) {
        int    qadr   = model->jnt_qposadr[cube_jnt];
        double cube_z = data->qpos[qadr + 2];
        std::cout << "cube Z after pick: " << std::fixed << std::setprecision(3) << cube_z
                  << " m\n";
    } else {
        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
