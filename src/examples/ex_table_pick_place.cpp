/* ex_table_pick_place.cpp
 * Kinova GEN3 + Robotiq 2F-85 picks a cube from one table location and
 * places it at another.
 *
 * Usage:
 *   ex_table_pick_place [--headless]
 *
 * With --headless runs the full sequence and prints final cube position. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>

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
static constexpr double kPickX       = 0.40;
static constexpr double kPickY       = 0.00;
static constexpr double kPlaceX      = 0.40;
static constexpr double kPlaceY      = 0.24;
static constexpr double kTableZ      = 0.70;
static constexpr double kIkTol       = 2e-3;

static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double   clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

static void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

static void wrap_unlimited_joints_to_seed(
  const KDL::JntArray     &seed,
  KDL::JntArray           &q,
  const std::vector<bool> &limited
)
{
    const double two_pi = 2.0 * M_PI;
    for (unsigned i = 0; i < q.rows(); ++i) {
        if (limited[i]) continue;
        q(i) += std::round((seed(i) - q(i)) / two_pi) * two_pi;
    }
}

static double max_abs_joint_delta(const KDL::JntArray &a, const KDL::JntArray &b)
{
    double max_delta = 0.0;
    for (unsigned i = 0; i < a.rows(); ++i) max_delta = std::max(max_delta, std::abs(a(i) - b(i)));
    return max_delta;
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

static mj_kdl::SceneObject make_cube(double surface_z)
{
    mj_kdl::SceneObject cube;
    cube.name    = "cube";
    cube.shape   = mj_kdl::Shape::BOX;
    cube.size[0] = cube.size[1] = cube.size[2] = kCubeHS;
    cube.pos[0]                                = kPickX;
    cube.pos[1]                                = kPickY;
    cube.pos[2]                                = surface_z + kCubeHS;
    cube.rgba[0]                               = 0.1f;
    cube.rgba[1]                               = 0.35f;
    cube.rgba[2]                               = 1.0f;
    cube.rgba[3]                               = 1.0f;
    cube.mass                                  = 0.1;
    cube.condim                                = 4;
    cube.friction[0]                           = 0.8;
    cube.friction[1]                           = 0.02;
    cube.friction[2]                           = 0.001;
    return cube;
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
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = "bracelet_link";
    gripper.prefix    = "g_";
    gripper.pos[2]    = -0.061525;
    gripper.euler[0]  = 180.0;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = arm_mjcf.c_str();
    robot_spec.pos[2] = kTableZ;
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.table.enabled     = true;
    scene.table.pos[2]      = kTableZ;
    scene.table.top_size[0] = 0.8;
    scene.table.top_size[1] = 0.6;
    scene.table.thickness   = 0.04;
    scene.table.leg_radius  = 0.03;
    scene.robots.push_back(robot_spec);
    scene.objects.push_back(make_cube(kTableZ));

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
    KDL::ChainDynParam  dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
    const KDL::Rotation kGraspRot = robot.tip_T_tcp.M;

    const double z_grasp = kCubeHS;
    const double z_above = z_grasp + 0.20;
    const double z_lift  = z_grasp + 0.30;

    KDL::JntArray q_pick_above(n), q_pick(n), q_lift(n), q_place_above(n), q_place(n);
    struct Waypoint
    {
        double               world_x;
        double               world_y;
        double               world_z;
        KDL::JntArray       *out;
        const KDL::JntArray *seed;
    };
    Waypoint waypoints[] = {
        { kPickX, kPickY, kTableZ + z_above, &q_pick_above, &q_home },
        { kPickX, kPickY, kTableZ + z_grasp, &q_pick, &q_pick_above },
        { kPickX, kPickY, kTableZ + z_lift, &q_lift, &q_pick },
        { kPlaceX, kPlaceY, kTableZ + z_above, &q_place_above, &q_lift },
        { kPlaceX, kPlaceY, kTableZ + z_grasp, &q_place, &q_place_above },
    };
    KDL::Frame world_T_base(KDL::Rotation::Identity(), KDL::Vector(0.0, 0.0, kTableZ));
    KDL::Frame base_T_world = world_T_base.Inverse();
    for (const auto &wp : waypoints) {
        KDL::Frame world_target(kGraspRot, KDL::Vector(wp.world_x, wp.world_y, wp.world_z));
        KDL::Frame base_target = base_T_world * world_target;
        if (!solve_near_seed(
              ik_vel, fk, *wp.seed, base_target, joint_limited, q_min, q_max, *wp.out
            )) {
            std::cerr << "IK failed for waypoint at world [" << wp.world_x << ", " << wp.world_y
                      << ", " << wp.world_z << "]\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        KDL::Frame fk_out;
        fk.JntToCart(*wp.out, fk_out);
        double pos_err = (base_target.p - fk_out.p).Norm();
        if (pos_err > kIkTol) {
            std::cerr << "IK pose error " << pos_err << " exceeds tolerance at world ["
                      << wp.world_x << ", " << wp.world_y << ", " << wp.world_z << "]\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    }
    const std::vector<Phase> phases = {
        { "HOME",        &q_home,        1.0,              2.5,              0.08, 0.0   },
        { "PICK_ABOVE",  &q_pick_above,  5.0,              7.0,              0.08, 0.0   },
        { "PICK",        &q_pick,        5.0,              8.0,              0.03, 0.0   },
        { "CLOSE",       &q_pick,        1.5,              2.5,             -1.0,  255.0 },
        { "LIFT",        &q_lift,        3.0,              5.0,              0.08, 255.0 },
        { "PLACE_ABOVE", &q_place_above, 3.0,              5.0,              0.08, 255.0 },
        { "PLACE",       &q_place,       5.0,              8.0,              0.03, 255.0 },
        { "OPEN",        &q_place,       1.0,              2.0,             -1.0,  0.0   },
        { "RETREAT",     &q_place_above, 2.0,              4.0,              0.08, 0.0   },
        { "HOLD",        &q_place_above, headless ? 1.0 : 1e9, headless ? 1.0 : 1e9, -1.0, 0.0 },
    };

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    mj_resetData(model, data);
    mj_kdl::set_joint_pos(&robot, q_home, false);
    int qadr             = model->jnt_qposadr[cube_jnt];
    data->qpos[qadr]     = kPickX;
    data->qpos[qadr + 1] = kPickY;
    data->qpos[qadr + 2] = kTableZ + kCubeHS;
    data->qpos[qadr + 3] = 1.0;
    data->qpos[qadr + 4] = data->qpos[qadr + 5] = data->qpos[qadr + 6] = 0.0;
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i) {
        robot.jnt_pos_cmd[i]                       = data->qpos[robot.kdl_to_mj_qpos[i]];
        robot.jnt_trq_cmd[i]                       = 0.0;
        data->qfrc_applied[robot.kdl_to_mj_dof[i]] = 0.0;
    }
    mj_kdl::update(&robot);
    data->ctrl[fingers_act] = 0.0;

    mj_kdl::Viewer viewer;
    if (!headless && !mj_kdl::init_window_sim(&viewer, &robot)) {
        std::cerr << "init_window_sim() failed\n";
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    KDL::JntArray q_enter(n), q_des(n);
    bool          closed        = false;
    double        prev_sim_time = data->time;
    bool          aborted       = false;
    bool          restart       = false;

    auto reset_scene = [&]() {
        mj_resetData(model, data);
        mj_kdl::set_joint_pos(&robot, q_home, false);
        data->qpos[qadr]     = kPickX;
        data->qpos[qadr + 1] = kPickY;
        data->qpos[qadr + 2] = kTableZ + kCubeHS;
        data->qpos[qadr + 3] = 1.0;
        data->qpos[qadr + 4] = data->qpos[qadr + 5] = data->qpos[qadr + 6] = 0.0;
        mj_forward(model, data);
        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_pos_cmd[i]                       = data->qpos[robot.kdl_to_mj_qpos[i]];
            robot.jnt_trq_cmd[i]                       = 0.0;
            data->qfrc_applied[robot.kdl_to_mj_dof[i]] = 0.0;
        }
        mj_kdl::update(&robot);
        data->ctrl[fingers_act] = 0.0;
        closed                  = false;
        prev_sim_time           = data->time;
        restart                 = true;
    };

    do {
        restart = false;
        for (const Phase &phase : phases) {
            if (aborted || restart) break;
            std::cout << "State: " << phase.name << "\n";
            double t_enter = data->time;
            snapshot_q(robot, n, q_enter);

            while (true) {
                if (data->time < prev_sim_time - 1e-6) {
                    reset_scene();
                    break;
                }
                prev_sim_time = data->time;

                double alpha =
                  phase.duration > 0.0 ? clamp01((data->time - t_enter) / phase.duration) : 1.0;
                lerp_q(q_enter, *phase.target, alpha, q_des);
                impedance_ctrl(robot, q_des, n, dyn);
                mj_kdl::update(&robot);
                data->ctrl[fingers_act] = phase.gripper_cmd;
                closed                  = phase.gripper_cmd > 0.0;

                double t_rel        = data->time - t_enter;
                bool   done_time    = t_rel >= phase.duration;
                bool   done_pose    = phase.settle_tol < 0.0
                                      || max_abs_joint_err(robot, *phase.target, n) <= phase.settle_tol;
                bool   done_timeout = phase.timeout > 0.0 && t_rel >= phase.timeout;
                if ((done_time && done_pose) || done_timeout) break;

                if (!mj_kdl::step(&robot)) {
                    aborted = true;
                    break;
                }
            }
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
                  << " target=[" << kPlaceX << ", " << kPlaceY << ", " << kTableZ + kCubeHS
                  << "] xy_error=" << place_err_xy << " gripper=" << (closed ? "closed" : "open")
                  << "\n";
        if (headless && place_err_xy > 0.08) ret = 1;
    }
    if (!headless) mj_kdl::cleanup(&viewer);
    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return ret;
}
