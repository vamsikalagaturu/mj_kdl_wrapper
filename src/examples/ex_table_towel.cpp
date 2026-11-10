/* ex_table_towel.cpp
 * Kinova GEN3 + Robotiq 2F-85 drags a towel on the table: a 0.30 m cloth (MuJoCo 2D flex,
 * assets/towel.xml) hangs from the gripper by the middle of one edge. From the home pose the
 * gripper closes, goes down until the towel's lower part lies on the table, drags it left and
 * right, and goes up, the gripper pointing down throughout. RNEA computed torque (KDL
 * ChainIdSolver_RNE) in TORQUE mode tracks joint waypoints, as in ex_rnea_pick_place; the
 * waypoints are the home TCP pose moved down, sideways and up, solved by IK once at start.
 * The scene runs MuJoCo's discrete integrator with Newton at tolerance 1e-6, as MuJoCo's modeling
 * guide advises for cloth.
 *
 * Timestep 2 ms, not the 1 ms of the other examples: every step solves the cloth implicitly, and
 * the compliant torque-controlled arm and the lightly damped towel keep each other swinging, so
 * the solver iterates each step. At 1 ms that runs at about 0.8x real time on a laptop CPU, at
 * 2 ms about 1.8x; MuJoCo's modeling guide gives 1-4 ms as typical for cloth under discrete.
 *
 * Usage:
 *   ex_table_towel [--headless]
 *
 * --headless skips the viewer and exits 1 unless the towel lay on the table while dragged, came
 * clear of it at the end, RNEA never failed and the cloth stayed finite. */

#include "mj_kdl_wrapper/viewer.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace ex = mj_kdl_examples;

static constexpr double kTowelLength = 0.30;  // [m] the towel hangs this far below the TCP
static constexpr double kOnTable     = 0.06;  // [m] of towel lying on the table while dragged
static constexpr double kDragY       = 0.25;  // [m] left and right of the start
static constexpr double kTimestep    = 0.002; // [s] see the file comment

static constexpr double kTouchHeight  = 0.010; // [m] above the table: a vertex on it
static constexpr int    kMinOnTable   = 9;     // towel vertices on the table while dragged
static constexpr double kMinClearance = 0.02;  // [m] lowest vertex above the table at the end

int main(int argc, char *argv[])
{
    const bool headless = ex::parse_args(argc, argv).headless;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = ex::asset("kinova_gen3/gen3.xml");
    robot_spec.pos[2] = ex::kTableZ;
    robot_spec.attachments.push_back(ex::gripper_attachment(ex::asset("robotiq_2f85/2f85.xml")));
    mj_kdl::AttachmentSpec towel;
    towel.mjcf_path = ex::asset("towel.xml");
    towel.attach_to = { mj_kdl::AttachKind::Body, "g_base" };
    towel.prefix    = "towel_";
    robot_spec.attachments.push_back(towel);

    mj_kdl::SceneSpec scene = ex::scene_spec();
    scene.timestep          = kTimestep;
    scene.integrator        = mj_kdl::Integrator::Discrete;
    scene.solver            = mj_kdl::Solver::Newton;
    scene.robots.push_back(robot_spec);
    scene.objects.push_back(ex::table_object(ex::asset("table.xml"), ex::kTableZ));

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &scene)) return 1;
    env.model->opt.tolerance         = 1e-6;
    const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool))
        return 1;
    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;
    mj_kdl::SceneActuatorSlot *fingers =
      mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    if (!fingers || env.model->nflex != 1) return 1;

    // Waypoints: the home TCP pose moved down, left, right and up, the orientation kept.
    const unsigned                  n = robot.chain.getNrOfJoints();
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainIkSolverVel_wdls      ik_vel(robot.chain, 1e-5, 150);
    ik_vel.setLambda(0.05);
    const KDL::JntArray home = ex::home_q(n);
    KDL::Frame          x_home;
    fk.JntToCart(home, x_home);
    const double      down     = (kTowelLength - kOnTable) - x_home.p.z();
    const KDL::Vector offset[] = {
        { 0.0, 0.0, down }, { 0.0, kDragY, down }, { 0.0, -kDragY, down }, { 0.0, -kDragY, 0.0 }
    };
    std::vector<KDL::JntArray> q(std::size(offset), KDL::JntArray(n));
    for (std::size_t i = 0; i < q.size(); ++i) {
        const KDL::Frame target(x_home.M, x_home.p + offset[i]);
        if (!ex::solve_near_seed(ik_vel, fk, robot, i == 0 ? home : q[i - 1], target, q[i])) {
            std::cerr << "IK failed for waypoint " << i << "\n";
            return 1;
        }
    }

    ex::RneaArm ctl(robot, scene.gravity_z);
    bool        restart = false;
    env.on_reset        = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, home);
        ctx->data->ctrl[fingers->ctrl_id] = 0.0;
        ex::prime_gravity(robot, ctl.dyn, home);
        restart = true;
    };
    mj_kdl::reset(&env);
    mj_kdl::Viewer viewer;
    viewer.paused = true;
    if (!headless && !mj_kdl::open_viewer(&viewer, &env)) return 1;

    const double closed = ex::kGripperClosed;
    // clang-format off
    const std::vector<ex::Phase> phases = {
        { "CLOSE",      &home, 1.0, 1.0, -1.0,  closed },
        { "DOWN",       &q[0], 2.0, 3.0,  0.03, closed },
        { "DRAG_LEFT",  &q[1], 2.0, 3.0,  0.03, closed },
        { "DRAG_RIGHT", &q[2], 3.0, 4.0,  0.03, closed },
        { "UP",         &q[3], 2.0, 3.0,  0.03, closed },
        { "HOLD",       &q[3], 1.0, 1.0, -1.0,  closed },
    };
    // clang-format on

    const int  first = env.model->flex_vertadr[0], count = env.model->flex_vertnum[0];
    int        on_table_max = 0;
    bool       solver_ok    = true;
    const auto control      = [&](std::size_t, const KDL::JntArray &q_des) {
        if (!ctl.control(q_des)) solver_ok = false;
    };
    const auto after_step = [&](const ex::Phase &phase, double) {
        const std::string name = phase.name;
        if (name != "DRAG_LEFT" && name != "DRAG_RIGHT") return;
        int on_table = 0;
        for (int v = first; v < first + count; ++v)
            if (env.data->flexvert_xpos[3 * v + 2] - ex::kTableZ < kTouchHeight) ++on_table;
        on_table_max = std::max(on_table_max, on_table);
    };
    const bool completed = ex::run_phases(
      env,
      [&viewer] { mj_kdl::pace_realtime(&viewer); },
      { { &robot, fingers } },
      phases,
      restart,
      control,
      after_step
    );
    mj_kdl::update(&env);

    bool   finite = true;
    double lowest = INFINITY;
    for (int v = first; v < first + count; ++v) {
        const mjtNum *p = env.data->flexvert_xpos + 3 * v;
        finite = finite && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
        lowest = std::min(lowest, p[2] - ex::kTableZ);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "towel vertices on the table while dragged: " << on_table_max << "/" << count
              << " (at least " << kMinOnTable << ")\n"
              << "lowest towel vertex at the end: " << lowest << " m above the table (at least "
              << kMinClearance << " m)" << (finite ? "" : "; the cloth went non-finite")
              << (solver_ok ? "" : "; RNEA failed") << "\n";
    mj_kdl::cleanup(&env);
    const bool ok =
      completed && solver_ok && finite && on_table_max >= kMinOnTable && lowest >= kMinClearance;
    return headless ? ex::verdict(ok, "the towel was dragged on the table and lifted clear") : 0;
}
