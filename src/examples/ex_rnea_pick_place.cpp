/* ex_rnea_pick_place.cpp
 * Two Kinova GEN3 + Robotiq 2F-85 arms face each other across the table; each picks its own cube
 * and places it 0.24 m to its left, with computed torque through KDL ChainIdSolver_RNE in TORQUE
 * mode:
 *
 *   qddot_des[i] = Kp[i] * (q_des[i] - q[i]) - Kd[i] * qdot[i]
 *   tau          = RNEA(q, qdot, qddot_des) = M(q) qddot_des + C(q, qdot) qdot + g(q)
 *
 * The second arm's names carry the prefix "r2_". Each arm has its own Robot, KDL chain and
 * solver; both see their cube at the same base-frame spot, so one IK solve serves both, and one
 * update() per step drives both. Free objects the arms must leave alone stand on the table; the
 * "overview" and "side" scene cameras join each arm's own "wrist" camera.
 *
 * Usage:
 *   ex_rnea_pick_place [--headless]
 *
 * --headless skips the viewer and exits 1 unless each cube rests on the table within kMaxPlaceErr
 * of its place spot, each elbow stayed above kMinElbowHeight, no free object moved more than
 * kMaxDisturb, the arms never touched and RNEA never failed. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainidsolver_recursive_newton_euler.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace ex = mj_kdl_examples;

static constexpr double kMaxPlaceErr    = 0.005; // [m] in the table plane
static constexpr double kMinElbowHeight = 0.45;  // [m] forearm_link origin above the table
static constexpr double kMaxDisturb     = 0.001; // [m] free-object displacement

// Acceleration gains: the closed loop is qddot = Kp e - Kd qdot [1/s^2], [1/s].
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 20, 28, 20, 28, 20, 28, 20 };

// Base x, y [m] and yaw [rad] on the tabletop: facing each other, each arm in its own half.
static constexpr double      kBase[2][3] = { { -0.70, -0.12, 0.0 }, { 0.70, 0.12, M_PI } };
static constexpr const char *kPrefix[2]  = { "", "r2_" };

static KDL::Frame world_T_base(int arm)
{
    return KDL::Frame(
      KDL::Rotation::RotZ(kBase[arm][2]), KDL::Vector(kBase[arm][0], kBase[arm][1], ex::kTableZ)
    );
}

// A cube's centre resting on the table at xy in the arm's base frame, in the world frame.
static KDL::Vector cube_spot(int arm, const double xy[2])
{
    return world_T_base(arm) * KDL::Vector(xy[0], xy[1], ex::kCubeHS);
}

struct FreeObject
{
    const char   *name;
    mj_kdl::Shape shape;
    double        x, y, half; // [m] on the table; half-size or radius
    float         rgb[3];
};

// clang-format off
static const FreeObject kFreeObjects[] = {
    { "red_box",       mj_kdl::Shape::BOX,     0.00,  0.42, 0.030, { 1.0f, 0.20f, 0.2f } },
    { "green_box",     mj_kdl::Shape::BOX,     0.00, -0.42, 0.030, { 0.2f, 1.00f, 0.2f } },
    { "yellow_box",    mj_kdl::Shape::BOX,    -0.45,  0.42, 0.040, { 1.0f, 0.85f, 0.1f } },
    { "orange_sphere", mj_kdl::Shape::SPHERE,  0.45, -0.42, 0.035, { 1.0f, 0.55f, 0.0f } },
    { "purple_sphere", mj_kdl::Shape::SPHERE,  0.00,  0.00, 0.025, { 0.7f, 0.00f, 0.9f } },
};
// clang-format on

static mj_kdl::SceneObject scene_object(const FreeObject &f)
{
    mj_kdl::SceneObject o;
    o.name  = f.name;
    o.shape = f.shape;
    for (int k = 0; k < 3; ++k) o.size[k] = f.shape == mj_kdl::Shape::BOX || k == 0 ? f.half : 0.0;
    o.pos[0] = f.x;
    o.pos[1] = f.y;
    o.pos[2] = ex::kTableZ + f.half;
    std::copy(f.rgb, f.rgb + 3, o.rgba);
    o.rgba[3]                = 1.0f;
    o.mass                   = 0.1;
    const double friction[3] = { 1.0, 0.005, 0.0001 }; // MuJoCo's geom default
    std::copy(friction, friction + 3, o.friction);
    return o;
}

static mj_kdl::CameraSpec
  camera(const char *name, const double pos[3], const double quat[4], double fovy)
{
    mj_kdl::CameraSpec cam;
    cam.name = name;
    std::copy(pos, pos + 3, cam.pos);
    std::copy(quat, quat + 4, cam.quat);
    cam.fovy = fovy;
    return cam;
}

// RNEA computed torque for one arm; the solvers keep a reference to robot.chain.
struct RneaArm
{
    RneaArm(mj_kdl::Robot &r, double gravity_z)
      : robot(r), n(r.chain.getNrOfJoints()), rnea(r.chain, KDL::Vector(0.0, 0.0, gravity_z)),
        dyn(r.chain, KDL::Vector(0.0, 0.0, gravity_z)), q(n), qdot(n), qddot(n), tau(n),
        f_ext(r.chain.getNrOfSegments(), KDL::Wrench::Zero())
    {}

    bool control(const KDL::JntArray &q_des)
    {
        ex::read_q(robot, q, qdot);
        for (unsigned i = 0; i < n; ++i) qddot(i) = kKp[i] * (q_des(i) - q(i)) - kKd[i] * qdot(i);
        if (rnea.CartToJnt(q, qdot, qddot, f_ext, tau) < 0) return false;
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = tau(i);
        return true;
    }

    mj_kdl::Robot         &robot;
    unsigned               n;
    KDL::ChainIdSolver_RNE rnea;
    KDL::ChainDynParam     dyn;
    KDL::JntArray          q, qdot, qddot, tau;
    KDL::Wrenches          f_ext;
};

int main(int argc, char *argv[])
{
    const bool headless = ex::parse_args(argc, argv).headless;

    mj_kdl::SceneSpec scene = ex::scene_spec();
    for (int a = 0; a < 2; ++a) {
        mj_kdl::RobotSpec spec;
        spec.path    = ex::menagerie_model("kinova_gen3/gen3.xml");
        spec.prefix  = kPrefix[a];
        spec.pos[0]  = kBase[a][0];
        spec.pos[1]  = kBase[a][1];
        spec.pos[2]  = ex::kTableZ;
        spec.quat[2] = std::sin(kBase[a][2] / 2.0);
        spec.quat[3] = std::cos(kBase[a][2] / 2.0);
        spec.attachments.push_back(ex::gripper_attachment(ex::asset("robotiq_2f85/2f85.xml")));
        scene.robots.push_back(spec);
    }
    scene.objects.push_back(ex::table_object(ex::asset("table.xml"), ex::kTableZ));
    for (int a = 0; a < 2; ++a) {
        const KDL::Vector   p    = cube_spot(a, ex::kPickXY);
        mj_kdl::SceneObject cube = ex::cube_object(p.x(), p.y(), ex::kTableZ);
        cube.name                = std::string(kPrefix[a]) + "cube";
        scene.objects.push_back(cube);
    }
    for (const FreeObject &f : kFreeObjects) scene.objects.push_back(scene_object(f));
    // Both look at the table centre: overview from -y above, side from +y.
    const double overview_pos[3] = { 0.0, -1.3, 1.9 };
    const double overview_q[4]   = { 0.410747, 0.0, 0.0, 0.911749 };
    const double side_pos[3]     = { 0.0, 1.5, 1.15 };
    const double side_q[4]       = { 0.0, 0.633989, 0.773342, 0.0 };
    scene.cameras.push_back(camera("overview", overview_pos, overview_q, 55.0));
    scene.cameras.push_back(camera("side", side_pos, side_q, 50.0));

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &scene)) return 1;
    std::cout << "cameras:";
    for (int i = 0; i < env.model->ncam; ++i)
        std::cout << " " << mj_id2name(env.model, mjOBJ_CAMERA, i);
    std::cout << "\n";

    const mj_kdl::ToolFrameSpec              tool = ex::gripper_tool();
    std::array<mj_kdl::Robot, 2>             robots;
    std::vector<ex::PhaseArm>                arms;
    std::vector<mj_kdl::SceneFreeBodySlot *> cubes, free_bodies;
    std::deque<RneaArm>                      ctl;
    std::array<int, 2>                       roots{};
    for (int a = 0; a < 2; ++a) {
        const std::string pre = kPrefix[a];
        if (!mj_kdl::init_robot_from_mjcf(
              &robots[a], &env, "base_link", "bracelet_link", pre.c_str(), &tool
            ))
            return 1;
        if (!mj_kdl::set_control_mode(&robots[a], mj_kdl::CtrlMode::TORQUE)) return 1;
        arms.push_back(
          { &robots[a],
            mj_kdl::bind_scene_actuator(&env.scene, (pre + "g_fingers_actuator").c_str()) }
        );
        cubes.push_back(mj_kdl::bind_scene_free_body(&env.scene, (pre + "cube").c_str()));
        if (!arms.back().gripper || !cubes.back()) return 1;
        ctl.emplace_back(robots[a], scene.gravity_z);
        const int base = mj_name2id(env.model, mjOBJ_BODY, (pre + "base_link").c_str());
        roots[a]       = env.model->body_rootid[base];
    }
    for (const FreeObject &f : kFreeObjects) {
        free_bodies.push_back(mj_kdl::bind_scene_free_body(&env.scene, f.name));
        if (!free_bodies.back()) return 1;
    }

    // Same waypoints for both arms: each target is in its own base frame.
    ex::PickPlaceWaypoints wp;
    if (!ex::solve_pick_place(robots[0], wp)) return 1;

    bool                  restart = false, solver_ok = true;
    std::array<double, 2> elbow_min{};
    double                disturb      = 0.0;
    int                   arm_contacts = 0;

    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        for (int a = 0; a < 2; ++a) {
            mj_kdl::set_joint_pos(&robots[a], wp.home);
            const KDL::Vector p      = cube_spot(a, ex::kPickXY);
            const double      pos[3] = { p.x(), p.y(), p.z() };
            mj_kdl::set_body_pose(&env, cubes[a]->name.c_str(), pos);
            ctx->data->ctrl[arms[a].gripper->ctrl_id] = 0.0;
            ex::prime_gravity(robots[a], ctl[a].dyn, wp.home);
            elbow_min[a] = INFINITY;
        }
        disturb      = 0.0;
        arm_contacts = 0;
        restart      = true;
    };
    mj_kdl::reset(&env);
    if (!headless && !mj_kdl::open_viewer(&env)) return 1;

    const auto control = [&](std::size_t a, const KDL::JntArray &q_des) {
        if (!ctl[a].control(q_des)) solver_ok = false;
    };
    const auto after_step = [&](const ex::Phase &, double) {
        for (int a = 0; a < 2; ++a) {
            KDL::Frame elbow;
            mj_kdl::get_body_frame(
              &env, (std::string(kPrefix[a]) + "forearm_link").c_str(), &elbow
            );
            elbow_min[a] = std::min(elbow_min[a], elbow.p.z());
        }
        for (std::size_t i = 0; i < free_bodies.size(); ++i) {
            const FreeObject &f = kFreeObjects[i];
            const KDL::Vector rest(f.x, f.y, ex::kTableZ + f.half);
            disturb = std::max(disturb, (free_bodies[i]->pose.p - rest).Norm());
        }
        for (int c = 0; c < env.data->ncon; ++c) {
            const mjContact &con = env.data->contact[c];
            const int        r0  = env.model->body_rootid[env.model->geom_bodyid[con.geom[0]]];
            const int        r1  = env.model->body_rootid[env.model->geom_bodyid[con.geom[1]]];
            if ((r0 == roots[0] && r1 == roots[1]) || (r0 == roots[1] && r1 == roots[0]))
                ++arm_contacts;
        }
    };
    const bool completed =
      ex::run_phases(env, arms, ex::pick_place_phases(wp), restart, control, after_step);
    mj_kdl::update(&env);
    if (!solver_ok) std::cerr << "RNEA failed during the run\n";

    bool ok = completed && solver_ok;
    std::cout << std::fixed << std::setprecision(4);
    for (int a = 0; a < 2; ++a) {
        const KDL::Vector c        = cubes[a]->pose.p;
        const KDL::Vector spot     = cube_spot(a, ex::kPlaceXY);
        const double      place_xy = std::hypot(c.x() - spot.x(), c.y() - spot.y());
        const bool        on_table = std::abs(c.z() - spot.z()) < 0.002;
        const double      elbow_h  = elbow_min[a] - ex::kTableZ;
        std::cout << "arm " << a + 1 << ": cube at [" << c.x() << ", " << c.y() << ", " << c.z()
                  << "] place error " << place_xy * 1000.0 << " mm (limit " << kMaxPlaceErr * 1000.0
                  << " mm)" << (on_table ? "" : ", not on the table")
                  << "; lowest elbow above the table " << elbow_h * 1000.0 << " mm (limit "
                  << kMinElbowHeight * 1000.0 << " mm)\n";
        ok = ok && on_table && place_xy <= kMaxPlaceErr && elbow_h >= kMinElbowHeight;
    }
    std::cout << "largest free-object displacement: " << disturb * 1000.0 << " mm (limit "
              << kMaxDisturb * 1000.0 << " mm)\narm-to-arm contacts: " << arm_contacts
              << " (limit 0)\n";
    mj_kdl::cleanup(&env);
    ok = ok && disturb <= kMaxDisturb && arm_contacts == 0;
    return headless ? ex::verdict(ok, "both arms placed their cubes and left the rest alone") : 0;
}
