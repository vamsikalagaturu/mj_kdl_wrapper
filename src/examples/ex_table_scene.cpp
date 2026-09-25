/* ex_table_scene.cpp  (MJCF)
 * Kinova GEN3 + Robotiq 2F-85 gripper on a table with a few free objects,
 * loaded from MuJoCo Menagerie MJCF.
 *
 * The arm runs KDL gravity compensation; the gripper cycles open/closed every 3 s.
 *
 * Requires MuJoCo Menagerie in cache.
 *
 * Usage:
 *   ex_table_scene_mjcf [--headless]
 *
 * Runs 500 steps and exits; --headless skips the viewer and prints the final EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

using mj_kdl_examples::kHomePose;
static constexpr double kObjectMass  = 0.1;                    // [kg]
static constexpr double kFriction[3] = { 1.0, 0.005, 0.0001 }; // MuJoCo's geom default
static constexpr int    kSteps       = 500;

static mj_kdl::SceneObject make_box(
  const char *name,
  double      x,
  double      y,
  double      hx,
  double      hy,
  double      hz,
  float       r,
  float       g,
  float       b,
  double      surface_z
)
{
    mj_kdl::SceneObject o;
    o.name    = name;
    o.shape   = mj_kdl::Shape::BOX;
    o.size[0] = hx;
    o.size[1] = hy;
    o.size[2] = hz;
    o.pos[0]  = x;
    o.pos[1]  = y;
    o.pos[2]  = surface_z + hz;
    o.rgba[0] = r;
    o.rgba[1] = g;
    o.rgba[2] = b;
    o.rgba[3] = 1.0f;
    o.mass    = kObjectMass;
    for (int k = 0; k < 3; ++k) o.friction[k] = kFriction[k];
    return o;
}

static mj_kdl::SceneObject make_sphere(
  const char *name,
  double      x,
  double      y,
  double      radius,
  float       r,
  float       g,
  float       b,
  double      surface_z
)
{
    mj_kdl::SceneObject o;
    o.name    = name;
    o.shape   = mj_kdl::Shape::SPHERE;
    o.size[0] = radius;
    o.pos[0]  = x;
    o.pos[1]  = y;
    o.pos[2]  = surface_z + radius;
    o.rgba[0] = r;
    o.rgba[1] = g;
    o.rgba[2] = b;
    o.rgba[3] = 1.0f;
    o.mass    = kObjectMass;
    for (int k = 0; k < 3; ++k) o.friction[k] = kFriction[k];
    return o;
}

int main(int argc, char *argv[])
{
    const bool headless = mj_kdl_examples::parse_args(argc, argv).headless;

    const std::string mjcf     = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string table_mjcf = mj_kdl_examples::asset("table.xml");

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    const double surface_z = 0.7;

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf;
    gs.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gs.prefix    = "g_";

    mj_kdl::RobotSpec r;
    r.path   = mjcf;
    r.pos[2] = surface_z;
    r.attachments.push_back(gs);
    sc.robots.push_back(r);

    mj_kdl::SceneObject table{
        .name      = "table",
        .mjcf_path = table_mjcf,
        .pos       = { 0.0, 0.0, surface_z },
        .fixed     = true,
    };
    sc.objects.push_back(table);
    sc.objects.push_back(
      make_box("red_cube", 0.35, 0.10, 0.03, 0.03, 0.03, 1.0f, 0.2f, 0.2f, surface_z)
    );
    sc.objects.push_back(
      make_box("green_cube", 0.35, -0.10, 0.03, 0.03, 0.03, 0.2f, 1.0f, 0.2f, surface_z)
    );
    sc.objects.push_back(
      make_box("blue_cube", 0.35, 0.30, 0.04, 0.04, 0.04, 0.2f, 0.2f, 1.0f, surface_z)
    );
    sc.objects.push_back(
      make_sphere("orange_sphere", -0.20, 0.20, 0.035, 1.0f, 0.55f, 0.0f, surface_z)
    );
    sc.objects.push_back(
      make_sphere("purple_sphere", -0.20, -0.20, 0.025, 0.7f, 0.0f, 0.9f, surface_z)
    );

    /* Static scene cameras.  The Kinova MJCF also contributes a "wrist" camera;
     * all of them are in env.model after init_env(). */
    sc.cameras.push_back(mj_kdl::CameraSpec{
        .name = "overview",
        .pos  = { 0.0, -0.6, 1.6 }, // in front of and above the table
        // roll 34 deg: tilt down toward table
        .quat = { 0.29237170472273677, 0.0, 0.0, 0.9563047559630354 },
        .fovy = 55.0,
    });
    sc.cameras.push_back(mj_kdl::CameraSpec{
        .name = "side",
        .pos  = { -1.0, 0.0, 1.1 }, // left side, arm height
        // pitch -68 deg: pitch down toward robot base
        .quat = { 0.0, -0.5591929034707469, 0.0, 0.8290375725550416 },
        .fovy = 50.0,
    });

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_env(&env, &sc)) {
        std::cerr << "init_env() failed\n";
        return 1;
    }

    KDL::Frame world_T_table_top;
    if (!mj_kdl::get_site_frame(&env, "table_top", &world_T_table_top)) {
        std::cerr << "table_top site not found\n";
        return 1;
    }
    std::cout << "table top z = " << world_T_table_top.p.z() << "\n";

    std::cout << "cameras:";
    for (int i = 0; i < env.model->ncam; ++i)
        std::cout << " " << mj_id2name(env.model, mjOBJ_CAMERA, i);
    std::cout << "\n";

    const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base_mount", .tcp_site = "g_pinch" };

    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool)) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        return 1;
    }

    unsigned                        n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, sc.gravity_z));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    mj_kdl::SceneActuatorSlot *fingers = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
    if (!fingers) return 1;

    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    // reset() seeds the finger slot from ctrl, so the hook sets ctrl.
    env.on_reset = [&](mj_kdl::ResetContext *ctx) {
        mj_kdl::set_joint_pos(&robot, q_home);
        ctx->data->ctrl[fingers->ctrl_id] = mj_kdl_examples::kGripperClosed;
    };

    mj_kdl::reset(&env);

    KDL::JntArray q(n), g(n);
    auto          ctrl_step = [&]() {
        mj_kdl::update(&env);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
        fingers->command =
          (std::fmod(env.data->time, 6.0) < 3.0) ? mj_kdl_examples::kGripperClosed : 0.0;
    };

    if (headless) {
        KDL::Frame ee_start;
        fk.JntToCart(q_home, ee_start);

        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            mj_kdl::step(&env);
        }

        KDL::JntArray q_end(n);
        for (unsigned i = 0; i < n; ++i) q_end(i) = robot.jnt_pos_msr[i];
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after " << kSteps << " steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
    } else {
        if (!mj_kdl::open_viewer(&env)) {
            std::cerr << "open_viewer() failed\n";
            return 1;
        }
        for (int step = 0; step < kSteps; ++step) {
            ctrl_step();
            if (!mj_kdl::step(&env)) break;
            mj_kdl::pace_realtime(&env);
        }
    }

    mj_kdl::cleanup(&env);
    return 0;
}
