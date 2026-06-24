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
 * With --headless runs 500 steps and prints final EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

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
    return o;
}

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const std::string mjcf     = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::asset("robotiq_2f85/2f85.xml");
    const std::string table_mjcf = mj_kdl_examples::asset("table.xml");

    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
    const double surface_z = 0.7;

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gs.prefix    = "g_";

    mj_kdl::RobotSpec r;
    r.path   = mjcf.c_str();
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
     * all of them are enumerated by get_camera_names() after build_scene(). */
    sc.cameras.push_back(mj_kdl::CameraSpec{
        .name  = "overview",
        .pos   = { 0.0, -0.6, 1.6 },  // in front of and above the table
        .euler = { 34.0, 0.0, 0.0 },  // tilt down toward table
        .fovy  = 55.0,
    });
    sc.cameras.push_back(mj_kdl::CameraSpec{
        .name  = "side",
        .pos   = { -1.0, 0.0, 1.1 },  // left side, arm height
        .euler = { 0.0, -68.0, 0.0 }, // pitch down toward robot base
        .fovy  = 50.0,
    });

    mjModel      *model = nullptr;
    mjData       *data  = nullptr;
    mj_kdl::Robot robot;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }

    KDL::Frame world_T_table_top;
    const std::string table_top_site = mj_kdl::scene_object_site_name(table, "table_top");
    if (!mj_kdl::get_site_frame(model, data, table_top_site.c_str(), &world_T_table_top)) {
        std::cerr << "table_top site not found\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    std::cout << "table top z = " << world_T_table_top.p.z() << "\n";

    std::cout << "cameras:";
    for (const auto &name : mj_kdl::get_camera_names(model))
        std::cout << " " << name;
    std::cout << "\n";

    const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };

    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", &tool
        )) {
        std::cerr << "init_robot_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    unsigned                        n = static_cast<unsigned>(robot.n_joints);
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainDynParam              dyn(robot.chain, KDL::Vector(0.0, 0.0, sc.gravity_z));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.spec  = sc;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);

    env.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_home, false);
        if (fingers_act >= 0) data->ctrl[fingers_act] = 0.8;
    };

    mj_kdl::reset(&env);

    KDL::JntArray q(n), g(n);
    auto          ctrl_step = [&]() {
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
        if (fingers_act >= 0)
            data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 0.8 : 0.0;
    };

    if (headless) {
        KDL::Frame ee_start;
        fk.JntToCart(q_home, ee_start);

        for (int step = 0; step < 500; ++step) {
            ctrl_step();
            mj_kdl::step(&robot);
        }

        KDL::JntArray q_end(n);
        for (unsigned i = 0; i < n; ++i) q_end(i) = robot.jnt_pos_msr[i];
        KDL::Frame ee_end;
        fk.JntToCart(q_end, ee_end);
        double drift = (ee_start.p - ee_end.p).Norm();
        std::cout << "EE drift after 500 steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) {
            std::cerr << "init_window_sim() failed\n";
            mj_kdl::cleanup(&robot);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }

        double prev_sim_time = data->time;
        while (true) {
            if (data->time < prev_sim_time - 1e-6) mj_kdl::reset(&env);
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::step(&robot)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
