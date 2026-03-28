/* ex_table_scene.cpp  (MJCF)
 * Kinova GEN3 + Robotiq 2F-85 gripper on a table with a few free objects,
 * loaded from MuJoCo Menagerie MJCF.
 *
 * The arm runs KDL gravity compensation; the gripper cycles open/closed every 3 s.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_table_scene_mjcf [--headless]
 *
 * With --headless runs 500 steps and prints final EE drift. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

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

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found - run:\n"
                     "  cmake -B build -DFETCH_MENAGERIE=ON\n";
        return 1;
    }

    const std::string mjcf     = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::SceneSpec sc;
    sc.table.enabled     = true;
    sc.table.pos[2]      = 0.7;
    sc.table.top_size[0] = 0.8;
    sc.table.top_size[1] = 0.6;
    sc.table.thickness   = 0.04;
    sc.table.leg_radius  = 0.03;

    const double surface_z = sc.table.pos[2];

    mj_kdl::AttachmentSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[2]    = -0.061525;
    gs.euler[0]  = 180.0;

    mj_kdl::RobotSpec r;
    r.path   = mjcf.c_str();
    r.pos[2] = surface_z;
    r.attachments.push_back(gs);
    sc.robots.push_back(r);

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

    mjModel      *model = nullptr;
    mjData       *data  = nullptr;
    mj_kdl::Robot robot;
    if (!mj_kdl::build_scene(&model, &data, &sc)) {
        std::cerr << "build_scene() failed\n";
        return 1;
    }
    if (!mj_kdl::init_robot_from_mjcf(
          &robot, model, data, "base_link", "bracelet_link", "", "g_base"
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

    auto reset_to_home = [&]() {
        mj_resetData(model, data);
        mj_kdl::set_joint_pos(&robot, q_home, false);
        mj_forward(model, data);
        for (unsigned i = 0; i < n; ++i) {
            robot.jnt_pos_cmd[i]                       = data->qpos[robot.kdl_to_mj_qpos[i]];
            robot.jnt_trq_cmd[i]                       = 0.0;
            data->qfrc_applied[robot.kdl_to_mj_dof[i]] = 0.0;
        }
        mj_kdl::update(&robot);
        if (fingers_act >= 0) data->ctrl[fingers_act] = 255.0;
    };

    reset_to_home();

    KDL::JntArray q(n), g(n);
    auto          ctrl_step = [&]() {
        mj_kdl::update(&robot);
        for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
        if (fingers_act >= 0)
            data->ctrl[fingers_act] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
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
            if (data->time < prev_sim_time - 1e-6) reset_to_home();
            prev_sim_time = data->time;
            ctrl_step();
            if (!mj_kdl::tick(&viewer, model, data)) break;
        }

        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
