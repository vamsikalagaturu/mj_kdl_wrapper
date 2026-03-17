// test_table_scene.cpp
// Load a robot arm on a table with pickable objects (cubes and spheres).
// Runs KDL gravity compensation so the arm holds position while the objects
// can be perturbed in GUI mode.
//
// Also demonstrates runtime scene_add_object / scene_remove_object.
//
// Usage: test_table_scene [urdf_path] [--gui]

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = {0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708};

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

int main(int argc, char* argv[])
{
    std::string urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool gui = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui") gui = true;
        else if (a[0] != '-') urdf = a;
    }

    // Table setup: surface at z = 0.7 m, robot mounted at table centre.
    mj_kdl::TableSpec table;
    table.enabled     = true;
    table.pos[0]      = 0.0;
    table.pos[1]      = 0.0;
    table.pos[2]      = 0.7;   // surface height
    table.top_size[0] = 0.8;   // half-extent x
    table.top_size[1] = 0.6;   // half-extent y
    table.thickness   = 0.04;
    table.leg_radius  = 0.03;

    double surface_z = table.pos[2];

    // Objects: 3 coloured boxes and 2 spheres resting on the table.
    std::vector<mj_kdl::SceneObject> objects;

    auto make_box = [&](const char* name,
                        double x, double y, double hx, double hy, double hz,
                        float r, float g, float b) {
        mj_kdl::SceneObject o;
        o.name     = name;
        o.shape    = mj_kdl::ObjShape::BOX;
        o.size[0]  = hx; o.size[1] = hy; o.size[2] = hz;
        o.pos[0]   = x;  o.pos[1]  = y;  o.pos[2]  = surface_z + hz;
        o.rgba[0]  = r;  o.rgba[1] = g;  o.rgba[2]  = b; o.rgba[3] = 1.0f;
        return o;
    };

    auto make_sphere = [&](const char* name,
                           double x, double y, double radius,
                           float r, float g, float b) {
        mj_kdl::SceneObject o;
        o.name     = name;
        o.shape    = mj_kdl::ObjShape::SPHERE;
        o.size[0]  = radius;
        o.pos[0]   = x;  o.pos[1] = y;  o.pos[2] = surface_z + radius;
        o.rgba[0]  = r;  o.rgba[1] = g;  o.rgba[2] = b; o.rgba[3] = 1.0f;
        return o;
    };

    objects.push_back(make_box("red_cube",   0.35,  0.10, 0.03, 0.03, 0.03,  1.0f, 0.2f, 0.2f));
    objects.push_back(make_box("green_cube", 0.35, -0.10, 0.03, 0.03, 0.03,  0.2f, 1.0f, 0.2f));
    objects.push_back(make_box("blue_cube",  0.35,  0.30, 0.04, 0.04, 0.04,  0.2f, 0.2f, 1.0f));
    objects.push_back(make_sphere("orange_sphere", -0.20,  0.20, 0.035, 1.0f, 0.55f, 0.0f));
    objects.push_back(make_sphere("purple_sphere", -0.20, -0.20, 0.025, 0.7f, 0.0f,  0.9f));

    // Build scene: robot at table surface (pos[2] = surface_z).
    mj_kdl::SceneSpec spec;
    spec.table   = table;
    spec.objects = objects;
    spec.add_floor   = true;
    spec.gravity_z   = -9.81;

    mj_kdl::SceneRobot robot;
    robot.urdf_path = urdf.c_str();
    robot.prefix    = "";
    robot.pos[0]    = 0.0;
    robot.pos[1]    = 0.0;
    robot.pos[2]    = surface_z;  // mount base on table surface
    spec.robots.push_back(robot);

    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &spec)) {
        std::cerr << "FAIL: build_scene\n";
        return 1;
    }
    std::cout << "Scene built — " << model->nbody << " bodies, "
              << model->nq << " DOFs\n";

    // Attach KDL state for the robot.
    mj_kdl::State s;
    if (!mj_kdl::init_robot(&s, model, data,
                             urdf.c_str(), "base_link", "EndEffector_Link")) {
        std::cerr << "FAIL: init_robot\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    std::cout << "Robot: " << s.n_joints << " joints\n";

    unsigned n = static_cast<unsigned>(s.n_joints);

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainDynParam             dyn(s.chain, KDL::Vector(0, 0, spec.gravity_z));

    // Set home pose.
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(model, data);

    KDL::Frame ee_init;
    fk.JntToCart(q_home, ee_init);

    // Headless: 500 steps then check EE drift.
    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q, g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
        mj_kdl::step(&s);
    }

    KDL::JntArray q_end;
    KDL::Frame    ee_end;
    mj_kdl::sync_to_kdl(&s, q_end);
    fk.JntToCart(q_end, ee_end);
    double drift = (ee_init.p - ee_end.p).Norm();

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "EE drift after 500 steps: " << drift * 1000.0 << " mm\n";
    if (drift > 0.001) {
        std::cerr << "FAIL: drift > 1 mm\n";
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }
    std::cout << "OK\n";

    // Runtime add / remove demo (headless only).
    if (!gui) {
        std::cout << "\n--- Runtime object add/remove ---\n";
        std::cout << "Bodies before add: " << model->nbody << "\n";

        mj_kdl::cleanup(&s);

        mj_kdl::SceneObject extra = make_box(
            "yellow_cube", 0.0, 0.4, 0.03, 0.03, 0.03, 1.0f, 1.0f, 0.0f);
        if (!mj_kdl::scene_add_object(&model, &data, &spec, extra)) {
            std::cerr << "FAIL: scene_add_object\n";
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        std::cout << "Bodies after add:    " << model->nbody << "\n";

        if (!mj_kdl::scene_remove_object(&model, &data, &spec, "yellow_cube")) {
            std::cerr << "FAIL: scene_remove_object\n";
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        std::cout << "Bodies after remove: " << model->nbody << "\n";
        std::cout << "OK\n";

        mj_kdl::destroy_scene(model, data);
        return 0;
    }

    // GUI: run simulate UI with gravity comp.
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i)
        data->ctrl[i] = data->qpos[s.kdl_to_mj_qpos[i]];

    std::cout << "\nGUI: close window to exit\n";
    mj_kdl::run_simulate_ui(model, data, urdf.c_str(),
        [&](mjModel*, mjData* d) {
            for (unsigned i = 0; i < n; ++i)
                d->ctrl[i] = d->qpos[s.kdl_to_mj_qpos[i]];
            KDL::JntArray q, g(n);
            mj_kdl::sync_to_kdl(&s, q);
            dyn.JntToGravity(q, g);
            mj_kdl::set_torques(&s, g);
        });

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
