/* test_table_scene.cpp
 * Load a robot arm on a table with pickable objects (cubes and spheres).
 * Runs KDL gravity compensation so the arm holds position while the objects
 * can be perturbed in GUI mode.
 *
 * Also demonstrates runtime scene_add_object / scene_remove_object.
 *
 * Usage: test_table_scene [urdf_path] [--gui] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

int main(int argc, char *argv[])
{
    std::string urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool        gui  = false;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui")
            gui = true;
        else if (a[0] != '-')
            urdf = a;
    }

    // Table setup: surface at z = 0.7 m, robot mounted at table centre.
    mj_kdl::TableSpec table;
    table.enabled     = true;
    table.pos[0]      = 0.0;
    table.pos[1]      = 0.0;
    table.pos[2]      = 0.7; // surface height
    table.top_size[0] = 0.8; // half-extent x
    table.top_size[1] = 0.6; // half-extent y
    table.thickness   = 0.04;
    table.leg_radius  = 0.03;

    double surface_z = table.pos[2];

    std::vector<mj_kdl::SceneObject> objects;

    auto make_box = [&](const char *name,
                      double        x,
                      double        y,
                      double        hx,
                      double        hy,
                      double        hz,
                      float         r,
                      float         g,
                      float         b) {
        mj_kdl::SceneObject o;
        o.name    = name;
        o.shape   = mj_kdl::ObjShape::BOX;
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
    };

    auto make_sphere =
      [&](const char *name, double x, double y, double radius, float r, float g, float b) {
          mj_kdl::SceneObject o;
          o.name    = name;
          o.shape   = mj_kdl::ObjShape::SPHERE;
          o.size[0] = radius;
          o.pos[0]  = x;
          o.pos[1]  = y;
          o.pos[2]  = surface_z + radius;
          o.rgba[0] = r;
          o.rgba[1] = g;
          o.rgba[2] = b;
          o.rgba[3] = 1.0f;
          return o;
      };

    objects.push_back(make_box("red_cube", 0.35, 0.10, 0.03, 0.03, 0.03, 1.0f, 0.2f, 0.2f));
    objects.push_back(make_box("green_cube", 0.35, -0.10, 0.03, 0.03, 0.03, 0.2f, 1.0f, 0.2f));
    objects.push_back(make_box("blue_cube", 0.35, 0.30, 0.04, 0.04, 0.04, 0.2f, 0.2f, 1.0f));
    objects.push_back(make_sphere("orange_sphere", -0.20, 0.20, 0.035, 1.0f, 0.55f, 0.0f));
    objects.push_back(make_sphere("purple_sphere", -0.20, -0.20, 0.025, 0.7f, 0.0f, 0.9f));

    mj_kdl::SceneSpec spec;
    spec.table     = table;
    spec.objects   = objects;
    spec.add_floor = true;
    spec.gravity_z = -9.81;

    mj_kdl::SceneRobot robot;
    robot.urdf_path = urdf.c_str();
    robot.prefix    = "";
    robot.pos[0]    = 0.0;
    robot.pos[1]    = 0.0;
    robot.pos[2]    = surface_z;
    spec.robots.push_back(robot);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;

    BEGIN_TEST("build table scene")
        if (!mj_kdl::build_scene(&model, &data, &spec)) {
            TEST_FAIL("build_scene() returned false");
            return 1;
        }
        TEST_INFO(model->nbody << " bodies, " << model->nq << " DOFs");
    END_TEST

    mj_kdl::State s;

    BEGIN_TEST("init robot")
        if (!mj_kdl::init_robot(&s, model, data, urdf.c_str(), "base_link", "EndEffector_Link")) {
            TEST_FAIL("init_robot() returned false");
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        TEST_INFO(s.n_joints << " joints");
    END_TEST

    unsigned n = static_cast<unsigned>(s.n_joints);

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainDynParam              dyn(s.chain, KDL::Vector(0, 0, spec.gravity_z));

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(model, data);

    KDL::Frame ee_init;
    fk.JntToCart(q_home, ee_init);

    BEGIN_TEST("gravity comp drift")
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

        TEST_INFO("EE drift after 500 steps: " << std::fixed << std::setprecision(3)
                  << drift * 1000.0 << " mm");
        if (drift > 0.001) {
            TEST_FAIL("drift " << drift * 1000.0 << " mm exceeds 1 mm threshold");
            mj_kdl::cleanup(&s);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    END_TEST

    if (!gui) {
        int nbody_before = model->nbody;
        mj_kdl::cleanup(&s);

        mj_kdl::SceneObject extra =
          make_box("yellow_cube", 0.0, 0.4, 0.03, 0.03, 0.03, 1.0f, 1.0f, 0.0f);

        BEGIN_TEST("runtime add/remove object")
            if (!mj_kdl::scene_add_object(&model, &data, &spec, extra)) {
                TEST_FAIL("scene_add_object() returned false");
                mj_kdl::destroy_scene(model, data);
                return 1;
            }
            TEST_INFO("bodies: " << nbody_before << " -> " << model->nbody << " (after add)");

            if (!mj_kdl::scene_remove_object(&model, &data, &spec, "yellow_cube")) {
                TEST_FAIL("scene_remove_object() returned false");
                mj_kdl::destroy_scene(model, data);
                return 1;
            }
            TEST_INFO("bodies: " << model->nbody << " (after remove)");
        END_TEST

        mj_kdl::destroy_scene(model, data);
        return 0;
    }

    // GUI: run simulate UI with gravity comp.
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i) data->ctrl[i] = data->qpos[s.kdl_to_mj_qpos[i]];

    std::cout << "\nGUI: close window to exit\n";
    mj_kdl::run_simulate_ui(model, data, urdf.c_str(), [&](mjModel *, mjData *d) {
        for (unsigned i = 0; i < n; ++i) d->ctrl[i] = d->qpos[s.kdl_to_mj_qpos[i]];
        KDL::JntArray q, g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
    });

    mj_kdl::cleanup(&s);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
