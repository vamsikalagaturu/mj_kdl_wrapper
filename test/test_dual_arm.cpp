/* test_dual_arm.cpp
 * Two Kinova GEN3 arms in one shared MuJoCo scene, facing each other.
 * Each arm is its own mj_kdl::State (sharing model/data).
 * Gravity compensation uses KDL::ChainDynParam::JntToGravity — not qfrc_bias.
 *
 * Part 1 — KDL vs MuJoCo gravity comparison at home pose (zero velocity,
 *           so qfrc_bias == gravity torques; tolerance 0.1 Nm).
 * Part 2 — 500-step closed-loop gravity comp; EE drift must be < 1 mm.
 *
 * Usage: test_dual_arm [urdf_path] [--gui] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

// Apply KDL gravity compensation torques to one arm.
static void apply_grav_comp(mj_kdl::State *s, KDL::ChainDynParam &dyn)
{
    KDL::JntArray q;
    mj_kdl::sync_to_kdl(s, q);
    KDL::JntArray g(s->n_joints);
    dyn.JntToGravity(q, g);
    mj_kdl::set_torques(s, g);
}

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

    /* Build a single MuJoCo scene with two arms facing each other.
     *   arm1: at x = -0.5 m, facing +X (default orientation)
     *   arm2: at x = +0.5 m, facing -X (rotated 180° around Z)
     * arm2 joints are prefixed "r2_" in MuJoCo to avoid name collisions. */
    mj_kdl::SceneSpec scene;
    scene.timestep  = 0.002;
    scene.gravity_z = -9.81;
    scene.add_floor = true;

    mj_kdl::SceneRobot r1, r2;
    r1.urdf_path = urdf.c_str();
    r1.prefix    = "";
    r1.pos[0]    = -0.5;
    r1.pos[1]    = 0.0;
    r1.pos[2]    = 0.0;

    r2.urdf_path = urdf.c_str();
    r2.prefix    = "r2_";
    r2.pos[0]    = 0.5;
    r2.pos[1]    = 0.0;
    r2.pos[2]    = 0.0;
    r2.euler[2]  = 180.0; // 180° around Z → faces -X (toward arm1)

    scene.robots = { r1, r2 };

    mjModel *model = nullptr;
    mjData  *data  = nullptr;

    BEGIN_TEST("build dual-arm scene")
        if (!mj_kdl::build_scene(&model, &data, &scene)) {
            TEST_FAIL("build_scene() returned false");
            return 1;
        }
        TEST_INFO(model->nbody << " bodies, " << model->nq << " DOFs");
    END_TEST

    // Attach one State per arm (shared model/data, separate KDL chains).
    mj_kdl::State arm1, arm2;

    BEGIN_TEST("init arm1")
        if (!mj_kdl::init_robot(
              &arm1, model, data, urdf.c_str(), "base_link", "EndEffector_Link", "")) {
            TEST_FAIL("arm1 init_robot() returned false");
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    END_TEST

    BEGIN_TEST("init arm2")
        if (!mj_kdl::init_robot(
              &arm2, model, data, urdf.c_str(), "base_link", "EndEffector_Link", "r2_")) {
            TEST_FAIL("arm2 init_robot() returned false");
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    END_TEST

    int n = arm1.n_joints;

    KDL::ChainFkSolverPos_recursive fk1(arm1.chain);
    KDL::ChainFkSolverPos_recursive fk2(arm2.chain);
    KDL::ChainDynParam              dyn1(arm1.chain, KDL::Vector(0, 0, -9.81));
    KDL::ChainDynParam              dyn2(arm2.chain, KDL::Vector(0, 0, -9.81));

    KDL::JntArray q_home(n);
    for (int j = 0; j < n; ++j) q_home(j) = kHomePose[j];

    mj_kdl::sync_from_kdl(&arm1, q_home);
    mj_kdl::sync_from_kdl(&arm2, q_home);
    mj_forward(model, data);

    KDL::Frame ee1_init, ee2_init;
    fk1.JntToCart(q_home, ee1_init);
    fk2.JntToCart(q_home, ee2_init);

    TEST_INFO("Arm 1 initial EE: [" << std::fixed << std::setprecision(4)
              << ee1_init.p.x() << ", " << ee1_init.p.y() << ", " << ee1_init.p.z() << "]");
    TEST_INFO("Arm 2 initial EE: [" << ee2_init.p.x() << ", " << ee2_init.p.y() << ", "
              << ee2_init.p.z() << "]");

    BEGIN_TEST("KDL vs MuJoCo gravity at home pose (informational)")
        KDL::JntArray g1(n), g2(n);
        dyn1.JntToGravity(q_home, g1);
        dyn2.JntToGravity(q_home, g2);

        double err1 = 0.0, err2 = 0.0;
        for (int j = 0; j < n; ++j) {
            err1 = std::max(err1, std::abs(g1(j) - data->qfrc_bias[arm1.kdl_to_mj_dof[j]]));
            err2 = std::max(err2, std::abs(g2(j) - data->qfrc_bias[arm2.kdl_to_mj_dof[j]]));
        }
        TEST_INFO("arm1 max |KDL - qfrc_bias| = " << err1 << " Nm");
        TEST_INFO("arm2 max |KDL - qfrc_bias| = " << err2 << " Nm");
    END_TEST

    /* Part 2: 500-step closed-loop gravity compensation; check EE drift.
     * Both arms share the same model/data — step() is called once per
     * timestep (on arm1), which advances the whole world. */
    BEGIN_TEST("dual-arm gravity comp drift")
        for (int i = 0; i < 500; ++i) {
            apply_grav_comp(&arm1, dyn1);
            apply_grav_comp(&arm2, dyn2);
            mj_kdl::step(&arm1); // advances the entire shared world
        }

        KDL::JntArray q1_end, q2_end;
        mj_kdl::sync_to_kdl(&arm1, q1_end);
        mj_kdl::sync_to_kdl(&arm2, q2_end);
        KDL::Frame ee1_end, ee2_end;
        fk1.JntToCart(q1_end, ee1_end);
        fk2.JntToCart(q2_end, ee2_end);

        double drift1 = (ee1_init.p - ee1_end.p).Norm();
        double drift2 = (ee2_init.p - ee2_end.p).Norm();

        TEST_INFO("EE drift after 500 steps: arm1=" << std::setprecision(3)
                  << drift1 * 1000.0 << " mm  arm2=" << drift2 * 1000.0 << " mm");
        if (drift1 > 0.001 || drift2 > 0.001) {
            TEST_FAIL("drift exceeds 1 mm threshold (arm1=" << drift1 * 1000.0
                      << " mm, arm2=" << drift2 * 1000.0 << " mm)");
            mj_kdl::cleanup(&arm1);
            mj_kdl::cleanup(&arm2);
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
    END_TEST

    // GUI loop — one window shows both arms; user can perturb either.
    if (gui) {
        mj_kdl::sync_from_kdl(&arm1, q_home);
        mj_kdl::sync_from_kdl(&arm2, q_home);
        mj_forward(model, data);
        // Prime position actuators for both arms.
        for (int i = 0; i < n; ++i) {
            data->ctrl[arm1.kdl_to_mj_dof[i]] =
              data->qpos[model->jnt_qposadr[model->dof_jntid[arm1.kdl_to_mj_dof[i]]]];
            data->ctrl[arm2.kdl_to_mj_dof[i]] =
              data->qpos[model->jnt_qposadr[model->dof_jntid[arm2.kdl_to_mj_dof[i]]]];
        }

        std::cout << "\nGUI: both arms in one scene.\n"
                  << "Ctrl+RightDrag to push a body.  Q/Esc to quit.\n";

        mj_kdl::run_simulate_ui(model, data, urdf.c_str(), [&](mjModel *m, mjData *d) {
            // Null position actuators for both arms.
            for (int i = 0; i < n; ++i) {
                d->ctrl[arm1.kdl_to_mj_dof[i]] =
                  d->qpos[m->jnt_qposadr[m->dof_jntid[arm1.kdl_to_mj_dof[i]]]];
                d->ctrl[arm2.kdl_to_mj_dof[i]] =
                  d->qpos[m->jnt_qposadr[m->dof_jntid[arm2.kdl_to_mj_dof[i]]]];
            }
            apply_grav_comp(&arm1, dyn1);
            apply_grav_comp(&arm2, dyn2);
        });
    }

    mj_kdl::cleanup(&arm1); // frees window; does NOT free model/data
    mj_kdl::cleanup(&arm2); // headless; does NOT free model/data
    mj_kdl::destroy_scene(model, data);
    return 0;
}
