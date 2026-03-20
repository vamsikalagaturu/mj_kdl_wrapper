/* test_dual_arm.cpp
 * Two Kinova GEN3 arms in one shared MuJoCo scene, facing each other.
 * Each arm is its own mj_kdl::Robot (sharing model/data).
 * Gravity compensation uses KDL::ChainDynParam::JntToGravity  - not qfrc_bias.
 *
 * GravityInformational  - KDL vs MuJoCo gravity comparison at home pose
 *   (zero velocity, so qfrc_bias == gravity torques); logged only, no assertion.
 * DualArmDrift  - 500-step closed-loop gravity comp; EE drift must be < 1 mm.
 *
 * Usage: test_dual_arm [urdf_path] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

/* Read state, compute KDL gravity torques, store in jnt_trq_cmd. */
static void apply_grav_comp(mj_kdl::Robot *s, KDL::ChainDynParam &dyn)
{
    mj_kdl::update(s);
    KDL::JntArray q(s->n_joints), g(s->n_joints);
    for (int i = 0; i < s->n_joints; ++i) q(i) = s->jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (int i = 0; i < s->n_joints; ++i) s->jnt_trq_cmd[i] = g(i);
}

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

/* g_urdf is set once from main() before RUN_ALL_TESTS(). */
static std::string g_urdf;

class DualArmTest : public ::testing::Test
{
  protected:
    std::string                                      urdf_;
    mjModel                                         *model = nullptr;
    mjData                                          *data  = nullptr;
    mj_kdl::Robot                                    arm1, arm2;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk1;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk2;
    std::unique_ptr<KDL::ChainDynParam>              dyn1;
    std::unique_ptr<KDL::ChainDynParam>              dyn2;
    KDL::JntArray                                    q_home;
    int                                              n = 0;

    void SetUp() override
    {
        urdf_ = g_urdf;

        /* Build a single MuJoCo scene with two arms facing each other.
         *   arm1: at x = -0.5 m, facing +X (default orientation)
         *   arm2: at x = +0.5 m, facing -X (rotated 180° around Z)
         * arm2 joints are prefixed "r2_" in MuJoCo to avoid name collisions. */
        mj_kdl::SceneSpec scene;
        scene.timestep  = 0.002;
        scene.gravity_z = -9.81;
        scene.add_floor = true;

        mj_kdl::RobotSpec r1, r2;
        r1.path   = urdf_.c_str();
        r1.prefix = "";
        r1.pos[0] = -0.5;
        r1.pos[1] = 0.0;
        r1.pos[2] = 0.0;

        r2.path     = urdf_.c_str();
        r2.prefix   = "r2_";
        r2.pos[0]   = 0.5;
        r2.pos[1]   = 0.0;
        r2.pos[2]   = 0.0;
        r2.euler[2] = 180.0;

        scene.robots = { r1, r2 };

        ASSERT_TRUE(mj_kdl::build_scene_from_urdfs(&model, &data, &scene))
          << "build_scene_from_urdfs() returned false";
        TEST_INFO(model->nbody << " bodies, " << model->nq << " DOFs");

        ASSERT_TRUE(mj_kdl::init_robot_from_urdf(
          &arm1, model, data, urdf_.c_str(), "base_link", "EndEffector_Link", ""))
          << "arm1 init_robot_from_urdf() returned false";

        ASSERT_TRUE(mj_kdl::init_robot_from_urdf(
          &arm2, model, data, urdf_.c_str(), "base_link", "EndEffector_Link", "r2_"))
          << "arm2 init_robot_from_urdf() returned false";

        n    = arm1.n_joints;
        fk1  = std::make_unique<KDL::ChainFkSolverPos_recursive>(arm1.chain);
        fk2  = std::make_unique<KDL::ChainFkSolverPos_recursive>(arm2.chain);
        dyn1 = std::make_unique<KDL::ChainDynParam>(arm1.chain, KDL::Vector(0, 0, -9.81));
        dyn2 = std::make_unique<KDL::ChainDynParam>(arm2.chain, KDL::Vector(0, 0, -9.81));

        q_home.resize(n);
        for (int j = 0; j < n; ++j) q_home(j) = kHomePose[j];
    }

    void TearDown() override
    {
        mj_kdl::cleanup(&arm1);
        mj_kdl::cleanup(&arm2);
        if (model) mj_kdl::destroy_scene(model, data);
    }
};

TEST_F(DualArmTest, GravityInformational)
{
    mj_kdl::set_joint_pos(&arm1, q_home);
    mj_kdl::set_joint_pos(&arm2, q_home);
    mj_forward(model, data);

    KDL::JntArray g1(n), g2(n);
    dyn1->JntToGravity(q_home, g1);
    dyn2->JntToGravity(q_home, g2);

    double err1 = 0.0, err2 = 0.0;
    for (int j = 0; j < n; ++j) {
        err1 = std::max(err1, std::abs(g1(j) - data->qfrc_bias[arm1.kdl_to_mj_dof[j]]));
        err2 = std::max(err2, std::abs(g2(j) - data->qfrc_bias[arm2.kdl_to_mj_dof[j]]));
    }
    TEST_INFO("arm1 max |KDL - qfrc_bias| = " << err1 << " Nm");
    TEST_INFO("arm2 max |KDL - qfrc_bias| = " << err2 << " Nm");
}

TEST_F(DualArmTest, DualArmDrift)
{
    /* Sync both arms to home pose and record initial EE frames. */
    mj_kdl::set_joint_pos(&arm1, q_home);
    mj_kdl::set_joint_pos(&arm2, q_home);
    mj_forward(model, data);
    arm1.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    arm2.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    KDL::Frame ee1_init, ee2_init;
    fk1->JntToCart(q_home, ee1_init);
    fk2->JntToCart(q_home, ee2_init);

    TEST_INFO("Arm 1 initial EE: [" << std::fixed << std::setprecision(4) << ee1_init.p.x() << ", "
                                    << ee1_init.p.y() << ", " << ee1_init.p.z() << "]");
    TEST_INFO("Arm 2 initial EE: [" << ee2_init.p.x() << ", " << ee2_init.p.y() << ", "
                                    << ee2_init.p.z() << "]");

    /* Prime jnt_trq_cmd for both arms so the first update() applies compensation
     * immediately, not zero torques (which would impart velocity that gravity comp
     * cannot damp). */
    {
        KDL::JntArray g1(n), g2(n);
        dyn1->JntToGravity(q_home, g1);
        dyn2->JntToGravity(q_home, g2);
        for (int j = 0; j < n; ++j) arm1.jnt_trq_cmd[j] = g1(j);
        for (int j = 0; j < n; ++j) arm2.jnt_trq_cmd[j] = g2(j);
    }

    /* Run 500-step closed-loop gravity compensation. Both arms share the same
     * model/data; step() on arm1 advances the entire world. */
    for (int i = 0; i < 500; ++i) {
        apply_grav_comp(&arm1, *dyn1);
        apply_grav_comp(&arm2, *dyn2);
        mj_kdl::step(&arm1);
    }

    KDL::JntArray q1_end(n), q2_end(n);
    for (int j = 0; j < n; ++j) {
        q1_end(j) = arm1.jnt_pos_msr[j];
        q2_end(j) = arm2.jnt_pos_msr[j];
    }
    KDL::Frame ee1_end, ee2_end;
    fk1->JntToCart(q1_end, ee1_end);
    fk2->JntToCart(q2_end, ee2_end);

    double drift1 = (ee1_init.p - ee1_end.p).Norm();
    double drift2 = (ee2_init.p - ee2_end.p).Norm();

    TEST_INFO("EE drift after 500 steps: arm1=" << std::setprecision(3) << drift1 * 1000.0
                                                << " mm  arm2=" << drift2 * 1000.0 << " mm");

    ASSERT_LE(drift1, 0.001) << "arm1 drift " << drift1 * 1000.0 << " mm exceeds 1 mm threshold";
    ASSERT_LE(drift2, 0.001) << "arm2 drift " << drift2 * 1000.0 << " mm exceeds 1 mm threshold";
}

int main(int argc, char *argv[])
{
    g_urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();

    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-') g_urdf = argv[i];

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
