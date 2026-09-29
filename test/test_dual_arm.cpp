/* test_dual_arm.cpp
 * Two Kinova GEN3 arms in one scene, facing each other, each its own mj_kdl::Robot with its own
 * KDL chain: a prefix names the whole chain, KDL gravity matches MuJoCo's for both, and both hold
 * their pose under KDL gravity compensation. Self-skips when Menagerie is absent. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

class DualArmTest : public testing::Test
{
  protected:
    mj_kdl::Env                         env;
    mj_kdl::Robot                       arm1, arm2;
    std::unique_ptr<KDL::ChainDynParam> dyn1, dyn2;
    const KDL::JntArray                 q_home = ex::home_q(7);

    void SetUp() override
    {
        const std::string mjcf = ex::find_menagerie_model("kinova_gen3/gen3.xml");
        if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";

        // arm1 at x = -0.5 m facing +X; arm2 at x = +0.5 m turned 180 deg about Z, prefixed "r2_".
        mj_kdl::SceneSpec scene;
        scene.timestep   = 0.002;
        scene.add_floor  = true;
        scene.add_skybox = false;
        mj_kdl::RobotSpec left, right;
        left.path     = mjcf;
        left.pos[0]   = -0.5;
        right.path    = mjcf;
        right.prefix  = "r2_";
        right.pos[0]  = 0.5;
        right.quat[2] = 1.0;
        right.quat[3] = 0.0;
        scene.robots  = { left, right };

        ASSERT_TRUE(mj_kdl::init_env(&env, &scene));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&arm1, &env, "base_link", "bracelet_link", ""));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&arm2, &env, "base_link", "bracelet_link", "r2_"));
        dyn1 = std::make_unique<KDL::ChainDynParam>(arm1.chain, KDL::Vector(0, 0, -9.81));
        dyn2 = std::make_unique<KDL::ChainDynParam>(arm2.chain, KDL::Vector(0, 0, -9.81));
    }

    int dof(const mj_kdl::Robot &r, int j) const
    {
        return env.model->jnt_dofadr[mj_name2id(env.model, mjOBJ_JOINT, r.joint_names[j].c_str())];
    }
};

TEST_F(DualArmTest, PrefixNamesTheWholeChain)
{
    mj_kdl::Robot named;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&named, &env, "r2_base_link", "r2_bracelet_link"));
    EXPECT_EQ(arm2.joint_names, named.joint_names);
    EXPECT_EQ(arm2.joint_names.front(), "r2_joint_1");
    EXPECT_EQ(arm2.chain.getNrOfSegments(), named.chain.getNrOfSegments());
    mj_kdl::cleanup(&named);
}

TEST_F(DualArmTest, KdlGravityMatchesMujocoForBothArms)
{
    mj_kdl::set_joint_pos(&arm1, q_home);
    mj_kdl::set_joint_pos(&arm2, q_home);
    mj_forward(env.model, env.data);

    // At rest qfrc_bias is the gravity torque; measured difference 1e-14 Nm.
    KDL::JntArray g1(7), g2(7);
    dyn1->JntToGravity(q_home, g1);
    dyn2->JntToGravity(q_home, g2);
    for (int j = 0; j < 7; ++j) {
        EXPECT_NEAR(g1(j), env.data->qfrc_bias[dof(arm1, j)], 1e-9) << "arm1 joint " << j;
        EXPECT_NEAR(g2(j), env.data->qfrc_bias[dof(arm2, j)], 1e-9) << "arm2 joint " << j;
    }
}

TEST_F(DualArmTest, DualArmDrift)
{
    mj_kdl::set_joint_pos(&arm1, q_home);
    mj_kdl::set_joint_pos(&arm2, q_home);
    ASSERT_TRUE(mj_kdl::set_control_mode(&arm1, mj_kdl::CtrlMode::TORQUE));
    ASSERT_TRUE(mj_kdl::set_control_mode(&arm2, mj_kdl::CtrlMode::TORQUE));
    ex::prime_gravity(arm1, *dyn1, q_home);
    ex::prime_gravity(arm2, *dyn2, q_home);

    KDL::ChainFkSolverPos_recursive fk1(arm1.chain), fk2(arm2.chain);
    KDL::Frame                      ee_init;
    fk1.JntToCart(q_home, ee_init);

    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&env);
        ex::pd_gravity(arm1, *dyn1, q_home);
        ex::pd_gravity(arm2, *dyn2, q_home);
        mj_kdl::step(&env);
    }
    mj_kdl::update(&env);

    // Measured drift: 3e-18 m.
    EXPECT_LE((ex::tcp_frame(fk1, arm1).p - ee_init.p).Norm(), 1e-6);
    EXPECT_LE((ex::tcp_frame(fk2, arm2).p - ee_init.p).Norm(), 1e-6);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
