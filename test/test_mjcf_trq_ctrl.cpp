/* test_mjcf_trq_ctrl.cpp
 * TORQUE mode on the Kinova GEN3 + Robotiq 2F-85: KDL gravity with the gripper's lumped mass,
 * impedance hold, and jnt_trq_msr as the drive torque. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

class MjcfTrqCtrlTest : public testing::Test
{
  protected:
    mj_kdl::Env         env_;
    mj_kdl::Robot       s_;
    const KDL::JntArray q_home_ = ex::home_q(7);

    void SetUp() override
    {
        const std::string arm_mjcf = ex::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string grp_mjcf = ex::find_asset("robotiq_2f85/2f85.xml");
        if (!fs::exists(arm_mjcf)) GTEST_SKIP() << arm_mjcf << " not found";
        if (!fs::exists(grp_mjcf)) GTEST_SKIP() << grp_mjcf << " not found";

        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf;
        rs.attachments.push_back(ex::gripper_attachment(grp_mjcf));
        mj_kdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = false;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link", "", &tool)
        );
        mj_kdl::set_joint_pos(&s_, q_home_);
    }

    int dof(const mj_kdl::Robot &r, unsigned i) const
    {
        const int jid = mj_name2id(env_.model, mjOBJ_JOINT, r.joint_names[i].c_str());
        return env_.model->jnt_dofadr[jid];
    }

    double max_gravity_error(const mj_kdl::Robot &r)
    {
        KDL::ChainDynParam dyn(r.chain, KDL::Vector(0, 0, -9.81));
        KDL::JntArray      g(7);
        EXPECT_GE(dyn.JntToGravity(q_home_, g), 0);
        double err = 0.0;
        for (unsigned i = 0; i < 7; ++i)
            err = std::max(err, std::abs(g(i) - env_.data->qfrc_bias[dof(r, i)]));
        return err;
    }
};

TEST_F(MjcfTrqCtrlTest, GravityIncludesTheGripperMass)
{
    // At home the gripper hangs off the wrist axes, so its weight loads the arm joints.
    mj_forward(env_.model, env_.data);
    EXPECT_LE(max_gravity_error(s_), 1e-9) << "measured 2e-14 Nm";

    mj_kdl::Robot bare;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&bare, &env_, "base_link", "bracelet_link"));
    EXPECT_GT(max_gravity_error(bare), 1.0) << "without the tool the chain misses ~5 Nm";
    mj_kdl::cleanup(&bare);
}

TEST_F(MjcfTrqCtrlTest, ImpedanceDrift)
{
    KDL::ChainFkSolverPos_recursive fk(s_.chain);
    KDL::ChainDynParam              dyn(s_.chain, KDL::Vector(0, 0, -9.81));
    KDL::Frame                      ee_init;
    fk.JntToCart(q_home_, ee_init);

    ASSERT_TRUE(mj_kdl::set_control_mode(&s_, mj_kdl::CtrlMode::TORQUE));
    ex::prime_gravity(s_, dyn, q_home_);
    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&env_);
        ex::pd_gravity(s_, dyn, q_home_, ex::kKp, ex::kKd);
        mj_kdl::step(&env_);
    }
    mj_kdl::update(&env_);
    EXPECT_LE((ex::tcp_frame(fk, s_).p - ee_init.p).Norm(), 1e-5) << "measured 7e-7 m";
}

TEST_F(MjcfTrqCtrlTest, TrqMsrReadsQfrcActuator)
{
    ASSERT_TRUE(mj_kdl::set_control_mode(&s_, mj_kdl::CtrlMode::TORQUE));
    for (unsigned i = 0; i < 7; ++i) s_.jnt_trq_cmd[i] = 1.0 + i;
    mj_kdl::update(&env_);
    mj_kdl::step(&env_);
    mj_kdl::update(&env_);

    for (unsigned i = 0; i < 7; ++i) {
        const double actuator = env_.data->qfrc_actuator[dof(s_, i)];
        EXPECT_DOUBLE_EQ(s_.jnt_trq_msr[i], actuator) << "joint " << i;
        EXPECT_NEAR(actuator, 1.0 + i, 1e-9) << "the motor delivers the command";
        EXPECT_NE(actuator, env_.data->qfrc_bias[dof(s_, i)]);
    }
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
