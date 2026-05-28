/* test_mjcf_trq_ctrl.cpp
 * Torque-mode control on the Kinova GEN3 + Robotiq 2F-85 (MJCF). */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kKp[7]       = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7]       = { 10, 20, 10, 20, 10, 20, 10 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

class MjcfTrqCtrlTest : public testing::Test
{
  protected:
    fs::path root_;
    mjModel *model_ = nullptr;
    mjData  *data_  = nullptr;

    mj_kdl::Robot                                    s_;
    unsigned                                         n_ = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;
    KDL::JntArray                                    q_home_;

    void SetUp() override
    {
        root_ = repo_root();
        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        if (!fs::exists(arm_mjcf)) {
            GTEST_SKIP() << arm_mjcf << " not found";
            return;
        }
        if (!fs::exists(grp_mjcf)) {
            GTEST_SKIP() << grp_mjcf << " not found";
            return;
        }

        mj_kdl::AttachmentSpec gs{
            .mjcf_path          = grp_mjcf.c_str(),
            .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
            .prefix             = "g_",
            .contact_exclusions = {},
        };
        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf.c_str();
        rs.attachments.push_back(gs);

        mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(
            &s_, model_, data_, "base_link", "bracelet_link", "", &tool
          )
        );

        n_   = s_.chain.getNrOfJoints();
        fk_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, -9.81));

        q_home_.resize(n_);
        for (unsigned i = 0; i < n_; ++i) q_home_(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&s_, q_home_);
        mj_forward(model_, data_);
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(MjcfTrqCtrlTest, GravityAccuracy)
{
    // At q=0 the arm is upright. KDL now includes gripper inertia via tool_body.
    KDL::JntArray q_zero(n_);
    mj_kdl::set_joint_pos(&s_, q_zero);
    mj_forward(model_, data_);

    KDL::JntArray g(n_);
    ASSERT_GE(dyn_->JntToGravity(q_zero, g), 0);

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err, std::abs(g(i) - data_->qfrc_bias[s_.kdl_to_mj_dof[i]]));
    EXPECT_LE(max_err, 5e-2);
}

TEST_F(MjcfTrqCtrlTest, ImpedanceDrift)
{
    KDL::Frame ee_init;
    fk_->JntToCart(q_home_, ee_init);

    s_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    KDL::JntArray q(n_), g(n_);
    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&s_);
        for (unsigned j = 0; j < n_; ++j) q(j) = s_.jnt_pos_msr[j];
        dyn_->JntToGravity(q, g);
        for (unsigned j = 0; j < n_; ++j) {
            s_.jnt_trq_cmd[j] =
              kKp[j] * (kHomePose[j] - s_.jnt_pos_msr[j]) - kKd[j] * s_.jnt_vel_msr[j] + g(j);
        }
        mj_kdl::step(&s_);
    }

    KDL::JntArray q_end(n_);
    for (unsigned j = 0; j < n_; ++j) q_end(j) = s_.jnt_pos_msr[j];
    KDL::Frame ee_end;
    fk_->JntToCart(q_end, ee_end);
    double drift = (ee_init.p - ee_end.p).Norm();
    EXPECT_LE(drift, 0.005);
}

TEST_F(MjcfTrqCtrlTest, TrqMsrReadsQfrcActuator)
{
    // jnt_trq_msr must reflect qfrc_actuator (the net actuator output torque),
    // NOT qfrc_bias (gravitational/Coriolis torques). After update() with zero
    // commands the robot is not yet running, so qfrc_actuator may be small but
    // the values must match element-wise.
    s_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    for (unsigned i = 0; i < n_; ++i) s_.jnt_trq_cmd[i] = 0.0;
    mj_kdl::update(&s_);
    mj_kdl::step(&s_);
    mj_kdl::update(&s_);

    for (unsigned i = 0; i < static_cast<unsigned>(s_.n_joints); ++i) {
        double expected = s_.data->qfrc_actuator[s_.kdl_to_mj_dof[i]];
        EXPECT_DOUBLE_EQ(s_.jnt_trq_msr[i], expected)
          << "jnt_trq_msr[" << i << "] does not match qfrc_actuator";
    }
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
