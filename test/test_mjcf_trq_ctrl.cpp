/* test_mjcf_trq_ctrl.cpp
 * Torque-mode control on the Kinova GEN3 + Robotiq 2F-85 (MJCF). */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found";
            return;
        }

        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

        mj_kdl::AttachmentSpec gs;
        gs.mjcf_path = grp_mjcf.c_str();
        gs.attach_to = "bracelet_link";
        gs.prefix    = "g_";
        gs.pos[2]    = -0.061525;
        gs.euler[0]  = 180.0;

        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf.c_str();
        rs.attachments.push_back(gs);

        mj_kdl::SceneSpec sc;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(
            &s_, model_, data_, "base_link", "bracelet_link", "", "g_base"
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
    TEST_INFO(
      "MJCF+gripper max|KDL - MuJoCo| gravity at q=0: " << std::fixed << std::setprecision(6)
                                                        << max_err << " Nm"
    );
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
    TEST_INFO(
      "MJCF+gripper impedance drift after 500 steps: " << std::fixed << std::setprecision(3)
                                                       << drift * 1000.0 << " mm"
    );
    EXPECT_LE(drift, 0.005);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
