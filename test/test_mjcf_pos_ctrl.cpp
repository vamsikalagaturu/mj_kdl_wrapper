/* test_mjcf_pos_ctrl.cpp
 * Joint position control on the Kinova GEN3 (MJCF).
 *
 * gen3.xml has high-gain position actuators (kp=2000, kv=100).
 * CtrlMode::POSITION writes the target joint position directly to ctrl[],
 * and the built-in servo drives the joint to that position.
 *
 * Uses a linearly interpolated trajectory from home to a nearby target pose. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7]    = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 1.5;  // s - linear interp from home to target
static constexpr double kSettleTime     = 0.5;  // s - extra time to let servo settle
static constexpr double kErrTol         = 0.05; // rad

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

class MjcfPosCtrlTest : public testing::Test
{
  protected:
    fs::path      root_;
    mjModel      *model_ = nullptr;
    mjData       *data_  = nullptr;
    mj_kdl::Robot s_;
    unsigned      n_ = 0;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found";
            return;
        }

        std::string arm_mjcf = (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();

        mj_kdl::SceneSpec sc;
        mj_kdl::RobotSpec r;
        r.path = arm_mjcf.c_str();
        sc.robots.push_back(r);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"));

        n_ = static_cast<unsigned>(s_.n_joints);

        KDL::JntArray q_home(n_);
        for (unsigned i = 0; i < n_; ++i) q_home(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&s_, q_home, false);
        mj_forward(model_, data_);

        s_.ctrl_mode = mj_kdl::CtrlMode::POSITION;
        for (unsigned i = 0; i < n_; ++i) { s_.jnt_pos_cmd[i] = kHomePose[i]; }
        mj_kdl::update(&s_);
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(MjcfPosCtrlTest, TrajectoryTracking)
{
    const double t_start = data_->time;
    const double t_end   = t_start + kMotionDuration + kSettleTime;

    while (data_->time < t_end) {
        mj_kdl::update(&s_);
        double alpha = std::clamp((data_->time - t_start) / kMotionDuration, 0.0, 1.0);
        for (unsigned i = 0; i < n_; ++i)
            s_.jnt_pos_cmd[i] = kHomePose[i] + alpha * (kTargetPose[i] - kHomePose[i]);
        mj_kdl::step(&s_);
    }

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err, std::abs(kTargetPose[i] - s_.jnt_pos_msr[i]));

    TEST_INFO(
      "MJCF position ctrl max joint error after "
      << (kMotionDuration + kSettleTime) << " s: " << std::fixed << std::setprecision(4) << max_err
      << " rad"
    );
    EXPECT_LE(max_err, kErrTol);
}

TEST_F(MjcfPosCtrlTest, ClampCtrlrange)
{
    // Set a command far outside any physical joint range and call update().
    // The ctrl[] written to MuJoCo must be clamped to [ctrlrange_lo, ctrlrange_hi].
    for (unsigned i = 0; i < n_; ++i) s_.jnt_pos_cmd[i] = 1e9;
    mj_kdl::update(&s_);

    for (unsigned i = 0; i < n_; ++i) {
        const int ci = s_.kdl_to_mj_ctrl[i];
        if (ci < 0) continue;
        if (!model_->actuator_ctrllimited[ci]) continue;
        double lo = model_->actuator_ctrlrange[2 * ci];
        double hi = model_->actuator_ctrlrange[2 * ci + 1];
        EXPECT_LE(data_->ctrl[ci], hi + 1e-12)
          << "ctrl[" << ci << "] exceeds ctrlrange upper bound";
        EXPECT_GE(data_->ctrl[ci], lo - 1e-12)
          << "ctrl[" << ci << "] below ctrlrange lower bound";
    }
}

TEST_F(MjcfPosCtrlTest, QfrcAppliedUnchangedInPositionMode)
{
    // In POSITION mode, update() must NOT zero qfrc_applied (user-set values
    // should be preserved so external disturbances can be applied).
    const int dof0 = s_.kdl_to_mj_dof[0];
    data_->qfrc_applied[dof0] = 5.0; // sentinel external disturbance
    mj_kdl::update(&s_);
    EXPECT_DOUBLE_EQ(data_->qfrc_applied[dof0], 5.0)
      << "update() clobbered qfrc_applied in POSITION mode";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
