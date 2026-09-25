/* test_mjcf_pos_ctrl.cpp
 * Joint position control on the Kinova GEN3 (MJCF).
 *
 * gen3.xml has high-gain position actuators (kp=2000, kv=100).
 * CtrlMode::POSITION writes the target joint position directly to ctrl[],
 * and the built-in servo drives the joint to that position.
 *
 * Uses a linearly interpolated trajectory from home to a nearby target pose. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

static constexpr double kHomePose[7]    = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 1.5;  // s - linear interp from home to target
static constexpr double kSettleTime     = 0.5;  // s - extra time to let servo settle
static constexpr double kErrTol         = 0.05; // rad

namespace fs = std::filesystem;
class MjcfPosCtrlTest : public testing::Test
{
  protected:
    fs::path      root_;
    mj_kdl::Env   env_;
    mjModel      *model_ = nullptr;
    mjData       *data_  = nullptr;
    mj_kdl::Robot s_;
    unsigned      n_ = 0;

    void SetUp() override
    {
        std::string arm_mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        if (!fs::exists(arm_mjcf)) {
            GTEST_SKIP() << arm_mjcf << " not found";
            return;
        }

        mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
        sc.robots.push_back(mj_kdl::RobotSpec{ .path = arm_mjcf, .attachments = {} });

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        model_ = env_.model;
        data_  = env_.data;
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));

        n_ = static_cast<unsigned>(s_.n_joints);

        KDL::JntArray q_home(n_);
        for (unsigned i = 0; i < n_; ++i) q_home(i) = kHomePose[i];
        mj_kdl::set_joint_pos(&s_, q_home);

        s_.ctrl_mode = mj_kdl::CtrlMode::POSITION;
        for (unsigned i = 0; i < n_; ++i) { s_.jnt_pos_cmd[i] = kHomePose[i]; }
        mj_kdl::update(&env_);
    }

    // The position servo driving joint i: robot 0's POSITION group is 1.
    int servo(unsigned i) const
    {
        const int jid = mj_name2id(model_, mjOBJ_JOINT, s_.joint_names[i].c_str());
        for (int a = 0; a < model_->nu; ++a) {
            if (model_->actuator_trnid[2 * a] == jid && model_->actuator_group[a] == 1) return a;
        }
        return -1;
    }
};

TEST_F(MjcfPosCtrlTest, TrajectoryTracking)
{
    const double t_start = data_->time;
    const double t_end   = t_start + kMotionDuration + kSettleTime;

    while (data_->time < t_end) {
        mj_kdl::update(&env_);
        double alpha = std::clamp((data_->time - t_start) / kMotionDuration, 0.0, 1.0);
        for (unsigned i = 0; i < n_; ++i)
            s_.jnt_pos_cmd[i] = kHomePose[i] + alpha * (kTargetPose[i] - kHomePose[i]);
        mj_kdl::step(&env_);
    }

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err, std::abs(kTargetPose[i] - s_.jnt_pos_msr[i]));

    EXPECT_LE(max_err, kErrTol);
}

TEST_F(MjcfPosCtrlTest, ClampCtrlrange)
{
    // Set a command far outside any physical joint range and call update().
    // The ctrl[] written to MuJoCo must be clamped to [ctrlrange_lo, ctrlrange_hi].
    for (unsigned i = 0; i < n_; ++i) s_.jnt_pos_cmd[i] = 1e9;
    mj_kdl::update(&env_);

    for (unsigned i = 0; i < n_; ++i) {
        const int ci = servo(i);
        ASSERT_GE(ci, 0) << s_.joint_names[i];
        if (!model_->actuator_ctrllimited[ci]) continue;
        EXPECT_EQ(s_.jnt_saturated[i], 1) << "joint " << i;
        double lo = model_->actuator_ctrlrange[2 * ci];
        double hi = model_->actuator_ctrlrange[2 * ci + 1];
        EXPECT_LE(data_->ctrl[ci], hi + 1e-12)
          << "ctrl[" << ci << "] exceeds ctrlrange upper bound";
        EXPECT_GE(data_->ctrl[ci], lo - 1e-12)
          << "ctrl[" << ci << "] below ctrlrange lower bound";
    }
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
