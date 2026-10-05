/* test_mjcf_pos_ctrl.cpp
 * POSITION mode on the Kinova GEN3: the MJCF's own servos (kp = 2000, kv = 100) track a linear
 * joint trajectory, and commands are clamped to the servos' ctrlrange. */

#include "mjkdl/mjkdl.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace ex = mjkdl_examples;
namespace fs = std::filesystem;

static constexpr double kTargetPose[7]  = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kMotionDuration = 1.5;  // [s]
static constexpr double kSettleTime     = 0.5;  // [s]
static constexpr double kErrTol         = 0.01; // [rad], measured 0.0061

class MjcfPosCtrlTest : public testing::Test
{
  protected:
    mjkdl::Env   env_;
    mjkdl::Robot s_;

    void SetUp() override
    {
        const std::string arm = ex::find_asset("kinova_gen3/gen3.xml");
        if (!fs::exists(arm)) GTEST_SKIP() << arm << " not found";

        mjkdl::RobotSpec rs;
        rs.path = arm;
        mjkdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = false;
        sc.robots.push_back(rs);
        ASSERT_TRUE(mjkdl::init_env(&env_, &sc));
        ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));
        mjkdl::set_joint_pos(&s_, ex::home_q(7));
        ASSERT_TRUE(mjkdl::set_control_mode(&s_, mjkdl::CtrlMode::POSITION));
    }

    // The position servo driving joint i: robot 0's POSITION group is 1.
    int servo(int i) const
    {
        const int jid = mj_name2id(env_.model, mjOBJ_JOINT, s_.joint_names[i].c_str());
        for (int a = 0; a < env_.model->nu; ++a) {
            if (env_.model->actuator_trnid[2 * a] == jid && env_.model->actuator_group[a] == 1)
                return a;
        }
        return -1;
    }
};

TEST_F(MjcfPosCtrlTest, TrajectoryTracking)
{
    const double t_start = env_.data->time;
    while (env_.data->time < t_start + kMotionDuration + kSettleTime) {
        mjkdl::update(&env_);
        const double alpha = ex::clamp01((env_.data->time - t_start) / kMotionDuration);
        for (int i = 0; i < 7; ++i)
            s_.jnt_pos_cmd[i] = ex::kHomePose[i] + alpha * (kTargetPose[i] - ex::kHomePose[i]);
        mjkdl::step(&env_);
    }
    mjkdl::update(&env_);

    double max_err = 0.0;
    for (int i = 0; i < 7; ++i)
        max_err = std::max(max_err, std::abs(kTargetPose[i] - s_.jnt_pos_msr[i]));
    EXPECT_LE(max_err, kErrTol);
}

TEST_F(MjcfPosCtrlTest, ClampCtrlrange)
{
    for (int i = 0; i < 7; ++i) s_.jnt_pos_cmd[i] = 1e9;
    mjkdl::update(&env_);

    int limited = 0;
    for (int i = 0; i < 7; ++i) {
        const int a = servo(i);
        ASSERT_GE(a, 0) << s_.joint_names[i];
        if (!env_.model->actuator_ctrllimited[a]) continue;
        ++limited;
        EXPECT_EQ(s_.jnt_saturated[i], 1) << "joint " << i;
        EXPECT_EQ(env_.data->ctrl[a], env_.model->actuator_ctrlrange[2 * a + 1]) << "joint " << i;
    }
    EXPECT_EQ(limited, 3) << "Gen3 joints 2, 4 and 6 have a ctrlrange";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
