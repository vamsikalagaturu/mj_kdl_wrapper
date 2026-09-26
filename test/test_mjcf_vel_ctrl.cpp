/* test_mjcf_vel_ctrl.cpp
 * VELOCITY mode on the Kinova GEN3: the <joint>_velocity actuators added through
 * RobotSpec::modes drive the arm from home to a target under a clamped proportional velocity
 * command, as ex_joint_ctrl's second motion does. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

static constexpr double kTargetPose[7] = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kVelGain       = 500.0; // velocity actuator kv [Nm s/rad]
static constexpr double kKv            = 2.0;   // [rad/s per rad]
static constexpr double kMaxVel        = 0.6;   // [rad/s]
static constexpr double kTol           = 0.01;  // [rad]
static constexpr double kTimeout       = 5.0;   // [s]

TEST(MjcfVelCtrlTest, ConvergesInVelocityMode)
{
    const std::string arm = ex::find_menagerie_model("kinova_gen3/gen3.xml");
    if (!fs::exists(arm)) GTEST_SKIP() << arm << " not found";

    mj_kdl::RobotSpec rs;
    rs.path  = arm;
    rs.modes = { { .mode = mj_kdl::CtrlMode::VELOCITY, .joints = {}, .kv = kVelGain } };
    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = false;
    sc.robots.push_back(rs);

    mj_kdl::Env   env;
    mj_kdl::Robot robot;
    ASSERT_TRUE(mj_kdl::init_env(&env, &sc));
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link"));
    mj_kdl::set_joint_pos(&robot, ex::home_q(7));
    ASSERT_TRUE(mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::VELOCITY));

    double max_err = 0.0;
    while (env.data->time < kTimeout) {
        mj_kdl::update(&env);
        max_err = 0.0;
        for (int i = 0; i < 7; ++i) {
            const double err     = kTargetPose[i] - robot.jnt_pos_msr[i];
            max_err              = std::max(max_err, std::abs(err));
            robot.jnt_vel_cmd[i] = ex::clamp_abs(kKv * err, kMaxVel);
        }
        if (max_err < kTol) break;
        mj_kdl::step(&env);
    }
    EXPECT_LT(max_err, kTol);
    EXPECT_LT(env.data->time, 2.5) << "measured 1.93 s";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
