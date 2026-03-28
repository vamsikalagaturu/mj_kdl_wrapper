/* test_mjcf_vel_ctrl.cpp
 * Joint velocity control on the Kinova GEN3 (MJCF).
 *
 * gen3.xml has high-gain position actuators (kp=2000, kv=100).
 * Velocity control is implemented in POSITION mode by integrating the
 * desired velocity into a position setpoint each step:
 *
 *   vel_des[i] = clamp(Kv * (target[i] - q[i]), -maxVel, maxVel)
 *   pos_cmd[i] += vel_des[i] * dt
 *
 * The position servo tracks the advancing setpoint, making the joint
 * follow the commanded velocity profile. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

static constexpr double kHomePose[7]   = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kTargetPose[7] = { 0.3, 0.5, 2.9, -2.0, 0.3, 1.2, 1.3 };
static constexpr double kKv            = 2.0;  // P gain [rad/s per rad error]
static constexpr double kMaxVel        = 0.6;  // velocity clamp [rad/s]
static constexpr double kTol           = 0.02; // convergence tolerance [rad]
static constexpr double kTimeout       = 5.0;  // max simulation time [s]

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

class MjcfVelCtrlTest : public testing::Test
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
        mj_kdl::set_joint_pos(&s_, q_home);

        // POSITION mode: initialise pos_cmd to home so the servo starts settled.
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

TEST_F(MjcfVelCtrlTest, Convergence)
{
    const double dt      = model_->opt.timestep;
    bool         arrived = false;

    while (data_->time < kTimeout && !arrived) {
        mj_kdl::update(&s_); // reads sensors, applies prev pos_cmd to servo

        double max_err = 0.0;
        for (unsigned i = 0; i < n_; ++i) {
            double err = kTargetPose[i] - s_.jnt_pos_msr[i];
            max_err    = std::max(max_err, std::abs(err));
            double vel = std::clamp(kKv * err, -kMaxVel, kMaxVel);
            s_.jnt_pos_cmd[i] += vel * dt; // integrate velocity into position setpoint
        }

        if (max_err < kTol) {
            arrived = true;
            // Freeze setpoint at current position to avoid overshoot.
            for (unsigned i = 0; i < n_; ++i) s_.jnt_pos_cmd[i] = s_.jnt_pos_msr[i];
        }

        mj_kdl::step(&s_);
    }

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err, std::abs(kTargetPose[i] - s_.jnt_pos_msr[i]));

    TEST_INFO(
      "MJCF velocity ctrl: " << (arrived ? "converged" : "TIMEOUT") << "  max_err=" << std::fixed
                             << std::setprecision(4) << max_err << " rad  t=" << data_->time << " s"
    );

    EXPECT_TRUE(arrived) << "velocity controller did not converge within " << kTimeout << " s";
    EXPECT_LE(max_err, kTol * 2);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
