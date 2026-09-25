/* test_control_modes.cpp
 * Control modes through actuator groups: the actuators build_scene adds, switching without a
 * jump, torque limits, two robots in different modes, a gripper that stays in POSITION while its
 * arm switches, and a motor-driven wheel in VELOCITY and TORQUE (the Eddie case) driven through
 * SceneState. Arm tests (Gen3, UR5e) self-skip without Menagerie. */

#include <gtest/gtest.h>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <kdl/chaindynparam.hpp>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr int kPos = static_cast<int>(mj_kdl::CtrlMode::POSITION);
static constexpr int kTrq = static_cast<int>(mj_kdl::CtrlMode::TORQUE);

static bool group_enabled(const mjModel *m, int group)
{
    return !(m->opt.disableactuator & (1 << group));
}

struct ArmCase
{
    const char         *name;
    const char         *mjcf;
    const char         *base;
    const char         *tip;
    std::vector<double> home;
    int                 limited_joint; // a joint whose forcerange is `limit`
    double              limit;
};

static const ArmCase kArms[] = {
    { "Gen3",
      "kinova_gen3/gen3.xml",
      "base_link",
      "bracelet_link",
      { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 },
      1,
      105.0 },
    { "UR5e",
      "universal_robots_ur5e/ur5e.xml",
      "base",
      "wrist_3_link",
      { -1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0 },
      1,
      150.0 },
};

class ArmModesTest : public testing::TestWithParam<ArmCase>
{
  protected:
    std::string       mjcf_;
    mjModel          *model_ = nullptr;
    mjData           *data_  = nullptr;
    mj_kdl::SceneSpec spec_;
    mj_kdl::Robot     arm_;
    int               n_ = 0;

    void SetUp() override
    {
        const ArmCase &c = GetParam();
        mjcf_            = mj_kdl_examples::find_menagerie_model(c.mjcf);
        if (!fs::exists(mjcf_)) GTEST_SKIP() << c.mjcf << " not found";
        spec_.timestep   = 0.002;
        spec_.add_floor  = true;
        spec_.add_skybox = false;
        spec_.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf_.c_str() });
        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &spec_));
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&arm_, model_, data_, c.base, c.tip));
        n_ = arm_.n_joints;
        ASSERT_EQ(n_, static_cast<int>(c.home.size()));
        KDL::JntArray q(n_);
        for (int i = 0; i < n_; ++i) q(i) = c.home[i];
        mj_kdl::set_joint_pos(&arm_, q);
    }

    void TearDown() override
    {
        if (model_) mj_kdl::destroy_scene(model_, data_);
    }

    double max_error(const std::vector<double> &q_ref) const
    {
        double err = 0.0;
        for (int i = 0; i < n_; ++i) err = std::max(err, std::abs(arm_.jnt_pos_msr[i] - q_ref[i]));
        return err;
    }
};

TEST_P(ArmModesTest, TorqueGroupIsAddedAndStartsDisabled)
{
    EXPECT_EQ(arm_.robot_index, 0);
    EXPECT_EQ(arm_.ctrl_mode, mj_kdl::CtrlMode::POSITION);
    for (int i = 0; i < n_; ++i) {
        const int servo = arm_.mode_ctrl[kPos][i];
        const int motor = arm_.mode_ctrl[kTrq][i];
        ASSERT_GE(servo, 0);
        ASSERT_GE(motor, 0);
        EXPECT_EQ(model_->actuator_group[servo], 1);
        EXPECT_EQ(model_->actuator_group[motor], 2);
        EXPECT_EQ(std::string(mj_id2name(model_, mjOBJ_ACTUATOR, motor)), arm_.joint_names[i] + "_torque");
    }
    EXPECT_TRUE(group_enabled(model_, 1));
    EXPECT_FALSE(group_enabled(model_, 2));
}

TEST_P(ArmModesTest, PositionTracksAJointTrajectory)
{
    mj_kdl::update(&arm_);
    const std::vector<double> q0    = arm_.jnt_pos_msr;
    std::vector<double>       q_ref = q0;
    double                    worst = 0.0;
    for (int k = 0; k < 750; ++k) {
        const double s = std::min(1.0, k / 500.0);
        for (int i = 0; i < n_; ++i) q_ref[i] = q0[i] - 0.2 * s;
        arm_.jnt_pos_cmd = q_ref;
        mj_kdl::update(&arm_);
        mj_kdl::step(&arm_);
        if (k >= 500) worst = std::max(worst, max_error(q_ref));
    }
    EXPECT_LT(worst, 0.05);
}

TEST_P(ArmModesTest, SwitchesBetweenPositionAndTorqueWithoutAJump)
{
    KDL::ChainDynParam dyn(arm_.chain, KDL::Vector(0, 0, spec_.gravity_z));
    KDL::JntArray      q(n_), g(n_);

    mj_kdl::update(&arm_);
    const std::vector<double> q_ref = arm_.jnt_pos_msr;
    arm_.jnt_pos_cmd                = q_ref;
    for (int k = 0; k < 200; ++k) {
        mj_kdl::update(&arm_);
        mj_kdl::step(&arm_);
    }

    ASSERT_TRUE(mj_kdl::set_control_mode(&arm_, mj_kdl::CtrlMode::TORQUE));
    EXPECT_FALSE(group_enabled(model_, 1));
    EXPECT_TRUE(group_enabled(model_, 2));
    double worst = 0.0;
    for (int k = 0; k < 300; ++k) {
        mj_kdl::update(&arm_);
        for (int i = 0; i < n_; ++i) q(i) = arm_.jnt_pos_msr[i];
        dyn.JntToGravity(q, g);
        for (int i = 0; i < n_; ++i) {
            arm_.jnt_trq_cmd[i] =
              g(i) + 50.0 * (q_ref[i] - arm_.jnt_pos_msr[i]) - 5.0 * arm_.jnt_vel_msr[i];
        }
        mj_kdl::update(&arm_);
        mj_kdl::step(&arm_);
        worst = std::max(worst, max_error(q_ref));
        for (int i = 0; i < n_; ++i) EXPECT_EQ(data_->actuator_force[arm_.mode_ctrl[kPos][i]], 0.0);
    }
    EXPECT_LT(worst, 0.01) << "torque hold after the switch";

    ASSERT_TRUE(mj_kdl::set_control_mode(&arm_, mj_kdl::CtrlMode::POSITION));
    worst = 0.0;
    for (int k = 0; k < 200; ++k) {
        mj_kdl::update(&arm_);
        mj_kdl::step(&arm_);
        worst = std::max(worst, max_error(q_ref));
    }
    EXPECT_LT(worst, 0.01) << "position hold after switching back";
}

TEST_P(ArmModesTest, TorqueIsLimitedByTheModelsForcerange)
{
    const int    j     = GetParam().limited_joint;
    const double limit = GetParam().limit;
    ASSERT_TRUE(mj_kdl::set_control_mode(&arm_, mj_kdl::CtrlMode::TORQUE));
    arm_.jnt_trq_cmd[j] = 1000.0;
    mj_kdl::update(&arm_);
    mj_forward(model_, data_);
    EXPECT_DOUBLE_EQ(data_->actuator_force[arm_.mode_ctrl[kTrq][j]], limit);
    EXPECT_DOUBLE_EQ(data_->qfrc_actuator[arm_.kdl_to_mj_dof[j]], limit);
    EXPECT_DOUBLE_EQ(mj_kdl::joint_force_limits(&arm_)[j], limit);
    for (int i = 0; i < n_; ++i) EXPECT_EQ(arm_.jnt_saturated[i], i == j) << "joint " << i;

    arm_.jnt_trq_cmd[j] = 10.0;
    mj_kdl::update(&arm_);
    EXPECT_EQ(arm_.jnt_saturated[j], 0);
}

TEST_P(ArmModesTest, QfrcAppliedIsLeftToTheUser)
{
    auto expect_untouched = [&](const char *when) {
        for (int i = 0; i < n_; ++i)
            EXPECT_EQ(data_->qfrc_applied[arm_.kdl_to_mj_dof[i]], 5.0) << when << ", joint " << i;
    };
    for (int i = 0; i < n_; ++i) data_->qfrc_applied[arm_.kdl_to_mj_dof[i]] = 5.0;

    arm_.jnt_pos_cmd[1] += 0.1;
    mj_kdl::update(&arm_);
    expect_untouched("POSITION");

    ASSERT_TRUE(mj_kdl::set_control_mode(&arm_, mj_kdl::CtrlMode::TORQUE));
    expect_untouched("switch to TORQUE");
    for (int i = 0; i < n_; ++i) arm_.jnt_trq_cmd[i] = 3.0;
    mj_kdl::update(&arm_);
    expect_untouched("TORQUE");

    arm_.ctrl_mode = mj_kdl::CtrlMode::POSITION;
    mj_kdl::update(&arm_);
    expect_untouched("switch back through ctrl_mode");
}

TEST_P(ArmModesTest, TwoArmsRunDifferentModes)
{
    const ArmCase &c = GetParam();
    mj_kdl::destroy_scene(model_, data_);
    model_ = nullptr;
    spec_.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf_.c_str(), .prefix = "r2_", .pos = { 1.0, 0, 0 } });
    ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &spec_));

    const std::string base = std::string("r2_") + c.base;
    const std::string tip  = std::string("r2_") + c.tip;
    mj_kdl::Robot     a, b;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&a, model_, data_, c.base, c.tip));
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&b, model_, data_, base.c_str(), tip.c_str()));
    EXPECT_EQ(a.robot_index, 0);
    EXPECT_EQ(b.robot_index, 1);

    ASSERT_TRUE(mj_kdl::set_control_mode(&a, mj_kdl::CtrlMode::TORQUE));
    EXPECT_FALSE(group_enabled(model_, 1));
    EXPECT_TRUE(group_enabled(model_, 2));
    EXPECT_TRUE(group_enabled(model_, 4)) << "the second arm stays in POSITION";
    EXPECT_FALSE(group_enabled(model_, 5));
}

INSTANTIATE_TEST_SUITE_P(Arms, ArmModesTest, testing::ValuesIn(kArms), [](const auto &info) {
    return std::string(info.param.name);
});

TEST(GripperModesTest, GripperStaysInPositionWhileTheArmSwitches)
{
    const std::string arm_mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
    const std::string grp_mjcf = mj_kdl_examples::find_asset("robotiq_2f85/2f85.xml");
    if (!fs::exists(arm_mjcf)) GTEST_SKIP() << "kinova_gen3/gen3.xml not found";
    if (!fs::exists(grp_mjcf)) GTEST_SKIP() << "robotiq_2f85/2f85.xml not found";

    mj_kdl::RobotSpec rs{ .path = arm_mjcf.c_str() };
    rs.attachments.push_back(mj_kdl::AttachmentSpec{
      .mjcf_path          = grp_mjcf.c_str(),
      .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
      .prefix             = "g_",
      .contact_exclusions = {},
    });
    mj_kdl::SceneSpec spec;
    spec.timestep   = 0.002;
    spec.add_floor  = true;
    spec.add_skybox = false;
    spec.robots.push_back(rs);

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    ASSERT_TRUE(mj_kdl::build_scene(&model, &data, &spec));
    mj_kdl::Robot arm;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&arm, model, data, "base_link", "bracelet_link"));
    KDL::JntArray q(7), g(7);
    for (int i = 0; i < 7; ++i) q(i) = kArms[0].home[i];
    mj_kdl::set_joint_pos(&arm, q);

    const int fingers = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    const int driver  = mj_name2id(model, mjOBJ_JOINT, "g_left_driver_joint");
    ASSERT_GE(fingers, 0);
    ASSERT_GE(driver, 0);
    EXPECT_EQ(model->actuator_group[fingers], 0);
    EXPECT_LT(mj_name2id(model, mjOBJ_ACTUATOR, "g_left_driver_joint_torque"), 0)
      << "the gripper gets no torque actuator";

    ASSERT_TRUE(mj_kdl::set_control_mode(&arm, mj_kdl::CtrlMode::TORQUE));
    EXPECT_TRUE(group_enabled(model, 0));

    KDL::ChainDynParam        dyn(arm.chain, KDL::Vector(0, 0, spec.gravity_z));
    const std::vector<double> q_ref = arm.jnt_pos_msr;
    auto                      run   = [&](double finger_cmd) {
        data->ctrl[fingers] = finger_cmd;
        for (int k = 0; k < 500; ++k) {
            mj_kdl::update(&arm);
            for (int i = 0; i < 7; ++i) q(i) = arm.jnt_pos_msr[i];
            dyn.JntToGravity(q, g);
            for (int i = 0; i < 7; ++i)
                arm.jnt_trq_cmd[i] =
                  g(i) + 50.0 * (q_ref[i] - arm.jnt_pos_msr[i]) - 5.0 * arm.jnt_vel_msr[i];
            mj_kdl::update(&arm);
            mj_kdl::step(&arm);
        }
        return data->qpos[model->jnt_qposadr[driver]];
    };
    EXPECT_GT(run(0.82), 0.7) << "closes";
    EXPECT_LT(run(0.0), 0.05) << "opens";

    mj_kdl::destroy_scene(model, data);
}

class MotorWheelModesTest : public testing::Test
{
  protected:
    const std::string fixture_ = std::string(MJ_KDL_TEST_FIXTURES) + "/motor_wheel.xml";
    mjModel          *model_   = nullptr;
    mjData           *data_    = nullptr;
    mj_kdl::SceneSpec spec_;

    void SetUp() override
    {
        spec_.timestep   = 0.002;
        spec_.add_floor  = false;
        spec_.add_skybox = false;
        spec_.robots.push_back(mj_kdl::RobotSpec{
          .path  = fixture_.c_str(),
          .modes = { { .mode = mj_kdl::CtrlMode::VELOCITY, .joints = { "wheel" }, .kv = 2.0 } },
        });
        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &spec_));
    }

    void TearDown() override
    {
        if (model_) mj_kdl::destroy_scene(model_, data_);
    }

    int actuator(const char *name) const { return mj_name2id(model_, mjOBJ_ACTUATOR, name); }
};

TEST_F(MotorWheelModesTest, OnlyTheListedJointTakesModes)
{
    ASSERT_GE(actuator("wheel_velocity"), 0);
    EXPECT_EQ(model_->actuator_group[actuator("wheel")], 2) << "a motor is natively TORQUE";
    EXPECT_EQ(model_->actuator_group[actuator("wheel_velocity")], 3);
    EXPECT_EQ(model_->actuator_group[actuator("pivot")], 0) << "the pivot is left alone";
    EXPECT_FALSE(group_enabled(model_, 3));
    EXPECT_FALSE(mj_kdl::set_control_mode(model_, data_, 0, mj_kdl::CtrlMode::POSITION));
}

TEST_F(MotorWheelModesTest, VelocityTracksThenTorqueTakesOver)
{
    mj_kdl::SceneState scene;
    ASSERT_TRUE(mj_kdl::init_scene_state(&scene, model_));
    mj_kdl::SceneActuatorSlot *vel   = mj_kdl::bind_scene_actuator(&scene, "wheel_velocity");
    mj_kdl::SceneJointSlot    *wheel = mj_kdl::bind_scene_joint(&scene, "wheel");
    mj_kdl::SceneJointSlot    *pivot = mj_kdl::bind_scene_joint(&scene, "pivot");
    ASSERT_NE(vel, nullptr);
    ASSERT_NE(wheel, nullptr);
    ASSERT_NE(pivot, nullptr);

    ASSERT_TRUE(mj_kdl::set_control_mode(model_, data_, 0, mj_kdl::CtrlMode::VELOCITY));
    vel->command = 5.0;
    for (int k = 0; k < 1000; ++k) {
        mj_kdl::apply_scene_state(&scene, data_);
        mj_step(model_, data_);
    }
    mj_kdl::read_scene_state(&scene, data_);
    EXPECT_NEAR(wheel->velocity, 5.0, 0.05);
    EXPECT_NEAR(pivot->position, 0.0, 1e-6);

    ASSERT_TRUE(mj_kdl::set_control_mode(model_, data_, 0, mj_kdl::CtrlMode::TORQUE));
    const double before = wheel->velocity;
    mj_step(model_, data_);
    mj_kdl::read_scene_state(&scene, data_);
    EXPECT_EQ(data_->actuator_force[actuator("wheel_velocity")], 0.0);
    // One step of coasting on joint damping alone (~21 rad/s^2 here), not a jump.
    EXPECT_NEAR(wheel->velocity, before, 0.1) << "no jump at the switch";
}

TEST_F(MotorWheelModesTest, SceneActuatorFlagsAClampedCommand)
{
    mj_kdl::SceneState scene;
    ASSERT_TRUE(mj_kdl::init_scene_state(&scene, model_));
    mj_kdl::SceneActuatorSlot *motor = mj_kdl::bind_scene_actuator(&scene, "wheel");
    ASSERT_NE(motor, nullptr);

    motor->command = 20.0;
    mj_kdl::apply_scene_state(&scene, data_);
    EXPECT_EQ(data_->ctrl[motor->ctrl_id], 12.0);
    EXPECT_TRUE(motor->saturated);

    motor->command = 5.0;
    mj_kdl::apply_scene_state(&scene, data_);
    EXPECT_FALSE(motor->saturated);
}

TEST_F(MotorWheelModesTest, ForceLimitsFollowTheActiveMode)
{
    mj_kdl::Robot wheel;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&wheel, model_, data_, "drive", "wheel"));
    ASSERT_EQ(wheel.n_joints, 1);
    ASSERT_EQ(wheel.ctrl_mode, mj_kdl::CtrlMode::TORQUE);
    EXPECT_DOUBLE_EQ(mj_kdl::joint_force_limits(&wheel)[0], 12.0) << "the motor's ctrlrange";

    ASSERT_TRUE(mj_kdl::set_control_mode(&wheel, mj_kdl::CtrlMode::VELOCITY));
    EXPECT_DOUBLE_EQ(mj_kdl::joint_force_limits(&wheel)[0], 12.0) << "the velocity forcerange";

    model_->actuator_forcerange[2 * actuator("wheel_velocity") + 1] = 20.0;
    EXPECT_DOUBLE_EQ(mj_kdl::joint_force_limits(&wheel)[0], 20.0);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
