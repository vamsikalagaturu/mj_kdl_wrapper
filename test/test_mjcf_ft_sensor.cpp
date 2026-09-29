/* test_mjcf_ft_sensor.cpp
 * A named F/T sensor between the Kinova GEN3 wrist and a Robotiq 2F-85: resolved from its
 * <force>/<torque> sensors, read by update() and reset(), and refused when incomplete. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

class MjcfFtSensorTest : public testing::Test
{
  protected:
    mj_kdl::Env env_;

    void SetUp() override
    {
        const std::string arm     = ex::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string ft      = ex::find_asset("ft_sensor.xml");
        const std::string gripper = ex::find_asset("robotiq_2f85/2f85.xml");
        if (!fs::exists(arm)) GTEST_SKIP() << arm << " not found";
        if (!fs::exists(ft)) GTEST_SKIP() << ft << " not found";
        if (!fs::exists(gripper)) GTEST_SKIP() << gripper << " not found";

        mj_kdl::AttachmentSpec ft_spec;
        ft_spec.mjcf_path = ft;
        ft_spec.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
        mj_kdl::RobotSpec rs;
        rs.path = arm;
        rs.attachments.push_back(ft_spec);
        rs.attachments.push_back(ex::gripper_attachment(gripper, "wrist_ft_site"));

        mj_kdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = false;
        sc.robots.push_back(rs);
        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
    }

    static mj_kdl::ToolFrameSpec tool_with(const mj_kdl::ForceTorqueSensorSpec &ft)
    {
        mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
        tool.ft_sensors.push_back(ft);
        return tool;
    }

    static mj_kdl::ForceTorqueSensorSpec wrist_ft()
    {
        mj_kdl::ForceTorqueSensorSpec ft;
        ft.name       = "wrist_ft";
        ft.frame_site = "wrist_ft_site";
        return ft;
    }
};

TEST_F(MjcfFtSensorTest, ReadsNamedWrench)
{
    const mj_kdl::ToolFrameSpec tool = tool_with(wrist_ft());
    mj_kdl::Robot               robot;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env_, "base_link", "bracelet_link", "", &tool)
    );
    ASSERT_EQ(robot.ft_sensors.size(), 1u);
    const mj_kdl::ForceTorqueSensor &sensor = robot.ft_sensors.front();
    EXPECT_EQ(sensor.name, "wrist_ft");
    EXPECT_EQ(sensor.force_sensor, "wrist_ft_force");
    EXPECT_EQ(sensor.torque_sensor, "wrist_ft_torque");
    EXPECT_EQ(sensor.frame_site_id, mj_name2id(env_.model, mjOBJ_SITE, "wrist_ft_site"));

    // Let the arm settle under its position servos, then read.
    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&env_);
        mj_kdl::step(&env_);
    }
    mj_kdl::update(&env_);
    const double *f = env_.data->sensordata + sensor.force_adr;
    const double *t = env_.data->sensordata + sensor.torque_adr;
    EXPECT_EQ(sensor.wrench.force, KDL::Vector(f[0], f[1], f[2]));
    EXPECT_EQ(sensor.wrench.torque, KDL::Vector(t[0], t[1], t[2]));

    // At rest it carries the weight of everything below it (measured to 6e-8 N).
    KDL::Frame world_T_site;
    ASSERT_TRUE(mj_kdl::get_site_frame(&env_, "wrist_ft_site", &world_T_site));
    const KDL::Vector world_f = world_T_site.M * sensor.wrench.force;
    const int         body    = env_.model->site_bodyid[sensor.frame_site_id];
    const double      weight  = env_.model->body_subtreemass[body] * 9.81;
    EXPECT_NEAR(world_f.x(), 0.0, 1e-5);
    EXPECT_NEAR(world_f.y(), 0.0, 1e-5);
    EXPECT_NEAR(world_f.z(), weight, 1e-5);
}

TEST_F(MjcfFtSensorTest, ResetReReadsTheWrench)
{
    mj_kdl::ToolFrameSpec tool = tool_with(wrist_ft());
    tool.tcp_site.clear();
    mj_kdl::Robot robot;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env_, "base_link", "bracelet_link", "", &tool)
    );
    robot.ft_sensors[0].wrench = KDL::Wrench(KDL::Vector(99, 99, 99), KDL::Vector(99, 99, 99));

    mj_kdl::reset(&env_);

    const mj_kdl::ForceTorqueSensor &sensor = robot.ft_sensors[0];
    const double                    *f      = env_.data->sensordata + sensor.force_adr;
    const double                    *t      = env_.data->sensordata + sensor.torque_adr;
    EXPECT_EQ(sensor.wrench.force, KDL::Vector(f[0], f[1], f[2]));
    EXPECT_EQ(sensor.wrench.torque, KDL::Vector(t[0], t[1], t[2]));
}

TEST_F(MjcfFtSensorTest, RejectsMissingTorqueSensor)
{
    mj_kdl::ForceTorqueSensorSpec ft;
    ft.name                          = "bad_ft";
    ft.force_sensor                  = "wrist_ft_force";
    ft.torque_sensor                 = "missing_torque";
    const mj_kdl::ToolFrameSpec tool = tool_with(ft);

    mj_kdl::Robot robot;
    EXPECT_FALSE(
      mj_kdl::init_robot_from_mjcf(&robot, &env_, "base_link", "bracelet_link", "", &tool)
    );
    EXPECT_TRUE(env_.robots.empty()) << "a robot that failed to init is not registered";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
