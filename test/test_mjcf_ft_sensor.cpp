/* test_mjcf_ft_sensor.cpp
 * Logical FT sensor exposure on the Kinova GEN3 default scene. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

class MjcfFtSensorTest : public testing::Test
{
  protected:
    mj_kdl::Env env_;

    std::string arm_mjcf_;
    std::string ft_mjcf_;
    std::string grp_mjcf_;

    void SetUp() override
    {
        arm_mjcf_ = mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        ft_mjcf_  = mj_kdl_examples::find_asset("ft_sensor.xml");
        grp_mjcf_ = mj_kdl_examples::find_asset("robotiq_2f85/2f85.xml");
        if (!fs::exists(arm_mjcf_)) GTEST_SKIP() << arm_mjcf_ << " not found";
        if (!fs::exists(ft_mjcf_)) GTEST_SKIP() << ft_mjcf_ << " not found";
        if (!fs::exists(grp_mjcf_)) GTEST_SKIP() << grp_mjcf_ << " not found";

        mj_kdl::AttachmentSpec ft{
            .mjcf_path          = ft_mjcf_,
            .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
            .prefix             = "",
            .contact_exclusions = {},
        };
        mj_kdl::AttachmentSpec gripper{
            .mjcf_path          = grp_mjcf_,
            .attach_to          = { mj_kdl::AttachKind::Site, "wrist_ft_site" },
            .prefix             = "g_",
            .contact_exclusions = {},
        };

        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf_;
        rs.attachments.push_back(ft);
        rs.attachments.push_back(gripper);

        mj_kdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = true;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
    }
};

TEST_F(MjcfFtSensorTest, ReadsNamedWrench)
{
    mj_kdl::ForceTorqueSensorSpec ft{ .name = "wrist_ft", .frame_site = "wrist_ft_site" };
    mj_kdl::ToolFrameSpec         tool{
                .tool_body  = "g_base_mount",
                .tcp_site   = "g_pinch",
                .ft_sensors = { ft },
    };

    mj_kdl::Robot robot;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env_, "base_link", "bracelet_link", "", &tool)
    );
    ASSERT_EQ(robot.ft_sensors.size(), 1u);
    const mj_kdl::ForceTorqueSensor *sensor = &robot.ft_sensors.front();
    EXPECT_EQ(sensor->name, "wrist_ft");
    EXPECT_EQ(sensor->force_sensor, "wrist_ft_force");
    EXPECT_EQ(sensor->torque_sensor, "wrist_ft_torque");
    EXPECT_GE(sensor->frame_site_id, 0);

    mj_forward(env_.model, env_.data);
    mj_kdl::update(&env_);
    EXPECT_TRUE(std::isfinite(sensor->wrench.force.x()));
    EXPECT_TRUE(std::isfinite(sensor->wrench.force.y()));
    EXPECT_TRUE(std::isfinite(sensor->wrench.force.z()));
    EXPECT_TRUE(std::isfinite(sensor->wrench.torque.x()));
    EXPECT_TRUE(std::isfinite(sensor->wrench.torque.y()));
    EXPECT_TRUE(std::isfinite(sensor->wrench.torque.z()));
}

TEST_F(MjcfFtSensorTest, ResetReReadsTheWrench)
{
    mj_kdl::ForceTorqueSensorSpec ft{ .name = "wrist_ft", .frame_site = "wrist_ft_site" };
    mj_kdl::ToolFrameSpec         tool{ .tool_body = "g_base_mount", .ft_sensors = { ft } };

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
    mj_kdl::ForceTorqueSensorSpec ft{
        .name          = "bad_ft",
        .force_sensor  = "wrist_ft_force",
        .torque_sensor = "missing_torque",
    };
    mj_kdl::ToolFrameSpec tool{
        .tool_body  = "g_base_mount",
        .tcp_site   = "g_pinch",
        .ft_sensors = { ft },
    };

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
