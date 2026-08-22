/* test_camera_ros.cpp
 * The CameraInfo a sim camera publishes, and the gate that keeps an unwatched topic free.
 * Built only when the camera_ros target is (MJ_KDL_WITH_ROS). */

#include "mj_kdl_wrapper/camera_ros.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

// 640x480 at fovy 45 deg: f = 480 / (2 * tan(22.5 deg)) = 579.4112549695428.
static constexpr const char *kMjcf = R"(<mujoco>
  <worldbody>
    <camera name="wrist" fovy="45" pos="0 0 1"/>
  </worldbody>
</mujoco>)";

class CameraRosTest : public testing::Test
{
  protected:
    mjModel                *model_ = nullptr;
    rclcpp::Node::SharedPtr node_;
    mj_kdl::CameraConf      conf_;

    void SetUp() override
    {
        const auto path = fs::temp_directory_path() / "mj_kdl_camera_ros_test.xml";
        std::ofstream(path) << kMjcf;

        char error[1024] = { 0 };
        model_           = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
        fs::remove(path);
        ASSERT_NE(model_, nullptr) << error;

        node_ = std::make_shared<rclcpp::Node>("mj_kdl_camera_ros_test");

        conf_.camera   = "wrist";
        conf_.frame_id = "wrist_optical";
        conf_.topic_ns = "wrist";
        conf_.width    = 640;
        conf_.height   = 480;
        conf_.rate_hz  = 30.0;
    }

    void TearDown() override
    {
        node_.reset();
        if (model_) mj_deleteModel(model_);
    }
};

TEST_F(CameraRosTest, IntrinsicsComeFromTheModelFovy)
{
    mj_kdl::CameraRosPublisher pub(*node_, model_, conf_);
    const auto                &k = pub.camera_info().k;

    EXPECT_NEAR(k[0], 579.4112549695428, 1e-9);
    EXPECT_NEAR(k[4], 579.4112549695428, 1e-9);
    EXPECT_DOUBLE_EQ(k[2], 320.0);
    EXPECT_DOUBLE_EQ(k[5], 240.0);
    EXPECT_DOUBLE_EQ(k[8], 1.0);
    EXPECT_EQ(pub.camera_info().width, 640u);
    EXPECT_EQ(pub.camera_info().height, 480u);
    EXPECT_EQ(pub.camera_info().header.frame_id, "wrist_optical");
}

TEST_F(CameraRosTest, NobodyWatchingCostsNothing)
{
    mj_kdl::CameraRosPublisher pub(*node_, model_, conf_);

    // Due by rate at both times; with no subscriber neither asks for a frame.
    EXPECT_FALSE(pub.wants_frame(0.0));
    EXPECT_FALSE(pub.wants_frame(10.0));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    const int result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
