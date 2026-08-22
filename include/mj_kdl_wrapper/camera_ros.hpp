/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <cstdint>
#include <string>

namespace mj_kdl {

/**
 * @ingroup grp_recorder
 * What a simulated camera is called on the ROS graph, and how often it is published.
 */
struct CameraConf
{
    std::string camera;   // camera name in the compiled model (a scene prefixes robot cameras)
    std::string frame_id; // must name a ROS-optical frame: +Z forward, +X right, +Y down
    std::string topic_ns; // -> <topic_ns>/color, <topic_ns>/camera_info
    int         width   = 0;
    int         height  = 0;
    double      rate_hz = 30.0;
};

/**
 * @ingroup grp_recorder
 * Publishes an already-rendered RGB frame as sensor_msgs/Image plus its CameraInfo.
 *
 * A frame sink, not a renderer: it creates no node, no executor and no thread, and never
 * touches mjModel or mjData after construction. Call it from the thread that rendered the
 * frame - render once, hand the same buffer to whatever else consumes it.
 *
 * Intrinsics are read from the model's cam_fovy at construction, so the published K is the one
 * MuJoCo rendered with. Pixels are published exactly as rendered: a MuJoCo camera looks along
 * its -Z with +Y up, so `frame_id` must name a frame already rotated into the ROS optical
 * convention, or every pose a consumer estimates comes out rotated.
 *
 *   CameraRosPublisher pub(*node, model, conf);
 *   if (pub.wants_frame(t) && render_rgb(&vr, model, data, rgb.data())) pub.publish(rgb.data(), t);
 */
class CameraRosPublisher
{
  public:
    CameraRosPublisher(rclcpp::Node &node, const mjModel *model, CameraConf conf);

    /** Due by rate, and somebody is subscribed to the image topic. No frame is touched. */
    bool wants_frame(double sim_t) const;

    /** One top-down RGB frame, width * height * 3 bytes, stamped with the sim time it shows. */
    void publish(const std::uint8_t *rgb, double sim_t);

    /** The CameraInfo published alongside every frame; built once, only its stamp changes. */
    const sensor_msgs::msg::CameraInfo &camera_info() const { return info_; }

  private:
    CameraConf                                                 conf_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr      image_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr info_pub_;
    sensor_msgs::msg::Image                                    image_;
    sensor_msgs::msg::CameraInfo                               info_;
    double                                                     period_s_   = 0.0;
    double                                                     next_due_s_ = 0.0;
};

} // namespace mj_kdl
