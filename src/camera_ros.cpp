/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/camera_ros.hpp"
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <cmath>
#include <cstring>
#include <utility>

namespace mj_kdl {

CameraRosPublisher::CameraRosPublisher(rclcpp::Node &node, const mjModel *model, CameraConf conf)
  : conf_(std::move(conf)), period_s_(conf_.rate_hz > 0.0 ? 1.0 / conf_.rate_hz : 0.0)
{
    image_pub_ = node.create_publisher<sensor_msgs::msg::Image>(
      conf_.topic_ns + "/color", rclcpp::SensorDataQoS()
    );
    info_pub_ = node.create_publisher<sensor_msgs::msg::CameraInfo>(
      conf_.topic_ns + "/camera_info", rclcpp::SensorDataQoS()
    );

    image_.header.frame_id = conf_.frame_id;
    image_.width           = static_cast<std::uint32_t>(conf_.width);
    image_.height          = static_cast<std::uint32_t>(conf_.height);
    image_.encoding        = "rgb8";
    image_.is_bigendian    = 0;
    image_.step            = static_cast<std::uint32_t>(conf_.width) * 3;
    image_.data.resize(static_cast<std::size_t>(image_.step) * conf_.height);

    const int cam_id = mj_name2id(model, mjOBJ_CAMERA, conf_.camera.c_str());
    if (cam_id < 0) {
        LOG_ERROR(
          "camera '" << conf_.camera << "': no such camera in the model; "
                     << "CameraInfo carries MuJoCo's default fovy, not this camera's"
        );
    }
    const double fovy_rad = (cam_id >= 0 ? model->cam_fovy[cam_id] : 45.0) * mjPI / 180.0;
    // MuJoCo's fovy is vertical, and its pixels are square: one focal length for both axes.
    const double f  = conf_.height / (2.0 * std::tan(fovy_rad / 2.0));
    const double cx = conf_.width / 2.0;
    const double cy = conf_.height / 2.0;

    info_.header.frame_id  = conf_.frame_id;
    info_.width            = static_cast<std::uint32_t>(conf_.width);
    info_.height           = static_cast<std::uint32_t>(conf_.height);
    info_.distortion_model = "plumb_bob";
    info_.d.assign(5, 0.0); // a rendered pinhole camera has no lens to distort the image
    info_.k = { f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0 };
    info_.r = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    info_.p = { f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0 };
}

bool CameraRosPublisher::wants_frame(double sim_t) const
{
    if (sim_t < next_due_s_) return false;
    return image_pub_->get_subscription_count() > 0;
}

void CameraRosPublisher::publish(const std::uint8_t *rgb, double sim_t)
{
    if (!rgb) return;

    // Hold the cadence on its own grid. A caller that is itself rate-gated hands each frame over
    // a tick late, and re-basing on arrival would then drop every other one. Re-base only when
    // the clock jumped: a simulation reset, or a stall longer than a period.
    next_due_s_ += period_s_;
    if (next_due_s_ < sim_t || next_due_s_ > sim_t + 2.0 * period_s_)
        next_due_s_ = sim_t + period_s_;

    // render_rgb() already hands the frame over top-down, so the rows go straight across.
    std::memcpy(image_.data.data(), rgb, image_.data.size());

    const auto stamp    = rclcpp::Time(static_cast<std::int64_t>(sim_t * 1e9));
    image_.header.stamp = stamp;
    info_.header.stamp  = stamp;
    image_pub_->publish(image_);
    info_pub_->publish(info_);
}

} // namespace mj_kdl
