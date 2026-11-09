/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>

#include <cstdint>

namespace mj_kdl {

/**
 * @ingroup grp_recorder
 * Publishes the simulation time on /clock (rosgraph_msgs/Clock), the topic nodes running with
 * use_sim_time follow, as a ROS simulator does. Part of mj_kdl_wrapper::camera_ros.
 *
 * Like CameraRosPublisher it borrows the caller's node and creates no executor or thread. Call
 * publish() from the thread that steps the Env, once per step or at whatever rate the clock's
 * consumers need; the stamps then match the ones CameraRosPublisher puts on its frames.
 *
 *   ClockRosPublisher clock(*node);
 *   step(&env);
 *   clock.publish(env.data->time);
 */
class ClockRosPublisher
{
  public:
    // "/clock", absolute: a node's namespace must not move the topic use_sim_time listens on.
    explicit ClockRosPublisher(rclcpp::Node &node)
      : pub_(node.create_publisher<rosgraph_msgs::msg::Clock>("/clock", rclcpp::ClockQoS()))
    {}

    /** The simulation time sim_t [s] as the current ROS time. */
    void publish(double sim_t)
    {
        msg_.clock = rclcpp::Time(static_cast<std::int64_t>(sim_t * 1e9));
        pub_->publish(msg_);
    }

  private:
    rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr pub_;
    rosgraph_msgs::msg::Clock                               msg_;
};

} // namespace mj_kdl
