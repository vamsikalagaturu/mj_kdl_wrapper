// Copyright (c) 2008, Willow Garage, Inc.
// SPDX-License-Identifier: BSD-3-Clause
// Source: https://github.com/ros/kdl_parser (jazzy branch)
// Modified: standalone build — uses urdfdom directly (no ROS urdf package).

#include "kdl_parser/kdl_parser.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

#include "kdl/frames_io.hpp"
#include "urdf_model/model.h"
#include "urdf_model/joint.h"
#include "urdf_model/link.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

namespace kdl_parser {

KDL::Vector toKdl(urdf::Vector3 v) { return KDL::Vector(v.x, v.y, v.z); }

KDL::Rotation toKdl(urdf::Rotation r) { return KDL::Rotation::Quaternion(r.x, r.y, r.z, r.w); }

KDL::Frame toKdl(urdf::Pose p) { return KDL::Frame(toKdl(p.rotation), toKdl(p.position)); }

KDL::Joint toKdl(urdf::JointSharedPtr jnt)
{
    KDL::Frame F = toKdl(jnt->parent_to_joint_origin_transform);
    switch (jnt->type) {
    case urdf::Joint::FIXED:
        return KDL::Joint(jnt->name, KDL::Joint::None);
    case urdf::Joint::REVOLUTE:
    case urdf::Joint::CONTINUOUS: {
        KDL::Vector axis = toKdl(jnt->axis);
        return KDL::Joint(jnt->name, F.p, F.M * axis, KDL::Joint::RotAxis);
    }
    case urdf::Joint::PRISMATIC: {
        KDL::Vector axis = toKdl(jnt->axis);
        return KDL::Joint(jnt->name, F.p, F.M * axis, KDL::Joint::TransAxis);
    }
    default:
        std::cerr << "[kdl_parser] Unknown joint type for '" << jnt->name << "', using Fixed\n";
        return KDL::Joint(jnt->name, KDL::Joint::None);
    }
}

KDL::RigidBodyInertia toKdl(urdf::InertialSharedPtr i)
{
    KDL::Frame             origin = toKdl(i->origin);
    KDL::RotationalInertia urdf_inertia(i->ixx, i->iyy, i->izz, i->ixy, i->ixz, i->iyz);
    KDL::RigidBodyInertia  tmp =
      origin.M * KDL::RigidBodyInertia(0, KDL::Vector::Zero(), urdf_inertia);
    return KDL::RigidBodyInertia(i->mass, origin.p, tmp.getRotationalInertia());
}

static bool addChildrenToTree(urdf::LinkConstSharedPtr root, KDL::Tree &tree)
{
    KDL::RigidBodyInertia inert(0);
    if (root->inertial) inert = toKdl(root->inertial);
    KDL::Joint   jnt = toKdl(root->parent_joint);
    KDL::Segment sgm(
      root->name, jnt, toKdl(root->parent_joint->parent_to_joint_origin_transform), inert);
    tree.addSegment(sgm, root->parent_joint->parent_link_name);
    for (auto &child : root->child_links) {
        if (!addChildrenToTree(child, tree)) return false;
    }
    return true;
}

bool treeFromFile(const std::string &file, KDL::Tree &tree)
{
    auto model = urdf::parseURDFFile(file);
    if (!model) {
        std::cerr << "[kdl_parser] parseURDFFile failed: " << file << "\n";
        return false;
    }
    return treeFromUrdfModel(*model, tree);
}

bool treeFromString(const std::string &xml, KDL::Tree &tree)
{
    auto model = urdf::parseURDF(xml);
    if (!model) {
        std::cerr << "[kdl_parser] parseURDF failed\n";
        return false;
    }
    return treeFromUrdfModel(*model, tree);
}

bool treeFromUrdfModel(const urdf::ModelInterface &robot_model, KDL::Tree &tree)
{
    if (!robot_model.getRoot()) return false;
    tree = KDL::Tree(robot_model.getRoot()->name);
    for (auto &child : robot_model.getRoot()->child_links) {
        if (!addChildrenToTree(child, tree)) return false;
    }
    return true;
}

} // namespace kdl_parser
