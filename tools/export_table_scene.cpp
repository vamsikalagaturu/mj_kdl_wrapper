// Build the table scene with the real robot and save a binary .mjb model.
// Usage: export_table_scene <robot.urdf> <output.mjb>

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <robot.urdf> <output.mjb>\n";
        return 1;
    }

    double surface_z = 0.7;

    mj_kdl::SceneSpec spec;
    spec.add_floor  = true;
    spec.gravity_z  = -9.81;

    spec.table.enabled      = true;
    spec.table.pos[0]       = 0.0;
    spec.table.pos[1]       = 0.0;
    spec.table.pos[2]       = surface_z;
    spec.table.top_size[0]  = 0.8;
    spec.table.top_size[1]  = 0.6;
    spec.table.thickness    = 0.04;
    spec.table.leg_radius   = 0.03;

    mj_kdl::SceneRobot robot;
    robot.urdf_path = argv[1];
    robot.prefix    = "";
    robot.pos[0]    = 0.0;
    robot.pos[1]    = 0.0;
    robot.pos[2]    = surface_z;
    spec.robots.push_back(robot);

    auto add_box = [&](const char* name, double x, double y,
                       double hx, double hy, double hz,
                       float r, float g, float b) {
        mj_kdl::SceneObject o;
        o.name    = name;
        o.shape   = mj_kdl::ObjShape::BOX;
        o.size[0] = hx; o.size[1] = hy; o.size[2] = hz;
        o.pos[0]  = x;  o.pos[1]  = y;  o.pos[2]  = surface_z + hz;
        o.rgba[0] = r;  o.rgba[1] = g;  o.rgba[2]  = b; o.rgba[3] = 1.0f;
        spec.objects.push_back(o);
    };
    auto add_sphere = [&](const char* name, double x, double y,
                          double radius, float r, float g, float b) {
        mj_kdl::SceneObject o;
        o.name    = name;
        o.shape   = mj_kdl::ObjShape::SPHERE;
        o.size[0] = radius;
        o.pos[0]  = x; o.pos[1] = y; o.pos[2] = surface_z + radius;
        o.rgba[0] = r; o.rgba[1] = g; o.rgba[2] = b; o.rgba[3] = 1.0f;
        spec.objects.push_back(o);
    };

    add_box   ("red_cube",      0.35,  0.10, 0.03, 0.03, 0.03, 1.0f, 0.2f, 0.2f);
    add_box   ("green_cube",    0.35, -0.10, 0.03, 0.03, 0.03, 0.2f, 1.0f, 0.2f);
    add_box   ("blue_cube",     0.35,  0.30, 0.04, 0.04, 0.04, 0.2f, 0.2f, 1.0f);
    add_sphere("orange_sphere",-0.20,  0.20, 0.035,       1.0f, 0.55f, 0.0f);
    add_sphere("purple_sphere",-0.20, -0.20, 0.025,       0.7f, 0.0f,  0.9f);

    mjModel* model = nullptr;
    mjData*  data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &spec)) {
        std::cerr << "build_scene failed\n";
        return 1;
    }

    mj_saveModel(model, argv[2], nullptr, 0);
    std::cout << "Saved " << argv[2] << "  ("
              << model->nbody << " bodies)\n";

    mj_kdl::destroy_scene(model, data);
    return 0;
}
