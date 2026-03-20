# mj-kdl-wrapper

A C++ library bridging [MuJoCo 3.6](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

## Preview

![Kinova GEN3 with Robotiq 2F-85 gripper](docs/gen3_with_gripper.png)

*Kinova GEN3 7-DOF arm with Robotiq 2F-85 gripper — loaded directly from MuJoCo Menagerie MJCF.*

![Robot arm on table with objects](docs/table_scene_screenshot.png)

*7-DOF arm on a table with pickable cubes and spheres — KDL gravity compensation.*

## Features

- **MJCF loading** — load MuJoCo models directly with `load_mjcf` + `init_robot_from_mjcf`; no URDF needed
- **URDF loading** — converts URDF to MJCF, auto-injects scene elements (floor, lights)
- **KDL chain** — builds a KDL chain from URDF or MJCF; FK, IK, Jacobian, dynamics
- **Gripper attachment** — combine arm + gripper MJCF with `attach_gripper`; handles name prefixing, mesh paths, connect constraints
- **Multi-robot scenes** — place multiple robots in one shared simulation via `SceneSpec`/`build_scene_from_urdfs`
- **Table + objects** — parametric table, boxes, spheres, cylinders; add/remove objects at runtime
- **Interactive viewer** — GLFW window with joint value overlay, pause/resume, body selection for force perturbation

## Dependencies

| Dependency | Version | Install |
|------------|---------|---------|
| MuJoCo | 3.6.0 | download to `/opt/mujoco-3.6.0` |
| GLFW | 3.x | `sudo apt install libglfw3-dev` |
| OpenGL | — | `sudo apt install libgl-dev` |
| orocos-kdl | — | `sudo apt install liborocos-kdl-dev` (or build from source, see below) |
| urdfdom | — | `sudo apt install liburdfdom-dev` |
| TinyXML2 | — | `sudo apt install libtinyxml2-dev` |

`kdl_parser` is bundled under `third_party/` — no separate install needed.

### orocos KDL from source (optional)

If the packaged `liborocos-kdl-dev` is unavailable or you need a specific version,
build and install KDL locally:

```bash
# 1. Clone
git clone https://github.com/secorolab/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics

# 2. Build and install to a local prefix (e.g. ~/ws/install)
cmake orocos_kdl \
      -B build_kdl \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=~/ws/install \
      -DENABLE_TESTS=OFF \
      -DENABLE_EXAMPLES=OFF
cmake --build build_kdl -j$(nproc)
cmake --install build_kdl
```

Then point this project's build at that prefix:

```bash
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
         -DCMAKE_PREFIX_PATH=~/ws/install
```

> **Note:** orocos_kdl registers itself in the CMake user package registry
> (`~/.cmake/packages/orocos_kdl/`) when built, so cmake may pick up a build
> from another project on your machine even without `CMAKE_PREFIX_PATH`.
> To force a specific install, add `-DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON`
> alongside `CMAKE_PREFIX_PATH`.

## Building

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/3.6.0/mujoco-3.6.0-linux-x86_64.tar.gz
tar -xzf mujoco-3.6.0-linux-x86_64.tar.gz -C /opt/

sudo apt install libglfw3-dev libgl-dev liborocos-kdl-dev liburdfdom-dev libtinyxml2-dev

mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

## API

### Load from MJCF (MuJoCo Menagerie format)

```cpp
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

mjModel *model = nullptr; mjData *data = nullptr;
mj_kdl::load_mjcf(&model, &data, "third_party/menagerie/kinova_gen3/scene.xml");

mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link");

unsigned n = robot.n_joints;
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), g(n);

robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
mj_kdl::run_simulate_ui(model, data, "", [&](mjModel *, mjData *) {
    mj_kdl::update(&robot);                          // read msr, apply cmd
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
});

mj_kdl::cleanup(&robot);
mj_kdl::destroy_scene(model, data);
```

### Attach a gripper to an arm

```cpp
mj_kdl::GripperSpec gs;
gs.mjcf_path = "third_party/menagerie/robotiq_2f85/2f85.xml";
gs.attach_to = "bracelet_link";
gs.prefix    = "g_";
gs.pos[2]    = -0.061525;  // offset along bracelet_link -Z
gs.euler[0]  = 180.0;      // 180 deg around X, gripper faces down [degrees]
gs.add_skybox = true;
gs.add_floor  = true;
gs.contact_exclusions = { { "bracelet_link", "g_base" }, { "bracelet_link", "g_left_pad" } };

mj_kdl::attach_gripper("third_party/menagerie/kinova_gen3/gen3.xml",
                        &gs, "/tmp/gen3_with_2f85.xml");

mjModel *model = nullptr; mjData *data = nullptr;
mj_kdl::load_mjcf(&model, &data, "/tmp/gen3_with_2f85.xml");

mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link");
```

### Load from URDF

```cpp
mj_kdl::SceneSpec sc;
mj_kdl::RobotSpec rs;
rs.path = "assets/gen3_urdf/GEN3_URDF_V12.urdf";
sc.robots.push_back(rs);

mjModel *model = nullptr; mjData *data = nullptr;
mj_kdl::build_scene_from_urdfs(&model, &data, &sc);

mj_kdl::Robot robot;
mj_kdl::init_robot_from_urdf(&robot, model, data,
    "assets/gen3_urdf/GEN3_URDF_V12.urdf", "base_link", "EndEffector_Link");

unsigned n = robot.n_joints;
KDL::JntArray q_home(n);
// fill q_home ...
mj_kdl::set_joint_pos(&robot, q_home); // writes qpos and calls mj_forward

mj_kdl::cleanup(&robot);
mj_kdl::destroy_scene(model, data);
```

### Robot on a table with objects

```cpp
mj_kdl::SceneSpec sc;
sc.table.enabled = true;
sc.table.pos[2]  = 0.7;

mj_kdl::RobotSpec rs;
rs.path   = "assets/gen3_urdf/GEN3_URDF_V12.urdf";
rs.pos[2] = 0.7;
sc.robots.push_back(rs);

mj_kdl::SceneObject cube;
cube.name    = "red_cube";
cube.shape   = mj_kdl::Shape::BOX;
cube.size[0] = cube.size[1] = cube.size[2] = 0.03;
cube.pos[0]  = 0.35; cube.pos[1] = 0.1; cube.pos[2] = 0.73;
cube.rgba[0] = 1.0f; cube.rgba[1] = 0.2f; cube.rgba[2] = 0.2f; cube.rgba[3] = 1.0f;
sc.objects.push_back(cube);

mjModel *model = nullptr; mjData *data = nullptr;
mj_kdl::build_scene_from_urdfs(&model, &data, &sc);

mj_kdl::Robot robot;
mj_kdl::init_robot_from_urdf(&robot, model, data,
    "assets/gen3_urdf/GEN3_URDF_V12.urdf", "base_link", "EndEffector_Link");
```

### Runtime add / remove objects

```cpp
mj_kdl::scene_add_object(&model, &data, &sc, obj);
mj_kdl::scene_remove_object(&model, &data, &sc, "red_cube");
// model/data are replaced; re-call init_robot_from_urdf() afterwards
```

## Viewer

When a body is selected, its name is shown in the top-left corner of the viewport.

### Controls

| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| **Double-click body** | **Select body for perturbation** |
| **Left drag** (selected) | **Apply translational force** |
| **Right drag** (selected) | **Apply torque** |
| `D` | Deselect body |
| `Space` | Pause / resume |
| `R` | Reset simulation |
| `Q` / `Esc` | Quit |

`Robot::paused` can also be set programmatically.

## Tests

### Overview

| Binary | What it tests |
|--------|---------------|
| `test_init` | URDF load, DOF count, 100-step simulation advance |
| `test_velocity` | FK at home pose; IK round-trip (NR_JL) < 0.01 mm; Jacobian shape |
| `test_gravity_comp` | KDL gravity torques vs MuJoCo `qfrc_bias` at q=0; EE drift < 1 mm after 500 steps with gravity comp |
| `test_dual_arm` | Two Gen3 arms in one shared scene; independent KDL states; EE drift < 0.1 mm per arm |
| `test_table_scene` | Robot + table + box/sphere objects; gravity comp drift; runtime add/remove of scene objects |
| `test_kinova_gen3_menagerie` | MJCF load from `third_party/menagerie/kinova_gen3/`; KDL chain has 7 joints; KDL gravity accuracy vs MuJoCo |
| `test_kinova_gen3_gripper` | `attach_gripper()` combines arm and Robotiq 2F-85; validates combined nq/nbody; KDL chain for arm only; gripper driver joint range |
| `test_kinova_gen3_impedance` | Gravity-comp impedance hold at home pose; EE drift < 1 mm over 200 simulation steps |
| `test_kinova_gen3_pick` | Full pick-and-place pipeline: IK for pre-grasp/grasp/lift (< 2 mm error); 10.5 s headless physics simulation; cube lifted > 0.20 m |

### Running

```bash
cd build
ctest --output-on-failure

# Or run individual binaries directly (all headless tests also accept --gui):
./test_gravity_comp --gui
./test_kinova_gen3_impedance --gui
./test_kinova_gen3_pick --gui
```

## Assets

| Path | Description |
|------|-------------|
| `third_party/menagerie/kinova_gen3/gen3.xml` | Kinova GEN3 7-DOF arm (MuJoCo Menagerie) |
| `third_party/menagerie/robotiq_2f85/2f85.xml` | Robotiq 2F-85 gripper (MuJoCo Menagerie) |
| `assets/gen3_urdf/GEN3_URDF_V12.urdf` | Kinova GEN3 URDF (used by URDF-loading tests) |

## Regenerate screenshots

```bash
MUJOCO_GL=egl python3 tools/render_gripper_scene.py
MUJOCO_GL=egl python3 tools/render_table_scene.py
```
