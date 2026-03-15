# mj-kdl-wrapper

A C++ library bridging [MuJoCo 3.6](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

## Preview

![Kinova GEN3 with Robotiq 2F-85 gripper](docs/gen3_with_gripper.png)

*Kinova GEN3 7-DOF arm with Robotiq 2F-85 gripper — loaded directly from MuJoCo Menagerie MJCF.*

![Robot arm on table with objects](docs/table_scene_screenshot.png)

*7-DOF arm on a table with pickable cubes and spheres — KDL gravity compensation.*

## Features

- **MJCF loading** — load MuJoCo models directly with `load_mjcf` + `init_from_mjcf`; no URDF needed
- **URDF loading** — converts URDF to MJCF, auto-injects scene elements (floor, lights)
- **KDL chain** — builds a KDL chain from URDF or MJCF; FK, IK, Jacobian, dynamics
- **Gripper attachment** — combine arm + gripper MJCF with `attach_gripper`; handles name prefixing, mesh paths, connect constraints
- **Multi-robot scenes** — place multiple robots in one shared simulation via `SceneSpec`/`build_scene`
- **Table + objects** — parametric table, boxes, spheres, cylinders; add/remove objects at runtime
- **Interactive viewer** — GLFW window with joint value overlay, pause/resume, body selection for force perturbation

## Dependencies

| Dependency | Version | Install |
|------------|---------|---------|
| MuJoCo | 3.6.0 | download to `/opt/mujoco-3.6.0` |
| GLFW | 3.x | `sudo apt install libglfw3-dev` |
| OpenGL | — | `sudo apt install libgl-dev` |
| orocos-kdl | — | `sudo apt install liborocos-kdl-dev` |
| urdfdom | — | `sudo apt install liburdfdom-dev` |
| TinyXML2 | — | `sudo apt install libtinyxml2-dev` |

`kdl_parser` is bundled under `third_party/` — no separate install needed.

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

mjModel* model; mjData* data;
mj_kdl::load_mjcf(&model, &data, "third_party/menagerie/kinova_gen3/gen3.xml");

mj_kdl::State s;
mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link");

while (mj_kdl::is_running(&s)) {
    KDL::JntArray q, g(s.n_joints);
    mj_kdl::sync_to_kdl(&s, q);
    KDL::ChainDynParam dyn(s.chain, KDL::Vector(0, 0, -9.81));
    dyn.JntToGravity(q, g);
    mj_kdl::set_torques(&s, g);
    mj_kdl::step(&s);
    mj_kdl::render(&s);
}
mj_kdl::cleanup(&s);
```

### Attach a gripper to an arm

```cpp
mj_kdl::GripperSpec gripper;
gripper.mjcf_path = "third_party/menagerie/robotiq_2f85/2f85.xml";
gripper.attach_to = "bracelet_link";
gripper.prefix    = "g_";
gripper.pos[2]    = -0.061525;   // offset along bracelet_link -Z
gripper.quat[1]   = 1.0;         // 180° around X → gripper faces down

mj_kdl::attach_gripper("third_party/menagerie/kinova_gen3/gen3.xml",
                        &gripper, "/tmp/gen3_with_2f85.xml");

mjModel* model; mjData* data;
mj_kdl::load_mjcf(&model, &data, "/tmp/gen3_with_2f85.xml");

mj_kdl::State s;
mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link");
```

### Load from URDF

```cpp
mj_kdl::Config cfg;
cfg.urdf_path = "assets/gen3_urdf/GEN3_URDF_V12.urdf";
cfg.base_link = "base_link";
cfg.tip_link  = "bracelet_link";

mj_kdl::State s;
mj_kdl::init(&s, &cfg);

while (mj_kdl::is_running(&s)) {
    KDL::JntArray q, g(s.n_joints);
    mj_kdl::sync_to_kdl(&s, q);
    KDL::ChainDynParam dyn(s.chain, KDL::Vector(0, 0, -9.81));
    dyn.JntToGravity(q, g);
    mj_kdl::set_torques(&s, g);
    mj_kdl::step(&s);
    mj_kdl::render(&s);
}
mj_kdl::cleanup(&s);
```

### Robot on a table with objects

```cpp
mj_kdl::SceneSpec spec;
spec.table.enabled = true;
spec.table.pos[2]  = 0.7;

mj_kdl::SceneRobot robot;
robot.urdf_path = "assets/gen3_urdf/GEN3_URDF_V12.urdf";
robot.pos[2]    = 0.7;
spec.robots.push_back(robot);

mj_kdl::SceneObject cube;
cube.name    = "red_cube";
cube.shape   = mj_kdl::ObjShape::BOX;
cube.size[0] = cube.size[1] = cube.size[2] = 0.03;
cube.pos[0]  = 0.35; cube.pos[1] = 0.1; cube.pos[2] = 0.73;
cube.rgba[0] = 1.0f; cube.rgba[1] = 0.2f; cube.rgba[2] = 0.2f; cube.rgba[3] = 1.0f;
spec.objects.push_back(cube);

mjModel* model; mjData* data;
mj_kdl::build_scene(&model, &data, &spec);

mj_kdl::State s;
mj_kdl::init_robot(&s, model, data, "assets/gen3_urdf/GEN3_URDF_V12.urdf",
                   "base_link", "bracelet_link");
```

### Runtime add / remove objects

```cpp
mj_kdl::scene_add_object(&model, &data, &spec, obj);
mj_kdl::scene_remove_object(&model, &data, &spec, "red_cube");
// model/data are replaced; re-call init_robot() afterwards
```

## Viewer

The viewer shows:
- **Top-left**: simulation time, pause state, and selected body name
- **Top-right**: live joint values (toggle with `J`)

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
| `J` | Toggle joint value overlay |
| `Q` / `Esc` | Quit |

`State::paused` and `State::show_joints` can also be set programmatically.

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

# Core library tests (URDF-based)
./test_init
./test_velocity
./test_gravity_comp
./test_dual_arm
./test_table_scene

# Kinova Gen3 + Menagerie MJCF tests
./test_kinova_gen3_menagerie
./test_kinova_gen3_gripper
./test_kinova_gen3_impedance
./test_kinova_gen3_pick

# GUI (interactive viewer) — all headless tests also accept --gui
./test_gravity_comp --gui
./test_dual_arm --gui
./test_table_scene --gui
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
