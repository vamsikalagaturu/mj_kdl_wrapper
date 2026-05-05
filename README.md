# mj-kdl-wrapper

A C++ library bridging [MuJoCo 3.8](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

## Screenshots

<table>
<tr>
  <td align="center"><img src="docs/screenshots/ex_init.png" width="380"/><br/><b>ex_init</b> &mdash; Single arm loaded from MJCF</td>
  <td align="center"><img src="docs/screenshots/ex_table_scene.png" width="380"/><br/><b>ex_table_scene</b> &mdash; Arm + table + scene objects</td>
</tr>
<tr>
  <td align="center"><img src="docs/screenshots/ex_pick.png" width="380"/><br/><b>ex_pick</b> &mdash; Pick-and-place with Robotiq 2F-85</td>
  <td align="center"><img src="docs/screenshots/ex_dual_arm.png" width="380"/><br/><b>ex_dual_arm</b> &mdash; Dual arm + grippers</td>
</tr>
</table>

## Features

- **Unified scene builder** -- `build_scene()` accepts MJCF files and builds floor, skybox, table, and objects via `mjSpec` with no intermediate XML files
- **Ordered attachment chains** -- `AttachmentSpec` attaches any MJCF body (mount, FT sensor, gripper, arm on a mobile base) under any named body; chains of arbitrary length are applied in declaration order
- **Multi-robot scenes** -- place multiple robots with independent KDL chains in one shared simulation via `SceneSpec::robots`
- **KDL chain from model** -- `init_robot_from_mjcf()` builds a KDL chain directly from a compiled MuJoCo model
- **Control ports** -- `update()` reads `qpos`/`qvel`/`qfrc_bias` into `*_msr` and applies `*_cmd` in POSITION or TORQUE mode
- **Interactive viewer** -- `init_window_sim()` + `tick()` gives your code the control loop while the MuJoCo simulate UI runs in a background render thread
- **Headless video recording** -- `VideoRecorder` uses EGL offscreen rendering and an ffmpeg pipe to record MP4s without a display

## Dependencies

| Dependency | Version | Install |
|------------|---------|---------|
| MuJoCo | 3.8.0 | download to `/opt/mujoco-3.8.0` |
| GLFW | 3.x | `sudo apt install libglfw3-dev` |
| OpenGL / EGL | -- | `sudo apt install libgl-dev libegl-dev` |
| orocos-kdl | -- | `sudo apt install liborocos-kdl-dev` |
| ffmpeg | -- | `sudo apt install ffmpeg` (for VideoRecorder) |

Only MuJoCo 3.8.0 is supported. CMake checks `mjVERSION_HEADER` and stops at configure time if `MUJOCO_ROOT` points to any other MuJoCo release.

### orocos KDL from source (optional)

```bash
git clone https://github.com/secorolab/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics
cmake orocos_kdl -B build_kdl \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=~/ws/install \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF
cmake --build build_kdl -j$(nproc)
cmake --install build_kdl
```

Then add `-DCMAKE_PREFIX_PATH=~/ws/install` to the build below.

## Building

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/3.8.0/mujoco-3.8.0-linux-x86_64.tar.gz
tar -xzf mujoco-3.8.0-linux-x86_64.tar.gz -C /opt/

sudo apt install libglfw3-dev libgl-dev libegl-dev liborocos-kdl-dev

cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
```

Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `BUILD_RECORDER=ON` | ON | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `FETCH_MENAGERIE=ON` | OFF | Download MuJoCo Menagerie robot models |
| `BUILD_TESTS=ON` | ON | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS=ON` | OFF | Generate Doxygen HTML docs (`cmake --build build --target docs`) |

The simulate UI screenshot button (`S` key) is always enabled; it uses an ffmpeg pipe to write PNGs without any third-party lodepng dependency.

## API

### Load from MJCF

```cpp
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

mj_kdl::SceneSpec sc;
mj_kdl::RobotSpec r;
r.path = "third_party/menagerie/kinova_gen3/gen3.xml";
sc.robots.push_back(r);

mjModel *model = nullptr;
mjData  *data  = nullptr;
mj_kdl::build_scene(&model, &data, &sc);
```

### Init KDL chain

```cpp
mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link");

unsigned n = robot.n_joints;  // 7 for Kinova GEN3
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
```

When a tool (gripper) is attached, pass its root body as `tool_body` so KDL dynamics include the full tool inertia:

```cpp
// Gripper prefix "g_", root body "g_base" lumped into the KDL chain as a fixed segment.
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", "g_base");

// ChainDynParam now accounts for arm + gripper mass.
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), g(n);
// ...
dyn.JntToGravity(q, g);  // correct for arm + gripper
```

### Attach a gripper (or any MJCF body)

```cpp
mj_kdl::AttachmentSpec gs;
gs.mjcf_path = "third_party/menagerie/robotiq_2f85/2f85.xml";
gs.attach_to = "bracelet_link";
gs.prefix    = "g_";
gs.pos[2]    = -0.061525;   // offset along bracelet_link -Z [m]
gs.euler[0]  = 180.0;       // flip 180 deg around X [degrees]
gs.contact_exclusions = {
    {"bracelet_link", "g_base"},
    {"bracelet_link", "g_left_pad"},
    {"bracelet_link", "g_right_pad"},
};

mj_kdl::RobotSpec rs;
rs.path = "third_party/menagerie/kinova_gen3/gen3.xml";
rs.attachments.push_back(gs);

mj_kdl::SceneSpec sc;
sc.robots.push_back(rs);
mj_kdl::build_scene(&model, &data, &sc);
```

Chains are supported: push multiple `AttachmentSpec` entries in order
(mount -> FT sensor -> gripper).

### Multi-robot scene

```cpp
mj_kdl::RobotSpec arm1, arm2;
arm1.path   = "gen3.xml";  arm1.pos[0] = -0.5;
arm2.path   = "gen3.xml";  arm2.pos[0] =  0.5;  arm2.prefix = "r2_";

mj_kdl::SceneSpec sc;
sc.robots.push_back(arm1);
sc.robots.push_back(arm2);
mj_kdl::build_scene(&model, &data, &sc);

mj_kdl::Robot robot1, robot2;
mj_kdl::init_robot_from_mjcf(&robot1, model, data, "base_link", "bracelet_link");
mj_kdl::init_robot_from_mjcf(&robot2, model, data, "base_link", "bracelet_link", "r2_");
```

### Table + objects

```cpp
mj_kdl::SceneSpec sc;
sc.table.enabled     = true;
sc.table.pos[2]      = 0.7;   // surface height [m]
sc.table.top_size[0] = 0.8;
sc.table.top_size[1] = 0.6;

mj_kdl::RobotSpec r;
r.path   = "third_party/menagerie/kinova_gen3/gen3.xml";
r.pos[2] = 0.7;
sc.robots.push_back(r);

mj_kdl::SceneObject cube;
cube.name    = "red_cube";
cube.shape   = mj_kdl::Shape::BOX;
cube.size[0] = cube.size[1] = cube.size[2] = 0.03;
cube.pos[0]  = 0.35;  cube.pos[1] = 0.1;  cube.pos[2] = 0.73;
cube.rgba[0] = 1.0f;  cube.rgba[3] = 1.0f;
sc.objects.push_back(cube);

mj_kdl::build_scene(&model, &data, &sc);
```

### Control loop

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot);

KDL::JntArray q(n), g(n);
while (mj_kdl::tick(&viewer, model, data)) {
    mj_kdl::update(&robot);                          // read *_msr, apply *_cmd
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
}

mj_kdl::cleanup(&viewer);
mj_kdl::cleanup(&robot);
mj_kdl::destroy_scene(model, data);
```

### Headless video recording

```cpp
/* Requires BUILD_RECORDER=ON (default) and ffmpeg in PATH. */
mj_kdl::VideoRecorder vr;
/* Preset resolutions: R360p, R480p, R720p, R1080p, R2K, R4K */
mj_kdl::init_video_recorder(&vr, model, "sim.mp4", mj_kdl::VideoResolution::R1080p);

/* Configure camera (optional - defaults to model-fitted free camera). */
vr.cam.azimuth   = 135.0;
vr.cam.elevation = -20.0;
vr.cam.distance  = 2.5;

for (int i = 0; i < 3000; ++i) {
    mj_kdl::update(&robot);
    // ... apply control ...
    mj_kdl::step(&robot);
    mj_kdl::record_frame(&vr, model, data);
}

mj_kdl::cleanup(&vr);   // flushes pipe, finalises MP4
```

### Runtime add / remove objects

```cpp
mj_kdl::scene_add_object(&model, &data, &sc, cube);
mj_kdl::scene_remove_object(&model, &data, &sc, "red_cube");
/* model/data are replaced; re-call init_robot_from_*() on the new pointers. */
```

## Viewer controls

| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| Double-click body | Select body for perturbation |
| Left drag (selected) | Apply translational force |
| Right drag (selected) | Apply torque |
| `D` | Deselect body |
| `Space` | Pause / resume |

All other controls (reset, quit, rendering flags) are in the MuJoCo left panel.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

See [test/README.md](test/README.md) for the full list of tests.

## Examples

See [src/examples/README.md](src/examples/README.md) for the full list of examples.

## Documentation

```bash
# Install Doxygen first
sudo apt install doxygen

cmake -B build -DBUILD_DOCS=ON
cmake --build build --target docs
# Open build/docs/html/index.html
```

## Assets

| Path | Description |
|------|-------------|
| `third_party/menagerie/kinova_gen3/gen3.xml` | Kinova GEN3 7-DOF arm (MuJoCo Menagerie) |
| `third_party/menagerie/robotiq_2f85/2f85.xml` | Robotiq 2F-85 gripper (MuJoCo Menagerie) |
