# Mujoco KDL Wrapper

A C++ library bridging [MuJoCo 3.8](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

## Screenshots

<table>
<tr>
  <td align="center"><img src="docs/screenshots/ex_gravity_comp.png" width="380"/><br/><b>ex_gravity_comp</b> &mdash; Single arm, KDL gravity compensation</td>
  <td align="center"><img src="docs/screenshots/ex_table_scene.png" width="380"/><br/><b>ex_table_scene</b> &mdash; Arm + table + scene objects</td>
</tr>
<tr>
  <td align="center"><img src="docs/screenshots/ex_pick.png" width="380"/><br/><b>ex_pick</b> &mdash; Pick-and-place with Robotiq 2F-85</td>
  <td align="center"><img src="docs/screenshots/ex_dual_arm.png" width="380"/><br/><b>ex_dual_arm</b> &mdash; Dual arm + grippers</td>
</tr>
</table>

## Start Here

For a full step-by-step walkthrough, read [docs/TUTORIAL.md](docs/TUTORIAL.md).
It starts from installation and gradually builds up robot control, attachments,
asset-backed objects, reset hooks, cameras, the Simulate UI, recording, and
multi-robot scenes.

## Features

- **Unified scene builder** -- `build_scene()` accepts MJCF files and builds floor, skybox, primitive objects, asset-backed objects, and cameras via `mjSpec` with no intermediate XML files
- **Ordered attachment chains** -- `AttachmentSpec` attaches any MJCF body (mount, FT sensor, gripper, arm on a mobile base) under any named body, site, or frame via a tagged `AttachTarget`; chains of arbitrary length are applied in declaration order
- **Relative placement everywhere** -- `RobotSpec.attach_to` and `SceneObject.attach_to` accept the same tagged `AttachTarget`, so a robot can be mounted on a site exported by a prior scene object (e.g. a tabletop site) without hand-threading world-frame heights
- **Multi-robot scenes** -- place multiple robots with independent KDL chains in one shared simulation via `SceneSpec::robots`
- **Runtime environments** -- `Env` owns model/data, registered robots, and reset hooks for task-specific object/controller state
- **KDL chain from model** -- `init_robot_from_mjcf()` builds a KDL chain directly from a compiled MuJoCo model
- **Control ports** -- `update()` reads `qpos`/`qvel`/`qfrc_actuator` into `*_msr` and applies `*_cmd` in POSITION or TORQUE mode
- **Dynamics probes** -- `test/urdf_solver_probe.cpp` uses the bundled Kinova GEN3 URDF to check ACHD fixed-joint outputs and compare URDF-vs-MuJoCo RNEA torques
- **Interactive viewer** -- `init_window_sim()` + `step()` gives your code the control loop while the MuJoCo simulate UI runs in a background render thread
- **Interactive and headless recording** -- Simulate UI recorder controls plus `VideoRecorder` for EGL offscreen MP4 recording

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

git clone https://github.com/secorolab/mj_kdl_wrapper.git
cd mj_kdl_wrapper

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

The repo also carries `third_party/kinova/GEN3_URDF_V12.urdf` for KDL parser
diagnostics.  The MuJoCo model remains sourced from Menagerie.

The simulate UI screenshot button (`S` key) is always enabled; it uses an ffmpeg pipe to write PNGs without any third-party lodepng dependency.

The local Simulate UI also includes wrapper-specific controls in the Simulation
panel:

- `RTF` shows the wrapper real-time factor controlled by `,` and `.`.
- `Recorder` lets you select output path, recording camera, resolution, FPS,
  and start/stop recording.

## API

### Load from MJCF

`SceneSpec` has no defaults for `timestep`, `add_floor`, or `add_skybox`:
those are choices, not values the library can guess. `build_scene` rejects
`timestep <= 0` at runtime.

```cpp
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;   // [s]; required, must be > 0
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(mj_kdl::RobotSpec{ .path = "third_party/menagerie/kinova_gen3/gen3.xml" });

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

When a tool (gripper) is attached, pass a `ToolFrameSpec` so KDL dynamics include the full tool inertia and FK uses the TCP site:

```cpp
const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", &tool);

// ChainDynParam now accounts for arm + gripper mass.
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), g(n);
dyn.JntToGravity(q, g);  // correct for arm + gripper
```

### Attach a gripper (or any MJCF body)

`AttachTarget` is a tagged pair of `AttachKind { World, Body, Site, Frame }`
and an element name. The Kinova GEN3 MJCF exports `pinch_site` on the
bracelet, which already encodes the tool offset and flip, so a gripper
attaches with no manual `pos`/`euler`:

```cpp
mj_kdl::AttachmentSpec gs{
    .mjcf_path          = "third_party/menagerie/robotiq_2f85/2f85.xml",
    .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
    .prefix             = "g_",
    .contact_exclusions = {},
};

mj_kdl::RobotSpec rs;
rs.path = "third_party/menagerie/kinova_gen3/gen3.xml";
rs.attachments.push_back(gs);

mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(rs);
mj_kdl::build_scene(&model, &data, &sc);
```

Optional `pos`/`euler` on the spec are **composed** with the site's pose, so
you can still nudge:

```cpp
gs.pos[2]   = 0.005;   // +5 mm along the tool z
gs.euler[2] = 15.0;    // +15 deg about the tool z
```

If a model has no suitable site, attach by body name instead:

```cpp
gs.attach_to = { mj_kdl::AttachKind::Body, "bracelet_link" };
gs.pos[2]    = -0.061525;
gs.euler[0]  = 180.0;
```

Chains are supported: push multiple `AttachmentSpec` entries in order
(mount -> FT sensor -> gripper). Each entry's `attach_to` may reference
any body, site, or frame added by prior entries.

### Multi-robot scene

```cpp
mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots = {
    mj_kdl::RobotSpec{ .path = "gen3.xml", .pos = { -0.5, 0.0, 0.0 } },
    mj_kdl::RobotSpec{ .path = "gen3.xml", .prefix = "r2_", .pos = { 0.5, 0.0, 0.0 } },
};
mj_kdl::build_scene(&model, &data, &sc);

mj_kdl::Robot robot1, robot2;
mj_kdl::init_robot_from_mjcf(&robot1, model, data, "base_link", "bracelet_link");
mj_kdl::init_robot_from_mjcf(&robot2, model, data, "base_link", "bracelet_link", "r2_");
```

### Table + objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so the robot
follows the table's tabletop site without hand-threaded heights. Build
order in `build_scene` is decorations -> objects (declaration order) ->
robots -> cameras, so a robot's `attach_to` can reference any prior object
and a child object's `attach_to` can reference any earlier object in
`SceneSpec::objects`.

`SceneObject` has no defaults for `shape`, `size`, `rgba`, `mass`, or
`friction`. For MJCF-backed objects (when `mjcf_path` is set) those fields
are ignored at runtime; for primitives, `build_scene` runs explicit checks:

- `shape == Shape::Unspecified` -> error, object skipped.
- `size[i] <= 0` for the relevant dimensions of the shape -> error, skipped.
- `mass <= 0` on a non-fixed primitive -> error, skipped.

```cpp
mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;

mj_kdl::SceneObject table{
    .name      = "table",
    .mjcf_path = "src/examples/assets/table.xml",  // ships a `table_top` site
    .pos       = { 0.0, 0.0, 0.7 },                // asset origin = tabletop center
    .fixed     = true,
};
sc.objects.push_back(table);

// scene_object_site_name() yields the compiled site name (prefixed by
// obj.name + "_"), here "table_table_top".
std::string mount = mj_kdl::scene_object_site_name(table, "table_top");

sc.robots.push_back(mj_kdl::RobotSpec{
    .path      = "third_party/menagerie/kinova_gen3/gen3.xml",
    .attach_to = { mj_kdl::AttachKind::Site, mount.c_str() },
});

// MJCF-asset SceneObjects expose their root body under obj.name in the
// compiled scene, so fixed objects can attach to it directly:
sc.objects.push_back(mj_kdl::SceneObject{
    .name      = "fixture",
    .mjcf_path = "fixture.xml",
    .attach_to = { mj_kdl::AttachKind::Body, "table" },
    .pos       = { 0.0, 0.0, 0.0 },
    .fixed     = true,
});

// MuJoCo restricts freejoints to top-level bodies, so a non-fixed primitive
// (with a freejoint) must stay world-anchored. Primitives require shape,
// size, rgba, mass, and friction.
sc.objects.push_back(mj_kdl::SceneObject{
    .name     = "red_cube",
    .shape    = mj_kdl::Shape::BOX,
    .size     = { 0.03, 0.03, 0.03 },                 // half-extents [m]
    .pos      = { 0.35, 0.10, 0.73 },                 // world frame; tabletop_z + half_height
    .rgba     = { 1.0f, 0.0f, 0.0f, 1.0f },
    .mass     = 0.1,                                  // [kg]
    .condim   = mj_kdl::Condim::Torsional,            // friction model (Tangential/Torsional/Rolling)
    .friction = { 0.8, 0.02, 0.001 },                 // [slide, spin, roll]
});

mj_kdl::build_scene(&model, &data, &sc);
```

### Control loop

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot);

KDL::JntArray q(n), g(n);
while (mj_kdl::step(&robot)) {                       // returns false when window closes
    mj_kdl::update(&robot);                          // read *_msr, apply *_cmd
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
}

mj_kdl::cleanup(&viewer);
mj_kdl::cleanup(&robot);
mj_kdl::destroy_scene(model, data);
```

### Reset

`reset(Env*)` resets the environment runtime to its initial keyframe, calls an
optional environment reset hook, re-seeds all registered robots' command ports to
the current measured state, and clears stale robot forces.  Use the hook to put
objects, controllers, and task state back at their episode start values:

```cpp
mj_kdl::Env env;
mj_kdl::init_env(&env, &scene);

mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, env.model, env.data, "base_link", "bracelet_link");
mj_kdl::env_add_robot(&env, &robot);

env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    // Restore robot/object/task state after mj_resetData, before mj_forward.
    mj_kdl::set_joint_pos(&robot, q_home, false);
    episode_step = 0;
};

mj_kdl::ResetOptions opts;
opts.keyframe = 0;
mj_kdl::ResetInfo info = mj_kdl::reset(&env, &opts);
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

Interactive recording is available from the Simulate UI:

1. Open the left Simulation panel.
2. Scroll to Recorder.
3. Set `Path`, `Camera`, `Resolution`, and `FPS`.
4. Press `Start rec`.
5. Press `Stop rec`.

The recorder camera list includes `Current`, `Free`, `Tracking`, robot MJCF
cameras, and cameras added through `SceneSpec::cameras`.  When ffmpeg closes
successfully the terminal prints:

```text
[mj_kdl] recording saved to <filename>
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
| `,` | Decrease wrapper real-time factor |
| `.` | Increase wrapper real-time factor |

All other controls (reset, quit, rendering flags, live camera selection, and
recording) are in the MuJoCo panels.

## More Documentation

- [Full tutorial](docs/TUTORIAL.md)
- [Examples guide](src/examples/README.md)
- [Torque control notes](docs/HOWTO_torque_control.md)
- [URDF to MJCF notes](docs/HOWTO_urdf_to_mjcf.md)

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
