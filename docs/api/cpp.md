# C++ API Guide {#page_api_cpp}

This page collects the C++ wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated Doxygen API pages
for `include/mj_kdl_wrapper/mj_kdl_wrapper.hpp`.

Coming from 0.2.x? Placement orientation moved from `euler` to `quat`
`[x, y, z, w]`; see [Migrating from 0.2.x](@ref sec_migrate_quat).

## Resolving Models And Assets

The examples and tests resolve paths through `example_paths.hpp` (a header-only
helper under `src/examples/`) so no checkout location is hard-coded. It mirrors
the Python `menagerie` resolver:

- `mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml")` returns a MuJoCo
  Menagerie model. It checks `$MJ_KDL_MENAGERIE` first, then the user cache
  `~/.cache/mj_kdl_wrapper/menagerie`. It throws
  with a fetch hint when absent; `find_menagerie_model(...)` returns `""`
  instead, which is how tests self-skip.
- `mj_kdl_examples::asset("table.xml")` returns a bundled asset from the user
  cache `~/.cache/mj_kdl_wrapper/assets`; `find_asset(...)` returns `""`.

Populate the cache with `cmake -DMJ_KDL_FETCH_MENAGERIE=ON` (it clones Menagerie
and copies the bundled assets into the cache) or the `mj-kdl-fetch-menagerie`
console script. The same cache backs both the C++ and Python examples.

**Overrides:** export `MJ_KDL_MENAGERIE=/path/to/menagerie` to resolve models
from a checkout outside the cache. The C++ helper has no per-asset override --
assets resolve from the cache only. (The Python examples additionally honor
per-file overrides such as `MJ_KDL_MODEL` and `MJ_KDL_GRIPPER`; see the Python
guide.)

## Load From MJCF

`SceneSpec` has no defaults for `timestep`, `add_floor`, or `add_skybox`.
Those are choices, not values the library can guess. `build_scene()` rejects
`timestep <= 0` at runtime. `SceneSpec::robots` may be empty; object-only
scenes are valid.

```cpp
#include "example_paths.hpp"
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;   // [s]; required, must be > 0
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(mj_kdl::RobotSpec{
    .path = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml")
});

mjModel *model = nullptr;
mjData  *data  = nullptr;
mj_kdl::build_scene(&model, &data, &sc);
```

For an object-only scene, add MJCF or primitive `SceneObject` entries and leave
`sc.robots` empty:

```cpp
mj_kdl::SceneObject cabinet{
    .name      = "cabinet",
    .mjcf_path = mj_kdl_examples::asset("cabinet/cabinet.xml"),
    .fixed     = true,
};
sc.objects.push_back(cabinet);
mj_kdl::build_scene(&model, &data, &sc);
```

`save_model_xml(model, path)` writes the most recently compiled model back to
MJCF. Use it when you want to build a combined scene once and reload the merged
model later through MuJoCo. `destroy_scene(model, data)` frees the pair returned
by `build_scene()`.

```cpp
mj_kdl::save_model_xml(model, "combined_scene.xml");
mj_saveModel(model, "combined_scene.mjb", nullptr, 0);
```

Set wrapper log verbosity globally when debugging scene construction:

```cpp
mj_kdl::set_log_level(mj_kdl::LogLevel::INFO);
```

The raw `mjSpec` helpers (`add_skybox_to_spec()`, `add_floor_to_spec()`,
`add_objects_to_spec()`, `compile_and_make_data()`, and
`ensure_plugins_loaded()`) exist for advanced callers that build MuJoCo specs
directly. Most users should go through `SceneSpec` and `build_scene()` so
plugins, decorations, objects, robots, cameras, compilation, and ownership all
follow the same path.

## Init A KDL Chain

```cpp
mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link");

unsigned n = robot.n_joints;  // 7 for Kinova GEN3
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
```

`Robot` borrows `model` and `data`; it never frees them. After init, the port
vectors are sized to `n_joints` and ordered like the KDL chain. Use
`joint_names` and `joint_limits` to inspect that mapping before writing
controllers.

When a tool or gripper is attached, pass a `ToolFrameSpec` so KDL dynamics
include the full tool inertia and FK uses the TCP site:

```cpp
const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
mj_kdl::init_robot_from_mjcf(
    &robot, model, data, "base_link", "bracelet_link", "", &tool);

KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), g(n);
dyn.JntToGravity(q, g);
```

For a wrist force-torque sensor, attach the sensor MJCF first, then attach the
gripper to a site exported by that sensor asset:

```cpp
mj_kdl::AttachmentSpec ft_sensor{
    .mjcf_path = mj_kdl_examples::asset("ft_sensor.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, "pinch_site" },
};

mj_kdl::AttachmentSpec gripper{
    .mjcf_path = mj_kdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, "wrist_ft_site" },
    .prefix    = "g_",
};

mj_kdl::RobotSpec robot_spec;
robot_spec.path = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
robot_spec.attachments = { ft_sensor, gripper };
```

Then register the logical force-torque sensor through `ToolFrameSpec`. MuJoCo
stores it as separate `<force>` and `<torque>` sensors; the wrapper combines one
pair into a `KDL::Wrench`.

```cpp
mj_kdl::ForceTorqueSensorSpec ft{ .name = "wrist_ft", .frame_site = "wrist_ft_site" };
mj_kdl::ToolFrameSpec tool{
    .tool_body  = "g_base",
    .tcp_site   = "g_pinch",
    .ft_sensors = { ft },
};

mj_kdl::update(&robot);
const auto *sensor = mj_kdl::find_ft_sensor(&robot, "wrist_ft");
KDL::Wrench wrench = sensor ? sensor->wrench : KDL::Wrench::Zero();
```

When `force_sensor` and `torque_sensor` are omitted, the wrapper resolves
`{name}_force` and `{name}_torque`.

## Attach MJCF Bodies

`AttachTarget` is a tagged pair of `AttachKind { World, Body, Site, Frame }`
and an element name. The Kinova GEN3 MJCF exports `pinch_site` on the bracelet,
which already encodes the tool offset and flip, so a gripper attaches with no
manual `pos` or `quat`:

```cpp
mj_kdl::AttachmentSpec gripper{
    .mjcf_path          = mj_kdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
    .prefix             = "g_",
    .contact_exclusions = {},
};

mj_kdl::RobotSpec robot_spec;
robot_spec.path = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml");
robot_spec.attachments.push_back(gripper);

mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(robot_spec);
mj_kdl::build_scene(&model, &data, &sc);
```

Optional `pos` and `quat` on the attachment spec are composed with the parent
site pose, so you can still add small offsets. `quat` is `[x, y, z, w]` and
defaults to identity `{ 0, 0, 0, 1 }`:

```cpp
gripper.pos[2] = 0.005; // +5 mm along the tool z
// +15 deg about the tool z
gripper.quat[0] = 0.0;
gripper.quat[1] = 0.0;
gripper.quat[2] = 0.130526;
gripper.quat[3] = 0.991445;
```

If a model has no suitable site, attach by body name instead:

```cpp
gripper.attach_to = { mj_kdl::AttachKind::Body, "bracelet_link" };
gripper.pos[2]    = -0.061525;
// 180 deg about x is exactly [x, y, z, w] = { 1, 0, 0, 0 }
gripper.quat[0]   = 1.0;
gripper.quat[3]   = 0.0;
```

Chains are supported: push multiple `AttachmentSpec` entries in order, such as
mount, force-torque sensor, then gripper. Each entry's `attach_to` may reference
any body, site, or frame added by prior entries.

## Multi-Robot Scene

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

## Table And Scene Objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so a robot can
follow a tabletop site without hand-threading world-frame heights. Build order
in `build_scene()` is decorations, objects in declaration order, robots, then
cameras. A robot's `attach_to` can reference any prior object, and a child
object's `attach_to` can reference any earlier object in `SceneSpec::objects`.

`SceneObject` has no defaults for `shape`, `size`, `rgba`, `mass`, or
`friction`. For MJCF-backed objects, when `mjcf_path` is set, those fields are
ignored at runtime. For primitives, `build_scene()` checks:

- `shape == Shape::Unspecified`: error, object skipped.
- `size[i] <= 0` for the relevant dimensions of the shape: error, skipped.
- `mass <= 0` on a non-fixed primitive: error, skipped.

```cpp
mj_kdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;

mj_kdl::SceneObject table{
    .name      = "table",
    .mjcf_path = mj_kdl_examples::asset("table.xml"),  // ships a table_top site
    .pos       = { 0.0, 0.0, 0.7 },
    .fixed     = true,
};
sc.objects.push_back(table);

std::string mount = mj_kdl::scene_object_site_name(table, "table_top");

sc.robots.push_back(mj_kdl::RobotSpec{
    .path      = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, mount.c_str() },
});

sc.objects.push_back(mj_kdl::SceneObject{
    .name      = "fixture",
    .mjcf_path = "fixture.xml",
    .attach_to = { mj_kdl::AttachKind::Body, "table" },
    .pos       = { 0.0, 0.0, 0.0 },
    .fixed     = true,
});

sc.objects.push_back(mj_kdl::SceneObject{
    .name     = "red_cube",
    .shape    = mj_kdl::Shape::BOX,
    .size     = { 0.03, 0.03, 0.03 },
    .pos      = { 0.35, 0.10, 0.73 },
    .rgba     = { 1.0f, 0.0f, 0.0f, 1.0f },
    .mass     = 0.1,
    .condim   = mj_kdl::Condim::Torsional,
    .friction = { 0.8, 0.02, 0.001 },
});

mj_kdl::build_scene(&model, &data, &sc);
```

MuJoCo restricts free joints to top-level bodies, so a non-fixed primitive with
a free joint must stay world-anchored.

## Cameras And Poses

Add fixed world cameras through `SceneSpec::cameras`. `pos` and `fovy` are
required; `quat` is `[x, y, z, w]` and defaults to identity `{ 0, 0, 0, 1 }`.

```cpp
sc.cameras.push_back(mj_kdl::CameraSpec{
    .name = "overview",
    .pos  = { 1.8, -2.0, 1.4 },
    .fovy = 45.0,
});
```

After building, `get_camera_names()` returns robot MJCF cameras and cameras
added through the scene spec. `use_camera()` switches a viewer or recorder to a
fixed camera; pass `nullptr` or `""` to return to the free camera.

```cpp
for (const auto &name : mj_kdl::get_camera_names(model)) {
    LOG_INFO("camera: " << name);
}
mj_kdl::use_camera(&viewer, model, "overview");
mj_kdl::use_camera(&vr, model, "overview");
```

Use `get_body_frame()` and `get_site_frame()` to read world poses as
`KDL::Frame`; both call `mj_forward()` before reading MuJoCo pose arrays.
`set_body_pose()` teleports a free body and zeroes its velocity. The quaternion,
when supplied in C++, uses MuJoCo order `[w, x, y, z]`.

```cpp
KDL::Frame tcp;
mj_kdl::get_site_frame(model, data, "g_pinch", &tcp);

const double pos[3]  = { 0.45, 0.0, 0.75 };
const double quat[4] = { 1.0, 0.0, 0.0, 0.0 };
mj_kdl::set_body_pose(model, data, "red_cube", pos, quat);
```

## Control Loop

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot);

KDL::JntArray q(n), g(n);
while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
}

mj_kdl::cleanup(&viewer);
mj_kdl::cleanup(&robot);
mj_kdl::destroy_scene(model, data);
```

`update(&robot)` does both halves of the port synchronization: it reads MuJoCo
joint state into `jnt_pos_msr`, `jnt_vel_msr`, and `jnt_trq_msr`, then applies
the command ports. In `POSITION` mode it writes `jnt_pos_cmd` to actuator
controls. In `TORQUE` mode it writes `jnt_trq_cmd` to `qfrc_applied` and
neutralizes position actuators at the current joint positions.

Use `set_joint_pos(&robot, q, call_forward)` to seed joint state directly in
KDL order. With `call_forward=true`, body/site poses are updated immediately.

```cpp
KDL::JntArray q_home(robot.n_joints);
for (unsigned i = 0; i < robot.n_joints; ++i) q_home(i) = 0.0;
mj_kdl::set_joint_pos(&robot, q_home);
```

## Reset

`reset(Env*)` resets the environment runtime to its initial keyframe, calls an
optional environment reset hook, re-seeds all registered robots' command ports
to the current measured state, and clears stale robot forces. Use the hook to
put objects, controllers, and task state back at their episode start values:

```cpp
mj_kdl::Env env;
mj_kdl::init_env(&env, &sc);

mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, env.model, env.data, "base_link", "bracelet_link");
mj_kdl::env_add_robot(&env, &robot);

env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    mj_kdl::set_joint_pos(&robot, q_home, false);
    episode_step = 0;
};

mj_kdl::ResetOptions opts;
opts.keyframe = 0;
mj_kdl::ResetInfo info = mj_kdl::reset(&env, &opts);
```

Use `Env` when runtime rebuilds or episode resets should keep robot handles
synchronized. Register every borrowed robot with `env_add_robot()`. Cleanup
destroys only the environment-owned model/data and clears registrations:

```cpp
mj_kdl::cleanup(&env);
```

## Headless Video Recording

```cpp
// Requires BUILD_RECORDER=ON (default) and ffmpeg in PATH.
mj_kdl::VideoRecorder vr;
mj_kdl::init_video_recorder(
    &vr, model, "sim.mp4", mj_kdl::VideoResolution::R1080p);

vr.cam.azimuth   = 135.0;
vr.cam.elevation = -20.0;
vr.cam.distance  = 2.5;

for (int i = 0; i < 3000; ++i) {
    mj_kdl::update(&robot);
    mj_kdl::step(&robot);
    mj_kdl::record_frame(&vr, model, data);
}

mj_kdl::cleanup(&vr);
```

Interactive recording is available from the Simulate UI:

1. Open the left Simulation panel.
2. Scroll to Recorder.
3. Set `Path`, `Camera`, `Resolution`, and `FPS`.
4. Press `Start rec`.
5. Press `Stop rec`.

The recorder camera list includes `Current`, `Free`, `Tracking`, robot MJCF
cameras, and cameras added through `SceneSpec::cameras`.

## Runtime Add And Remove Objects

```cpp
mj_kdl::scene_add_object(&model, &data, &sc, cube);
mj_kdl::scene_remove_object(&model, &data, &sc, "red_cube");
// model/data are replaced; re-call init_robot_from_*() on the new pointers.
```

Prefer the `Env` overloads when registered robot handles should be re-initialized
automatically:

```cpp
mj_kdl::scene_add_object(&env, cube);
mj_kdl::scene_remove_object(&env, "red_cube");
```

With raw `mjModel**` / `mjData**` overloads, any `Robot` that borrowed the old
pointers is stale until you call `init_robot_from_mjcf()` again. In Python,
`Scene.add_object()`, `Scene.remove_object()`, `Env.add_object()`, and
`Env.remove_object()` perform that rebind step for existing Python `Robot`
 handles. See [Python API guide](python.md) for Python ownership rules.

## Manual Viewer Loop

`init_window_sim()` opens the full Simulate UI in a background render thread and
lets `step(&robot)` handle physics, rendering, pause, perturbation, and pacing.
For a lower-level GLFW window, use `init_window()`, drive MuJoCo yourself, and
call `render()`:

```cpp
mj_kdl::Viewer viewer;
mj_kdl::init_window(&viewer, &robot);

while (mj_kdl::is_running(&viewer)) {
    mj_step(model, data);
    mj_kdl::render(&viewer, model, data);
}

mj_kdl::cleanup(&viewer);
```

Use `init_window_sim(&viewer, model, data, "object scene")` for object-only
scenes that still need the full Simulate UI. Use `step(&viewer, model, data)`
for multi-robot or no-robot loops where the viewer should own the same pacing
behavior as the Simulate UI path.

## Viewer Controls

| Input | Action |
|-------|--------|
| Left drag | Orbit camera |
| Right drag | Pan camera |
| Scroll | Zoom |
| Double-click body | Select body, name shown in the Perturb panel |
| `D` | Deselect body |
| `Space` | Pause or resume |
| `,` | Decrease wrapper real-time factor |
| `.` | Increase wrapper real-time factor |

Applying a perturbation force or torque to the selected body can be done in two
ways:

- Perturb panel: set `Drag` to `Force` or `Torque`, then left-drag in the 3D
  view. Right-drag still pans the camera. `Shift` drags in the horizontal plane
  instead of the vertical.
- Keyboard and mouse: `Ctrl` + right-drag applies force, `Ctrl` + left-drag
  applies torque.

All other controls, including reset, quit, rendering flags, live camera
selection, and recording, are in the MuJoCo panels.
