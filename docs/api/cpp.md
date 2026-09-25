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
scenes are valid. `floor_z` places the ground plane along the world z axis
(default `0.0`), for scenes whose world frame is not at ground level.

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

mj_kdl::Env env;   // owns the model/data; not copied or moved
mj_kdl::init_env(&env, &sc);
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
mj_kdl::init_env(&env, &sc);
```

`save_model_xml(model, path)` writes the most recently compiled model back to
MJCF. Use it when you want to build a combined scene once and reload the merged
model later through MuJoCo. `build_scene(&model, &data, &sc)` compiles a raw pair
without an `Env` (for such tools); `destroy_scene(model, data)` frees it.

```cpp
mj_kdl::save_model_xml(env.model, "combined_scene.xml");
mj_saveModel(env.model, "combined_scene.mjb", nullptr, 0);
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
mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link");

unsigned n = robot.n_joints;  // 7 for Kinova GEN3
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
```

Init registers the robot with `env`, which reads, commands and resets it from then
on; the `Env` never deletes it, and a `Robot` is neither copied nor moved. After
init, the port vectors (`RobotPorts`, which `Robot` derives from) are sized to
`n_joints`, ordered like the KDL chain, and hold the current pose. Use
`joint_names` and `joint_limits` to inspect that order before writing
controllers; the MuJoCo index maps are private.

When a tool or gripper is attached, pass a `ToolFrameSpec` so KDL dynamics
include the full tool inertia and FK uses the TCP site:

```cpp
const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base_mount", .tcp_site = "g_pinch" };
mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool);

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
    .tool_body  = "g_base_mount",
    .tcp_site   = "g_pinch",
    .ft_sensors = { ft },
};

mj_kdl::update(&env);
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
mj_kdl::init_env(&env, &sc);
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
mj_kdl::init_env(&env, &sc);

mj_kdl::Robot robot1, robot2;
mj_kdl::init_robot_from_mjcf(&robot1, &env, "base_link", "bracelet_link");
mj_kdl::init_robot_from_mjcf(&robot2, &env, "base_link", "bracelet_link", "r2_");
```

Each robot gets a group of actuators per control mode (see
[Torque control](@ref page_howto_torque_control)), so the two can run different modes.

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
    .attach_to = { mj_kdl::AttachKind::Site, mount },
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

mj_kdl::init_env(&env, &sc);
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
for (const auto &name : mj_kdl::get_camera_names(env.model)) {
    LOG_INFO("camera: " << name);
}
mj_kdl::use_camera(&env.viewer, env.model, "overview");
mj_kdl::use_camera(&vr, env.model, "overview");
```

Use `get_body_frame()` and `get_site_frame()` to read world poses as
`KDL::Frame`. They recompute the kinematics only when the state has changed
since they were last computed, so many reads per step cost one forward pass and a
direct `qpos` write is picked up with no extra call.
`set_body_pose()` teleports a free body and zeroes its velocity. The quaternion is
`[x, y, z, w]`, like every quaternion in the API.

```cpp
KDL::Frame tcp;
mj_kdl::get_site_frame(&env, "g_pinch", &tcp);

const double pos[3]  = { 0.45, 0.0, 0.75 };
const double quat[4] = { 1.0, 0.0, 0.0, 0.0 };
mj_kdl::set_body_pose(&env, "red_cube", pos, quat);
```

`get_joint_position(&env, name, &q)` and `get_joint_velocity()` read one joint by
name (or the joint an actuator of that name drives).

## Control Loop

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

mj_kdl::open_viewer(&env);   // optional; the loop is the same headless

KDL::JntArray q(n), g(n);
while (mj_kdl::step(&env)) {
    mj_kdl::update(&env);
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
}

mj_kdl::cleanup(&env);   // closes the viewer, frees model/data
```

`step(&env)` advances one timestep (`mj_step2()` then `mj_step1()`), so joint
state, frames and position/velocity sensors all describe the new state. It
returns false once the viewer window is closed.

`update(&env)` does both halves of the port synchronization for every registered
robot and every scene slot: it reads MuJoCo joint state into `jnt_pos_msr`,
`jnt_vel_msr`, and `jnt_trq_msr` (the active mode's actuator torque) and the F/T
wrenches, then applies the command ports. Each control mode has its own
actuators: `POSITION` writes `jnt_pos_cmd`, `VELOCITY` `jnt_vel_cmd`, `TORQUE`
`jnt_trq_cmd`, to that mode's actuator `ctrl` (clamped to `ctrlrange`, flagged in
`jnt_saturated`). Nothing is written to `qfrc_applied`.

Scene slots cover what no `Robot` chain owns: bind them once with
`bind_scene_joint()`, `bind_scene_free_body()`, `bind_scene_wrench()` and
`bind_scene_actuator()` on `&env.scene`, then `update(&env)` samples and applies
them. A gripper drive is an actuator slot:

```cpp
auto *fingers = mj_kdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
fingers->command = 0.8;   // ctrl units; applied by the next update(&env)
```

Use `set_joint_pos(&robot, q)` to seed joint state directly in KDL order;
frames read afterwards follow the new positions.

```cpp
KDL::JntArray q_home(robot.n_joints);
for (unsigned i = 0; i < robot.n_joints; ++i) q_home(i) = 0.0;
mj_kdl::set_joint_pos(&robot, q_home);
```

## Reset

`reset(Env*)` resets everything the `Env` holds: MuJoCo data to the keyframe (or
the model default), then the optional `on_reset` hook, then every registered
robot's ports and F/T readings and every scene slot, seeded from the reset state
so nothing jumps. Use the hook to put objects, controllers, and task state back
at their episode start values:

```cpp
mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link");

env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    mj_kdl::set_joint_pos(&robot, q_home);
    episode_step = 0;
};

mj_kdl::ResetOptions opts;
opts.keyframe = 0;
mj_kdl::ResetInfo info = mj_kdl::reset(&env, &opts);
```

Each part's runtime state is one struct (`RobotPorts`, `ForceTorqueReading`, and
the `Scene*Reading` / `Scene*Command` bases of the slots) that reset assigns afresh,
so a field added to one is reset without further code; a part without a reset
overload does not compile. The Simulate UI's reset button runs the same path,
hook included.

`cleanup(&env)` closes the viewer, frees the model/data and forgets the robots,
which are not deleted.

## Headless Video Recording

```cpp
// Requires BUILD_RECORDER=ON (default) and ffmpeg in PATH.
mj_kdl::VideoRecorder vr;
mj_kdl::init_video_recorder(
    &vr, env.model, "sim.mp4", mj_kdl::VideoResolution::R1080p);

vr.cam.azimuth   = 135.0;
vr.cam.elevation = -20.0;
vr.cam.distance  = 2.5;

for (int i = 0; i < 3000; ++i) {
    mj_kdl::step(&env);
    mj_kdl::update(&env);
    mj_kdl::record_frame(&vr, &env);
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

## Recording What The Window Shows

Recording is offscreen only: a `VideoRecorder` has its own headless context and
its own thread, so a window never stalls on a readback while it records. To
record the view a freshly opened GUI window shows, point the recorder at the
default free camera:

```cpp
mj_kdl::use_camera(&vr, env.model, "");   // mjv_defaultFreeCamera, the view a window opens with
```

The Simulate UI's Recorder panel offers `Current`, `Free`, `Tracking`, and the
fixed model cameras; every one of them renders offscreen. `Current` follows the
camera the GUI is being driven with, frame by frame.

## Runtime Add And Remove Objects

```cpp
mj_kdl::scene_add_object(&env, cube);
mj_kdl::scene_remove_object(&env, "red_cube");
```

Both append to or erase from `env.spec.objects` and rebuild. The model/data are
replaced, and registered robots, scene slots and the viewer follow the new model;
a slot whose name is gone is unbound and skipped. MuJoCo ids you cached yourself
must be recomputed. In Python, `Env.add_object()` and `Env.remove_object()` do the
same. See [Python API guide](python.md) for Python ownership rules.

## Viewer Loop

`open_viewer(&env, "title")` opens the full Simulate UI in a background render
thread; `step(&env)` then also handles its pause, perturbation and recording.
It works the same for object-only scenes. The viewer is `env.viewer`; pass it to
`key_pressed()`, `capture_key()`, `use_camera()`, `set_free_camera()` and the
overlay helpers. `pace_realtime(&env)` paces a loop to the viewer's real-time
factor and does nothing headless. `cleanup(&env)` closes the window.

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
