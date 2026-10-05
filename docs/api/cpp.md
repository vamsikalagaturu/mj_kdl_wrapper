# C++ API Guide {#page_api_cpp}

This page collects the C++ wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated Doxygen API pages
for `include/mjkdl/mjkdl.hpp`.

Coming from 0.4 (`mj_kdl_wrapper`)? See [Migrating to 0.5](@ref sec_migrate_mjkdl).
From 0.3.x, the `Env` also took over the loop; see [Migrating to 0.4](@ref sec_migrate_env).
From 0.2.x, placement orientation also moved from `euler` to `quat` `[x, y, z, w]`; see
[Migrating from 0.2.x](@ref sec_migrate_quat). Units, frames and what persists between
calls: [Conventions](@ref page_conventions).

## Errors

Every call that can fail on its input returns `mjkdl::Status`. It converts to `true` on
success; on failure `error` says why (the same text is logged):

```cpp
if (mjkdl::Status s = mjkdl::init_env(&env, &sc); !s) {
    std::cerr << "init_env failed: " << s.error << "\n";
    return 1;
}
```

Per-cycle getters (`get_body_frame()`, `get_site_frame()`) return `bool`, and the
`bind_scene_*()` calls return a pointer (`nullptr` when the name does not resolve).

## Resolving Models And Assets

The examples and tests resolve paths through `example_paths.hpp` (a header-only
helper under `src/examples/`) against the source `assets/` directory, compiled in as
`MJKDL_ASSETS_DIR`:

- `mjkdl_examples::asset("kinova_gen3/gen3.xml")` returns the bundled file's path and
  throws when it is missing.
- `mjkdl_examples::find_asset(...)` returns `""` instead, which is how tests self-skip.

The Python counterpart is `mjkdl.ASSETS_DIR / "kinova_gen3/gen3.xml"`. `cmake --install` copies
the same files into `~/.cache/mjkdl/assets` for programs outside this tree. Any other MJCF
(e.g. MuJoCo Menagerie's) goes into `RobotSpec::path`; one that brings its own floor needs
`add_floor = false`.

## Load From MJCF

`SceneSpec` has no defaults for `timestep`, `add_floor`, or `add_skybox`.
Those are choices, not values the library can guess. `timestep` starts unset (NaN), and
`build_scene()` fails unless it is > 0. String fields throughout the specs are `std::string`;
empty means "not set", and the `Env` keeps its own copy of the spec. `SceneSpec::robots` may be empty; object-only
scenes are valid. `floor_z` places the ground plane along the world z axis
(default `0.0`), for scenes whose world frame is not at ground level.

```cpp
#include "example_paths.hpp"
#include "mjkdl/mjkdl.hpp"

mjkdl::SceneSpec sc;
sc.timestep   = 0.002;   // [s]; required, must be > 0
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(mjkdl::RobotSpec{
    .path = mjkdl_examples::asset("kinova_gen3/gen3.xml")
});

mjkdl::Env env;   // owns the model/data; not copied or moved
mjkdl::init_env(&env, &sc);
```

Every control mode writes its command to actuators, so each robot joint needs one in the MJCF.
`build_scene()` adds the actuators for the extra modes in `RobotSpec::modes` from the one each
joint already has; a joint with none (a raw URDF import, say) gets none, and
`set_control_mode()` refuses the mode.

For an object-only scene, add MJCF or primitive `SceneObject` entries and leave
`sc.robots` empty:

```cpp
mjkdl::SceneObject cabinet{
    .name      = "cabinet",
    .mjcf_path = mjkdl_examples::asset("cabinet/cabinet.xml"),
    .fixed     = true,
};
sc.objects.push_back(cabinet);
mjkdl::init_env(&env, &sc);
```

`save_model_xml(model, path)` writes a live model from `build_scene()` or `init_env()` back to
MJCF, including runtime changes to its real-valued fields. Use it when you want to build a
combined scene once and reload the merged model later through MuJoCo.
`build_scene(&model, &data, &sc)` compiles a raw pair without an `Env` (for such tools);
`destroy_scene(model, data)` frees it.

```cpp
mjkdl::save_model_xml(env.model, "combined_scene.xml");
mj_saveModel(env.model, "combined_scene.mjb", nullptr, 0);
```

`SceneSpec` and `build_scene()` are the only way in: plugins, decorations, objects, robots,
cameras, compilation and ownership all follow the same path.

`env.model` and `env.data` are plain MuJoCo pointers, so any `mj_*` call works on them
directly. To run the `Env` on a pair of your own, set `env.adopt` before `init_env()`: it
receives each compiled `(mjModel*, mjData*)` (at init and at every `scene_add_object()` /
`scene_remove_object()` rebuild) and returns the pair the `Env` runs on, which the `Env` then
never frees; freeing it, and the compiled pair when you return a different one, is yours.

## Logging

`LogLevel` is a severity threshold, `INFO` < `WARN` < `ERROR` < `NONE`: a message prints when
its level is at or above the threshold. The default, `INFO`, prints everything; `WARN` prints
warnings and errors, `ERROR` errors only, `NONE` nothing. It is one library-wide setting:

```cpp
mjkdl::set_log_level(mjkdl::LogLevel::WARN);   // quiet the scene-construction INFO lines
```

Your own code can log through the same threshold with `MJ_LOG_INFO()`, `MJ_LOG_WARN()` and
`MJ_LOG_ERROR()`, which print to stderr with the file, line and function; the argument may
stream:

```cpp
MJ_LOG_WARN("joint " << i << " saturated");
```

## Init A KDL Chain

```cpp
mjkdl::Robot robot;
mjkdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link");

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
include the full tool inertia and FK uses the TCP site. `tool_body` is the tool's root body:
for the 2F-85 that is `g_base_mount`, whose mass (inferred from its mesh) KDL would otherwise
miss. The inertia is lumped at the pose the model is in when the robot is initialized:

```cpp
const mjkdl::ToolFrameSpec tool{ .tool_body = "g_base_mount", .tcp_site = "g_pinch" };
mjkdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool);

KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), g(n);
dyn.JntToGravity(q, g);
```

For a wrist force-torque sensor, attach the sensor MJCF first, then attach the
gripper to a site exported by that sensor asset:

```cpp
mjkdl::AttachmentSpec ft_sensor{
    .mjcf_path = mjkdl_examples::asset("ft_sensor.xml"),
    .attach_to = { mjkdl::AttachKind::Site, "pinch_site" },
};

mjkdl::AttachmentSpec gripper{
    .mjcf_path = mjkdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to = { mjkdl::AttachKind::Site, "wrist_ft_site" },
    .prefix    = "g_",
};

mjkdl::RobotSpec robot_spec;
robot_spec.path = mjkdl_examples::asset("kinova_gen3/gen3.xml");
robot_spec.attachments = { ft_sensor, gripper };
```

Then register the logical force-torque sensor through `ToolFrameSpec`. MuJoCo
stores it as separate `<force>` and `<torque>` sensors; the wrapper combines one
pair into a `KDL::Wrench`.

```cpp
mjkdl::ForceTorqueSensorSpec ft{ .name = "wrist_ft", .frame_site = "wrist_ft_site" };
mjkdl::ToolFrameSpec tool{
    .tool_body  = "g_base_mount",
    .tcp_site   = "g_pinch",
    .ft_sensors = { ft },
};

mjkdl::update(&env);
KDL::Wrench wrench = robot.ft_sensors[0].wrench;   // robot.ft_sensors, in ToolFrameSpec order
```

When `force_sensor` and `torque_sensor` are omitted, the wrapper resolves
`{name}_force` and `{name}_torque`.

When the chain comes from the same description that produced the MJCF (so the solvers use the
authored dynamics), pass it instead of deriving it; `joint_names` are the MuJoCo joints in chain
order, and no tool inertia is lumped onto it:

```cpp
mjkdl::init_robot_from_chain(&robot, &env, chain, joint_names, "", &tool);
```

## Attach MJCF Bodies

`AttachTarget` is a tagged pair of `AttachKind { World, Body, Site, Frame }`
and an element name. The Kinova GEN3 MJCF exports `pinch_site` on the bracelet,
which already encodes the tool offset and flip, so a gripper attaches with no
manual `pos` or `quat`:

```cpp
mjkdl::AttachmentSpec gripper{
    .mjcf_path          = mjkdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to          = { mjkdl::AttachKind::Site, "pinch_site" },
    .prefix             = "g_",
    .contact_exclusions = {},
};

mjkdl::RobotSpec robot_spec;
robot_spec.path = mjkdl_examples::asset("kinova_gen3/gen3.xml");
robot_spec.attachments.push_back(gripper);

mjkdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots.push_back(robot_spec);
mjkdl::init_env(&env, &sc);
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
gripper.attach_to = { mjkdl::AttachKind::Body, "bracelet_link" };
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
mjkdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;
sc.robots = {
    mjkdl::RobotSpec{ .path = "gen3.xml", .pos = { -0.5, 0.0, 0.0 } },
    mjkdl::RobotSpec{ .path = "gen3.xml", .prefix = "r2_", .pos = { 0.5, 0.0, 0.0 } },
};
mjkdl::init_env(&env, &sc);

mjkdl::Robot robot1, robot2;
mjkdl::init_robot_from_mjcf(&robot1, &env, "base_link", "bracelet_link");
mjkdl::init_robot_from_mjcf(&robot2, &env, "r2_base_link", "r2_bracelet_link");
```

The `prefix` argument is prepended to every name the call resolves: base and tip bodies, tool
body, TCP site and F/T sensor names. `("base_link", "bracelet_link", "r2_")` and
`("r2_base_link", "r2_bracelet_link", "")` build the same robot.

Each robot gets a group of actuators per control mode (see
[Torque control](@ref page_howto_torque_control)), so the two can run different modes.

## Table And Scene Objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so a robot can
follow a tabletop site without hand-threading world-frame heights. Build order
in `build_scene()` is decorations, objects in declaration order, robots, then
cameras. A robot's `attach_to` can reference any prior object, and a child
object's `attach_to` can reference any earlier object in `SceneSpec::objects`.

`SceneObject` has no defaults for `shape`, `size`, `rgba`, `mass`, or
`friction`: the numbers start unset (NaN). For MJCF-backed objects, when `mjcf_path` is set,
they are ignored (`rgba` with `has_rgba` recolours the asset). For primitives,
`build_scene()` fails, naming the object, when:

- `shape == Shape::Unspecified`;
- a size the shape uses is unset or not positive;
- `rgba` or `friction` is unset;
- `mass` is unset or not positive on a non-fixed primitive.

```cpp
mjkdl::SceneSpec sc;
sc.timestep   = 0.002;
sc.add_floor  = true;
sc.add_skybox = true;

mjkdl::SceneObject table{
    .name      = "table",
    .mjcf_path = mjkdl_examples::asset("table.xml"),  // ships a table_top site
    .pos       = { 0.0, 0.0, 0.7 },
    .fixed     = true,
};
sc.objects.push_back(table);

std::string mount = "table_top";   // the asset's own site name; SceneObject::prefix would prepend to it

sc.robots.push_back(mjkdl::RobotSpec{
    .path      = mjkdl_examples::asset("kinova_gen3/gen3.xml"),
    .attach_to = { mjkdl::AttachKind::Site, mount },
});

sc.objects.push_back(mjkdl::SceneObject{
    .name      = "fixture",
    .mjcf_path = "fixture.xml",
    .attach_to = { mjkdl::AttachKind::Body, "table" },
    .pos       = { 0.0, 0.0, 0.0 },
    .fixed     = true,
});

sc.objects.push_back(mjkdl::SceneObject{
    .name     = "red_cube",
    .shape    = mjkdl::Shape::BOX,
    .size     = { 0.03, 0.03, 0.03 },
    .pos      = { 0.35, 0.10, 0.73 },
    .rgba     = { 1.0f, 0.0f, 0.0f, 1.0f },
    .mass     = 0.1,
    .condim   = mjkdl::Condim::Torsional,
    .friction = { 0.8, 0.02, 0.001 },
});

mjkdl::init_env(&env, &sc);
```

MuJoCo restricts free joints to top-level bodies, so a non-fixed primitive with
a free joint must stay world-anchored.

## Cameras And Poses

Add fixed world cameras through `SceneSpec::cameras`. `pos` and `fovy` are
required; `quat` is `[x, y, z, w]` and defaults to identity `{ 0, 0, 0, 1 }`.

```cpp
sc.cameras.push_back(mjkdl::CameraSpec{
    .name = "overview",
    .pos  = { 1.8, -2.0, 1.4 },
    .fovy = 45.0,
});
```

After building, the model holds robot MJCF cameras and cameras added through the scene spec.
`use_camera()` switches the viewer to a fixed camera; pass `nullptr` or `""` to return to the
free camera. A recorder's camera is its `vr.cam`, written directly:

```cpp
for (int i = 0; i < env.model->ncam; ++i) {
    MJ_LOG_INFO("camera: " << mj_id2name(env.model, mjOBJ_CAMERA, i));
}
mjkdl::use_camera(&env.viewer, env.model, "overview");
vr.cam.type       = mjCAMERA_FIXED;
vr.cam.fixedcamid = mj_name2id(env.model, mjOBJ_CAMERA, "overview");
```

Use `get_body_frame()` and `get_site_frame()` to read world poses as
`KDL::Frame`. They recompute the kinematics only when the state has changed
since they were last computed, so many reads per step cost one forward pass and a
direct `qpos` write is picked up with no extra call.
`set_body_pose()` teleports a free body and zeroes its velocity. The quaternion is
`[x, y, z, w]`, like every quaternion in the API.

```cpp
KDL::Frame tcp;
mjkdl::get_site_frame(&env, "g_pinch", &tcp);

const double pos[3]  = { 0.45, 0.0, 0.75 };
const double quat[4] = { 0.0, 0.0, 0.0, 1.0 };   // [x, y, z, w]: identity
mjkdl::set_body_pose(&env, "red_cube", pos, quat);
```

A joint no `Robot` owns is read through a slot: `bind_scene_joint(&env.scene, name)` once,
then its `position` and `velocity` after each `update(&env)`. Or read `env.data->qpos` at the
joint's `jnt_qposadr` directly.

## Control Loop

```cpp
mjkdl::set_control_mode(&robot, mjkdl::CtrlMode::TORQUE);   // seeds the torque ports, no jump

mjkdl::open_viewer(&env);   // optional; the loop is the same headless

KDL::JntArray q(n), g(n);
while (env.data->time < 5.0 && mjkdl::step(&env)) {   // step() is false only on window close
    mjkdl::update(&env);
    for (unsigned i = 0; i < n; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = g(i);
}

mjkdl::cleanup(&env);   // closes the viewer, frees model/data
```

`step(&env)` advances one timestep (`mj_step2()` then `mj_step1()`), so joint
state, frames and position/velocity sensors all describe the new state. It
returns false once the viewer window is closed.

`update(&env)` does both halves of the port synchronization for every registered
robot and every scene slot: it reads MuJoCo joint state into `jnt_pos_msr`,
`jnt_vel_msr`, and `jnt_trq_msr` (the active mode's actuator torque) and the F/T
wrenches, then applies the command ports. Each control mode has its own
actuators: `POSITION` writes `jnt_pos_cmd`, `VELOCITY` `jnt_vel_cmd`, `TORQUE`
`jnt_trq_cmd`, to that mode's actuator `ctrl` (gear applied, clamped to `ctrlrange`, flagged
in `jnt_saturated`). Nothing is written to `qfrc_applied`. `set_control_mode(&robot, mode)`
switches modes without a jump (setting `robot.ctrl_mode` directly switches at the next
`update()` without seeding, keeping commands you primed), and `joint_force_limits(&robot)` returns each joint's torque limit in the
active mode. Which modes a robot offers is set in `RobotSpec::modes`, and for an attached arm
in `AttachmentSpec::modes`; see
[Torque control](@ref page_howto_torque_control).

Scene slots cover what no `Robot` chain owns: bind them once with
`bind_scene_joint()`, `bind_scene_free_body()`, `bind_scene_wrench()` and
`bind_scene_actuator()` on `&env.scene`, then `update(&env)` samples and applies
them. A gripper drive is an actuator slot:

```cpp
auto *fingers = mjkdl::bind_scene_actuator(&env.scene, "g_fingers_actuator");
fingers->command = 0.82;   // ctrl units (the 2F-85's driver angle, closed); next update(&env)
```

Use `set_joint_pos(&robot, q)` to seed joint state directly in KDL order;
frames read afterwards follow the new positions.

```cpp
KDL::JntArray q_home(robot.n_joints);
for (unsigned i = 0; i < robot.n_joints; ++i) q_home(i) = 0.0;
mjkdl::set_joint_pos(&robot, q_home);
```

## Reset

`reset(Env*)` resets everything the `Env` holds: MuJoCo data to the keyframe (or
the model default); then every registered robot's ports and F/T readings and every scene
slot, seeded from the reset state so nothing jumps (a requested `ctrl_mode` is kept); then
the optional `on_reset` hook; then the measurements are read from the result. Because the
hook runs after the re-seed, it can prime commands, and a pose it sets is what the ports
read. Use it to put objects, controllers, and task state back at their episode start values:

```cpp
mjkdl::Robot robot;
mjkdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link");

env.on_reset = [&](mjkdl::ResetContext *ctx) {
    mjkdl::set_joint_pos(&robot, q_home);
    episode_step = 0;
};

mjkdl::ResetOptions opts;
opts.keyframe = 0;
mjkdl::ResetInfo info = mjkdl::reset(&env, &opts);
```

Each part's runtime state is one struct (`ForceTorqueReading`, and the
`Scene*Reading` / `Scene*Command` bases of the slots) that reset assigns afresh,
so a field added to one is reset without further code; a part without a reset
overload does not compile. `RobotPorts` is rewritten in place instead: its vectors
are sized once by `init_robot_*()`, so a pointer to a port element stays valid
across resets. The Simulate UI's reset button runs the same path,
hook included. `on_reset` may be set before or after `init_env()`.

`cleanup(&env)` closes the viewer, frees the model/data and forgets the robots,
which are not deleted.

## Headless Video Recording

```cpp
// Requires BUILD_RECORDER=ON (default) and ffmpeg in PATH.
mjkdl::VideoRecorder vr;
mjkdl::init_video_recorder(
    &vr, env.model, "sim.mp4", mjkdl::VideoResolution::R1080p);

vr.cam.azimuth   = 135.0;
vr.cam.elevation = -20.0;
vr.cam.distance  = 2.5;

for (int i = 0; i < 3000; ++i) {
    mjkdl::step(&env);
    mjkdl::update(&env);
    mjkdl::record_frame(&vr, &env);
}

mjkdl::cleanup(&vr);
```

To get frames into memory instead of a file, initialize offscreen rendering only and read each
frame as top-down RGB8:

```cpp
mjkdl::VideoRecorder vr;
mjkdl::init_offscreen(&vr, env.model, 640, 480);
std::vector<std::uint8_t> rgb(640 * 480 * 3);
mjkdl::render_rgb(&vr, &env, rgb.data());
mjkdl::cleanup(&vr);
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
mjv_defaultFreeCamera(env.model, &vr.cam);   // the view a window opens with
```

The Simulate UI's Recorder panel offers `Current`, `Free`, `Tracking`, and the
fixed model cameras; every one of them renders offscreen. `Current` follows the
camera the GUI is being driven with, frame by frame.

## Runtime Add And Remove Objects

```cpp
mjkdl::scene_add_object(&env, cube);
mjkdl::scene_remove_object(&env, "red_cube");
```

Both append to or erase from `env.spec.objects` and rebuild. The model/data are
replaced; the time, `qpos`/`qvel` (by joint name and type) and `act`/`ctrl` (by actuator
name) carry over, and the port commands are kept. Registered robots, scene slots, the viewer
and `VideoRecorder`s follow the new model; a slot whose name is gone is unbound and skipped.
If a robot or F/T sensor cannot be rebound, the call returns the error and the `Env` is left
as it was. Both fail when called from `on_reset`. MuJoCo ids you cached yourself must be
recomputed. In Python, `Env.add_object()` and `Env.remove_object()` do the
same. See [Python API guide](python.md) for Python ownership rules.

## Viewer Loop

`open_viewer(&env, "title")` opens the full Simulate UI in a background render
thread, or returns an error `Status` when no window can be created; `step(&env)` then also
handles its pause, perturbation and recording.
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
