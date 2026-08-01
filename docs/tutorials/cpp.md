# C++ Tutorial {#page_tutorial_cpp}

This tutorial builds a C++ simulation application in layers: compile a robot scene,
add KDL control, add reset hooks, add objects and cameras, use the Simulate UI,
record video, and then extend the scene to more robots and more complex task assets.

The snippets assume the package is built with MuJoCo Menagerie available, because
the examples use the Kinova GEN3 arm and Robotiq 2F-85 gripper.

## How To Read This Tutorial

The wrapper has four layers. Keep them separate and the API stays simple:

| Layer | Type | Responsibility |
|-------|------|----------------|
| Scene description | `SceneSpec`, `RobotSpec`, `AttachmentSpec`, `SceneObject`, `CameraSpec` | Describe what should be compiled into MuJoCo |
| Runtime environment | `Env` | Own `mjModel`/`mjData`, registered robots, and reset hooks |
| Robot control handle | `Robot` | KDL chain, joint maps, measured ports, command ports |
| Visualization/recording | `Viewer`, `VideoRecorder` | Interactive Simulate UI and offscreen MP4 recording |
| Python package | `mj_kdl_wrapper` | Python wrappers for the same scene, robot, reset, viewer, and recorder concepts |

The important ownership rule:

- `Env` owns `mjModel` and `mjData`.
- `Robot` borrows `mjModel` and `mjData`.
- `Viewer` borrows the same model/data through the step calls.
- `VideoRecorder` owns only its EGL/rendering/ffmpeg resources; it also borrows
  model/data when recording frames.

Most examples follow this flow:

1. Build a `SceneSpec`.
2. Call `init_env()`.
3. Initialize one or more `Robot` handles from `env.model` and `env.data`.
4. Register robots with `env_add_robot()`.
5. Install `env.on_reset` if the task has object/controller state.
6. Start `init_window_sim()` or a headless loop.
7. In the loop: `step()`, `update()`, compute commands, write command ports.
8. Cleanup in reverse order: viewer/recorder, robots, env.

## 2. Start With One Robot Scene

Scenes are declared with `SceneSpec`. `timestep`, `add_floor`, and
`add_skybox` have no defaults - they are choices the wrapper refuses to
guess on your behalf. `build_scene` rejects `timestep <= 0` at runtime.
`gravity_z` defaults to Earth gravity (-9.81 m/s^2).

```cpp
#include "example_paths.hpp"
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

mj_kdl::SceneSpec scene;
scene.timestep   = 0.002;   // [s]
scene.add_floor  = true;
scene.add_skybox = true;
scene.robots.push_back(mj_kdl::RobotSpec{
    .path = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
});

mj_kdl::Env env;
if (!mj_kdl::init_env(&env, &scene)) {
    throw std::runtime_error("failed to build scene");
}
```

`Env` owns `env.model` and `env.data`. Call `mj_kdl::cleanup(&env)` when done.

You can also build raw pointers directly:

```cpp
mjModel *model = nullptr;
mjData  *data  = nullptr;
mj_kdl::build_scene(&model, &data, &scene);
```

Prefer `Env` for applications that need reset hooks or registered robots.

### What `SceneSpec` Compiles

`build_scene()` creates one MuJoCo `mjSpec`, attaches robot MJCF trees into it,
adds global scene options, adds objects and cameras, then compiles the model.
The robot list may be empty when the scene only contains objects.

Important fields:

```cpp
mj_kdl::SceneSpec scene;
scene.timestep   = 0.002;
scene.gravity_z  = -9.81;
scene.add_floor  = true;
scene.add_skybox = true;
scene.robots     = {};
scene.objects    = {};
scene.cameras    = {};
```

Use `RobotSpec::prefix` for repeated robots. The prefix is applied during MJCF
attachment so names stay unique in the compiled model:

```cpp
scene.robots.push_back(mj_kdl::RobotSpec{
    .path   = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .prefix = "left_",
    .pos    = { -0.6, 0.0, 0.0 },
});
```

Use `RobotSpec::pos` and `RobotSpec::quat` to place the robot root in the world.
`quat` is `[x, y, z, w]` and defaults to identity `{ 0, 0, 0, 1 }`.

### Cleanup Order

Use a consistent cleanup order:

```cpp
mj_kdl::cleanup(&viewer);   // stops render thread / closes window
mj_kdl::cleanup(&robot);    // clears borrowed pointers and KDL/port state
mj_kdl::cleanup(&env);      // frees mjData and mjModel
```

If you created raw `mjModel*` and `mjData*` with `build_scene()`, free them with:

```cpp
mj_kdl::destroy_scene(model, data);
```

Do not call both `cleanup(&env)` and `destroy_scene(env.model, env.data)` for the
same model/data. `Env` owns those pointers once `init_env()` succeeds.

## 3. Initialize A KDL Robot

`Robot` is the runtime handle for one controllable articulation. It stores the
borrowed `mjModel`/`mjData` pointers, KDL chain, joint-name maps, measured ports,
and command ports.

```cpp
mj_kdl::Robot robot;
if (!mj_kdl::init_robot_from_mjcf(
        &robot,
        env.model,
        env.data,
        "base_link",
        "bracelet_link")) {
    throw std::runtime_error("failed to init robot");
}

mj_kdl::env_add_robot(&env, &robot);
```

Registering with `env_add_robot()` lets `reset(&env)` sync the robot command
ports after MuJoCo data is reset.

## 4. Run A Position Control Loop

For position mode, write `jnt_pos_cmd`; `update()` copies it to MuJoCo actuator
controls.

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::POSITION;

mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot, "position control");

while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);

    for (int i = 0; i < robot.n_joints; ++i) {
        robot.jnt_pos_cmd[i] = robot.jnt_pos_msr[i];
    }
}

mj_kdl::cleanup(&viewer);
```

`init_window_sim()` starts MuJoCo Simulate in a render thread. Your loop still
owns stepping, updating control, and task logic.

## 5. Add KDL Gravity Compensation

KDL dynamics work directly from the generated chain:

```cpp
robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
KDL::JntArray q(robot.n_joints);
KDL::JntArray g(robot.n_joints);

while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);

    for (int i = 0; i < robot.n_joints; ++i) {
        q(i) = robot.jnt_pos_msr[i];
    }
    dyn.JntToGravity(q, g);

    for (int i = 0; i < robot.n_joints; ++i) {
        robot.jnt_trq_cmd[i] = g(i);
    }
}
```

This is the core pattern used by `ex_gravity_comp`.

## 6. Attach A Gripper Or Tool

Attachments are MJCF assets attached under a body, site, or frame in the
accumulated robot spec. They are applied in order, so mount -> sensor ->
gripper chains are natural.

`AttachTarget` is a tagged pair of `AttachKind { World, Body, Site, Frame }`
and an element name. Sites are the natural mount-point: the Kinova GEN3 MJCF
ships `pinch_site` on the bracelet, with its own pos/quat that already encode
the gripper offset and 180-degree flip. So the gripper attachment is just:

```cpp
mj_kdl::AttachmentSpec gripper{
    .mjcf_path = mj_kdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, "pinch_site" },
    .prefix    = "g_",
};

mj_kdl::RobotSpec arm{
    .path        = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .attachments = { gripper },
};

scene.robots.clear();
scene.robots.push_back(arm);
```

`pos` and `quat` on the spec are still available as offsets. For body and
frame parents they ride on an intermediate frame, preserving the child's
authored pose. For site parents they compose with the child root's authored
pose so authored values are never silently dropped.

When the model has no suitable site, fall back to a body name and add the
offset by hand:

```cpp
gripper.attach_to = { mj_kdl::AttachKind::Body, "bracelet_link" };
gripper.pos[2]    = -0.061525;
// 180 deg about x is exactly [x, y, z, w] = { 1, 0, 0, 0 }
gripper.quat[0]   = 1.0;
gripper.quat[3]   = 0.0;
```

Tell KDL about the attached tool when initializing the robot:

```cpp
mj_kdl::ToolFrameSpec tool{
    .tool_body = "g_base",
    .tcp_site  = "g_pinch",
};

mj_kdl::init_robot_from_mjcf(
    &robot, env.model, env.data, "base_link", "bracelet_link", "", &tool);
```

`tool_body` lumps the tool subtree inertia into the KDL chain. `tcp_site` adds a
terminal frame from a MuJoCo site.

### Attachment Chains

`attachments` is ordered. Each attachment is applied to the robot spec after all
previous attachments, so later `attach_to` values may refer to bodies introduced
by earlier attachments.

```cpp
mj_kdl::AttachmentSpec gripper{
    .mjcf_path = mj_kdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, "pinch_site" },
    .prefix    = "g_",
};

// The mug attaches to "g_base", a body the gripper attachment introduced.
mj_kdl::AttachmentSpec mug{
    .mjcf_path = mj_kdl_examples::asset("mug.xml"),
    .attach_to = { mj_kdl::AttachKind::Body, "g_base" },
    .prefix    = "mug_",
};

scene.robots.push_back(mj_kdl::RobotSpec{
    .path        = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .attachments = { gripper, mug },
});
```

Both assets ship with the package; `ex_table_pour` runs this exact chain to hold
a mug in the gripper.

For contact stability, add contact exclusions only when two attached bodies are
known to overlap structurally:

```cpp
gripper.contact_exclusions.push_back({ "bracelet_link", "g_base" });
```

Do not use exclusions to hide unstable task contacts; fix geometry, friction,
or controller gains instead.

## 7. Add Tables, Objects, And Asset Sites

`SceneObject` supports primitive objects and MJCF-backed assets. Tables are
just assets, not special first-class fields. `SceneObject::attach_to` accepts
the same tagged `AttachTarget` as `RobotSpec` and `AttachmentSpec`. Inside
`build_scene` the order is decorations -> objects (declaration order) ->
robots -> cameras, so a robot or a later object may reference any prior
object via its name or one of its sites.

`SceneObject` has no defaults for `shape`, `size`, `rgba`, `mass`, or
`friction`. For MJCF-backed objects (when `mjcf_path` is set) those fields
are ignored at runtime, so leaving them zero-initialised is fine. For
primitives, `build_scene` runs explicit checks:

- `shape == Shape::Unspecified` -> error, the object is skipped.
- `size[i] <= 0` for the relevant dimensions of the shape -> error, skipped.
- `mass <= 0` on a non-fixed primitive -> error, skipped.

```cpp
mj_kdl::SceneObject table{
    .name      = "table",
    .mjcf_path = mj_kdl_examples::asset("table.xml"),
    .pos       = { 0.0, 0.0, 0.7 },
    .fixed     = true,
};

scene.objects.push_back(table);

// Primitives require shape, size, rgba, mass, and friction.
scene.objects.push_back(mj_kdl::SceneObject{
    .name     = "cube",
    .shape    = mj_kdl::Shape::BOX,
    .size     = { 0.03, 0.03, 0.03 },                 // half-extents [m]
    .pos      = { 0.35, 0.05, 0.73 },                 // free cubes stay world-anchored
    .rgba     = { 0.1f, 0.3f, 1.0f, 1.0f },
    .mass     = 0.1,                                  // [kg]
    .condim   = mj_kdl::Condim::Torsional,            // friction model
    .friction = { 0.8, 0.02, 0.001 },                 // [slide, spin, roll]
});
```

`Condim` is a typed enum (`Tangential` = 3, `Torsional` = 4, `Rolling` = 6)
matching MuJoCo's contact-dimensionality integers. Default is `Tangential`.

`scene.robots` can be empty. Object-only scenes still compile and can be opened
in the Simulate UI with `init_window_sim(&viewer, model, data, "object scene")`.

After `build_scene`, an MJCF-backed `SceneObject` exposes its root body in
the compiled scene under `obj.name` (i.e. the asset's internal root body name
is rewritten so callers never need to know it). All other elements (sites,
geoms, joints, child bodies) keep the `obj.name + "_"` prefix.

`scene_object_site_name(obj, "site_name")` returns the compiled name of a
site authored inside the asset, for places that need a string at runtime:

```cpp
const std::string site = mj_kdl::scene_object_site_name(table, "table_top");
KDL::Frame world_T_table_top;
mj_kdl::get_site_frame(env.model, env.data, site.c_str(), &world_T_table_top);
```

Combined, this lets a robot sit on the tabletop without hand-threading
heights:

```cpp
const std::string mount = mj_kdl::scene_object_site_name(table, "table_top");

scene.robots.push_back(mj_kdl::RobotSpec{
    .path      = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, mount.c_str() },
});
```

A fixed object can also attach to the table body directly:

```cpp
scene.objects.push_back(mj_kdl::SceneObject{
    .name      = "fixture",
    .mjcf_path = "fixture.xml",
    .attach_to = { mj_kdl::AttachKind::Body, "table" },
    .fixed     = true,
});
```

MuJoCo restricts freejoints to top-level bodies, so non-fixed primitives and
asset roots with a freejoint must use `AttachKind::World` (the default).
`mj_compile` reports the violation if this rule is broken.

## 8. Add Cameras

Add fixed scene cameras through `SceneSpec::cameras`. `CameraSpec` requires
`pos` and `fovy`; `quat` is `[x, y, z, w]` and defaults to identity
`{ 0, 0, 0, 1 }`.

```cpp
scene.cameras.push_back(mj_kdl::CameraSpec{
    .name = "front",
    .pos  = { 0.0, -0.8, 1.45 },                  // required
    .quat = { 0.300706, 0.0, 0.0, 0.953717 },     // 35 deg about x, tilt down
    .fovy = 45.0,                                 // required
});
```

After compile, all cameras are visible to MuJoCo and include:

- robot MJCF cameras,
- cameras added through `SceneSpec::cameras`,
- cameras in MJCF object assets.

List them:

```cpp
for (const auto &name : mj_kdl::get_camera_names(env.model)) {
    std::cout << name << "\n";
}
```

Use one in the viewer or recorder:

```cpp
mj_kdl::use_camera(&viewer, env.model, "front");
mj_kdl::use_camera(&recorder, env.model, "front");
```

The Simulate UI also has its own live camera selector in the Rendering panel.

## 9. Write Production Reset Hooks

`reset(&env)` resets MuJoCo, runs your hook, forwards dynamics, then syncs all
registered robots so stale commands do not hit the first post-reset step.

```cpp
KDL::JntArray q_home(robot.n_joints);
double cube_start[3] = { 0.35, 0.05, 0.73 };

env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    mj_kdl::set_joint_pos(&robot, q_home, false);
    mj_kdl::set_body_pose(ctx->model, ctx->data, "cube", cube_start);
    task_state = TaskState::HOME;
    episode_step = 0;
};

mj_kdl::ResetOptions opts{
    .keyframe     = 0,
    .use_keyframe = true,
};

mj_kdl::ResetInfo info = mj_kdl::reset(&env, &opts);
```

Put robot, object, controller, randomization, and task-specific state in the
hook. Do not hide reset work in the control loop.

### Reset Lifecycle

`reset(&env)` performs the reset in this order:

1. Reset MuJoCo data to keyframe/default state.
2. Create a `ResetContext`.
3. Call `env.on_reset`, if provided.
4. Call `mj_forward()`.
5. Sync every registered `Robot`:
   - `jnt_pos_cmd` becomes the measured joint position,
   - `jnt_trq_cmd` is cleared,
   - stale applied forces are cleared.

That order is deliberate. User hooks restore task state after the low-level
MuJoCo reset, and robot ports are synchronized after the hook so the first
post-reset update does not apply stale commands.

Use `ResetContext` when your hook needs direct MuJoCo access:

```cpp
env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    const double q_identity[4] = { 1.0, 0.0, 0.0, 0.0 };
    const double cube_pos[3]   = { 0.35, 0.05, 0.73 };
    mj_kdl::set_body_pose(ctx->model, ctx->data, "cube", cube_pos, q_identity);

    controller_integral.assign(robot.n_joints, 0.0);
    task_state = TaskState::HOME;
};
```

For randomized starts, generate random values in the hook and write them to
MuJoCo before `reset()` returns.

## 10. Use The Simulate UI

The full viewer path is:

```cpp
mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot, "task");

while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);
    // control...
}
```

Useful wrapper-specific controls:

| Control | Behavior |
|---------|----------|
| `,` | slow the wrapper real-time factor |
| `.` | speed the wrapper real-time factor |
| Simulation -> `RTF` | current wrapper real-time factor |
| Simulation -> Recorder | path, camera, resolution, FPS, start/stop, status |

Recorder camera options include `Current`, `Free`, `Tracking`, and every fixed
camera compiled into the model.

The right panel adds debug sections:

| Section | Behavior |
|---------|----------|
| `Frames` | per-body/site coordinate triads (RViz-style), with a `Scale` slider |
| `Trace` | `Trace EE` draws a trail following the robot TCP site |
| `Perturb` | `Body` shows the double-clicked selection; `Drag` = `Camera`/`Force`/`Torque`, then left-drag to apply (or `Ctrl`+right/left-drag) |

The `Equality` and `Group enable` sections are hidden by default; rebuild with
`-DSHOW_EQUALITY_PANEL=ON` / `-DSHOW_GROUP_PANEL=ON` to restore them. These
debug geoms appear in the live viewer only, not in recorded MP4s.

### Simulate UI Recorder Controls

The recorder controls live in the left Simulation panel:

| Field | Meaning |
|-------|---------|
| `Path` | Output MP4 path, relative to the process working directory unless absolute |
| `Camera` | `Current`, `Free`, `Tracking`, or any compiled fixed camera |
| `Resolution` | `360p`, `480p`, `720p`, or `1080p` |
| `FPS` | Recording frame rate |
| `Start rec` | Opens EGL recorder and starts feeding frames |
| `Stop rec` | Closes ffmpeg and finalizes the MP4 |
| `Rec` | `idle`, `recording`, or `failed` |

`Current` means the recorder follows the Simulate viewer camera. A fixed camera
records from that camera even if the live viewer is moved elsewhere.

When recording stops successfully, the terminal prints:

```text
[mj_kdl] recording saved to recording.mp4
```

If the status changes to `failed`, check that `ffmpeg` is installed and the path
is writable.

### Draw A Live Trajectory Trace Overlay

`init_window_sim()` owns a user scene that the render thread merges into every
frame. Two helpers let you draw your own line geometry into it -- the built-in
use is a live polyline of recent end-effector positions, which makes it obvious
at a glance whether the EE is tracking a commanded path:

```cpp
#include <deque>

std::deque<KDL::Vector> trace;          // ring buffer of recent EE points
constexpr size_t kTraceMax = 4096;      // bounded by the user-scene geom budget

while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);
    // ... run your controller; advance the EE ...

    // FK gives the EE in the chain-root frame; the overlay renders in world
    // frame, so lift it through the (fixed) base body pose -- cache this once.
    static KDL::Frame world_T_base;
    static bool have_base =
        mj_kdl::get_body_frame(robot.model, robot.data, "base_link", &world_T_base);

    KDL::Frame ee_base;
    fk_solver.JntToCart(q, ee_base);
    trace.push_back(world_T_base * ee_base.p);
    if (trace.size() > kTraceMax) trace.pop_front();

    mj_kdl::clear_trace(&viewer);                 // reset the overlay each frame
    static constexpr float kOrange[4] = {1.0f, 0.5f, 0.1f, 1.0f};
    for (size_t i = 1; i < trace.size(); ++i)
        mj_kdl::add_trace_segment(&viewer, trace[i - 1], trace[i], kOrange);
}
```

Key points:

- Both helpers are **no-ops when `viewer` is not backed by an `init_window_sim()`
  window** (e.g. headless runs), so the same loop compiles and runs unchanged
  with no display -- guard the bookkeeping with `if (!headless)` to skip the
  allocation entirely.
- `add_trace_segment()` is thread-safe and silently drops segments once the
  user-scene geom buffer is full (8192 geoms). Keep your ring buffer at or below
  that to avoid a truncated trace.
- Pass `nullptr` for `rgba` to use the default warm orange.
- Points are in **world frame**. Robot FK is in the chain-root frame, so compose
  with the base body's world pose (constant for a fixed base; read it once).

The motion-spec code generator wires this up automatically: declare a `TRACE`
block in the `ENVIRONMENT` of a `.robmot` model (`enabled`, `length`, `color`)
and every generated demo renders the trace with no hand-written loop code.

## 11. Record Video

Interactive recording is available in the Simulate UI:

1. Open the Simulation panel.
2. Scroll to Recorder.
3. Set `Path`, `Camera`, `Resolution`, and `FPS`.
4. Press `Start rec`.
5. Press `Stop rec`.

The terminal prints:

```text
[mj_kdl] recording saved to <path>
```

Headless recording uses the same `VideoRecorder` API:

```cpp
mj_kdl::VideoRecorder recorder;
mj_kdl::init_video_recorder(
    &recorder, env.model, "episode.mp4", mj_kdl::VideoResolution::R1080p, 60);

for (int i = 0; i < 3000; ++i) {
    mj_step(env.model, env.data);
    if (i % 4 == 0) {
        mj_kdl::record_frame(&recorder, env.model, env.data);
    }
}

mj_kdl::cleanup(&recorder);
```

## 12. Build A Tabletop Pick-Place Example

This section assembles the pieces into a complete pick-place task. The goal is
to pick a cube from one table location, move it to another location, open the
gripper, and retreat. The full implementation is `src/examples/ex_table_pick_place.cpp`;
the code below shows the structure you should reproduce in your own application.

The application has five parts:

1. Build a scene with an arm, gripper, table asset, cube, and camera.
2. Read the table site to compute reliable tabletop coordinates.
3. Build KDL FK/IK/dynamics solvers.
4. Define a reset hook and state machine.
5. Run torque impedance control against state-specific IK targets.

### 12.1 Build The Scene

Start with the gripper attachment. The Kinova `pinch_site` already encodes
the tool offset and 180-degree flip, so no `pos`/`quat` are needed:

```cpp
mj_kdl::AttachmentSpec gripper{
    .mjcf_path = mj_kdl_examples::asset("robotiq_2f85/2f85.xml"),
    .attach_to = { mj_kdl::AttachKind::Site, "pinch_site" },
    .prefix    = "g_",
};
```

Add the table as an MJCF-backed `SceneObject`. The table asset origin is the
tabletop surface center, so placing it at `z = 0.7` makes the top surface `0.7 m`.

```cpp
mj_kdl::SceneObject table{
    .name      = "table",
    .mjcf_path = mj_kdl_examples::asset("table.xml"),
    .pos       = { 0.0, 0.0, 0.7 },
    .fixed     = true,
};
```

Add a free cube. For a box, `size` is half-extents, so the world-frame
center z is `surface_z + half_height`. Free objects must stay world-anchored
(MuJoCo restricts freejoints to top-level bodies).

```cpp
constexpr double kSurfaceZ = 0.7;
constexpr double kCubeHalf = 0.025;

mj_kdl::SceneObject cube{
    .name     = "cube",
    .shape    = mj_kdl::Shape::BOX,
    .size     = { kCubeHalf, kCubeHalf, kCubeHalf },
    .pos      = { 0.35, 0.10, kSurfaceZ + kCubeHalf },
    .rgba     = { 0.1f, 0.25f, 1.0f, 1.0f },
    .mass     = 0.1,
    .condim   = mj_kdl::Condim::Torsional,
    .friction = { 0.8, 0.02, 0.001 },
};
```

Assemble the scene. Set `timestep`, `add_floor`, and `add_skybox`
explicitly, then mount the arm on the table's `table_top` site instead of
writing `surface_z` into the robot position by hand:

```cpp
mj_kdl::SceneSpec scene;
scene.timestep   = 0.002;
scene.add_floor  = true;
scene.add_skybox = true;
scene.objects.push_back(table);
scene.objects.push_back(cube);

const std::string mount =
    mj_kdl::scene_object_site_name(table, "table_top");

scene.robots.push_back(mj_kdl::RobotSpec{
    .path        = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
    .attach_to   = { mj_kdl::AttachKind::Site, mount.c_str() },
    .attachments = { gripper },
});
scene.cameras.push_back(mj_kdl::CameraSpec{
    .name = "task",
    .pos  = { 0.1, -0.9, 1.45 },
    // extrinsic XYZ (35, 0, 5) deg: tilt down, yaw slightly right
    .quat = { 0.300420, 0.013117, 0.041601, 0.952809 },
    .fovy = 45.0,
});

mj_kdl::Env env;
mj_kdl::init_env(&env, &scene);
```

### 12.2 Read Table Sites Instead Of Hardcoding Geometry

The table asset defines a `table_top` site. Because MJCF-backed objects are
prefixed when attached, get the compiled site name through the helper:

```cpp
KDL::Frame world_T_table_top;
const std::string table_top_site =
    mj_kdl::scene_object_site_name(table, "table_top");

if (!mj_kdl::get_site_frame(
        env.model, env.data, table_top_site.c_str(), &world_T_table_top)) {
    throw std::runtime_error("table_top site not found");
}

const double surface_z = world_T_table_top.p.z();
```

Now define task points from the table surface:

```cpp
KDL::Vector pick_pos(0.35, 0.10, surface_z + 0.02);
KDL::Vector place_pos(0.20, -0.18, surface_z + 0.02);

KDL::Vector approach_offset(0.0, 0.0, 0.16);
KDL::Vector lift_offset(0.0, 0.0, 0.22);
```

This makes the task robust if you move the table asset or swap it for another
asset with the same site convention.

### 12.3 Initialize Robot, Tool, And Solvers

Initialize the robot with gripper inertia and TCP site:

```cpp
mj_kdl::Robot robot;
mj_kdl::ToolFrameSpec tool{
    .tool_body = "g_base",
    .tcp_site  = "g_pinch",
};

mj_kdl::init_robot_from_mjcf(
    &robot, env.model, env.data, "base_link", "bracelet_link", "", &tool);
mj_kdl::env_add_robot(&env, &robot);

robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
```

Create KDL solvers:

```cpp
KDL::ChainFkSolverPos_recursive fk(robot.chain);
KDL::ChainIkSolverVel_pinv ik_vel(robot.chain);
KDL::ChainIkSolverPos_NR_JL ik_pos(
    robot.chain, q_min, q_max, fk, ik_vel, 100, 1e-5);
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
```

`q_min` and `q_max` come from `robot.joint_limits`. Keep them in `KDL::JntArray`
so IK respects the compiled model limits.

### 12.4 Generate IK Waypoints

Use one orientation for the gripper and solve a sequence of poses:

```cpp
KDL::Rotation grasp_R = KDL::Rotation::RPY(M_PI, 0.0, M_PI / 2.0);

KDL::Frame pick_above(grasp_R, pick_pos + approach_offset);
KDL::Frame pick_frame(grasp_R, pick_pos);
KDL::Frame lift_frame(grasp_R, pick_pos + lift_offset);
KDL::Frame place_above(grasp_R, place_pos + approach_offset);
KDL::Frame place_frame(grasp_R, place_pos);
KDL::Frame retreat_frame(grasp_R, place_pos + lift_offset);
```

Solve each waypoint seeded from the previous solution:

```cpp
auto solve = [&](const KDL::Frame &target, const KDL::JntArray &seed) {
    KDL::JntArray q_out(robot.n_joints);
    if (ik_pos.CartToJnt(seed, target, q_out) < 0) {
        throw std::runtime_error("IK failed");
    }
    return q_out;
};

KDL::JntArray q_home = measured_home(robot);
KDL::JntArray q_pick_above = solve(pick_above, q_home);
KDL::JntArray q_pick       = solve(pick_frame, q_pick_above);
KDL::JntArray q_lift       = solve(lift_frame, q_pick);
KDL::JntArray q_place_above = solve(place_above, q_lift);
KDL::JntArray q_place      = solve(place_frame, q_place_above);
KDL::JntArray q_retreat    = solve(retreat_frame, q_place);
```

In production code, print the target names when IK fails. It saves time when a
single pose is outside the workspace or has an impossible orientation.

### 12.5 Define The State Machine

Keep state configuration data-driven. Each row says where to move, how long to
interpolate, how long to wait before forcing transition, and what the gripper
should do.

```cpp
enum class TaskState {
    HOME,
    PICK_ABOVE,
    PICK,
    CLOSE,
    LIFT,
    PLACE_ABOVE,
    PLACE,
    OPEN,
    RETREAT,
    HOLD,
};

struct StateConfig {
    TaskState state;
    KDL::JntArray target;
    double duration;
    double timeout;
    double settle_tol;
    double gripper_cmd;
};

std::vector<StateConfig> plan = {
    { TaskState::HOME,        q_home,        1.0, 2.5, 0.08, 0.0   },
    { TaskState::PICK_ABOVE,  q_pick_above,  2.0, 4.0, 0.08, 0.0   },
    { TaskState::PICK,        q_pick,        1.5, 3.5, 0.06, 0.0   },
    { TaskState::CLOSE,       q_pick,        0.8, 1.5, 0.00, 255.0 },
    { TaskState::LIFT,        q_lift,        2.0, 4.0, 0.08, 255.0 },
    { TaskState::PLACE_ABOVE, q_place_above, 2.0, 4.0, 0.08, 255.0 },
    { TaskState::PLACE,       q_place,       1.5, 3.0, 0.06, 255.0 },
    { TaskState::OPEN,        q_place,       0.8, 1.5, 0.00, 0.0   },
    { TaskState::RETREAT,     q_retreat,     1.5, 3.0, 0.08, 0.0   },
    { TaskState::HOLD,        q_retreat,     1.0, 1.0, 0.00, 0.0   },
};
```

On entry to a state, capture the measured joint position. Interpolate from that
entry pose to the state target:

```cpp
KDL::JntArray q_enter = current_q(robot);
double t_enter = env.data->time;

double alpha = std::clamp(
    (env.data->time - t_enter) / cfg.duration, 0.0, 1.0);

KDL::JntArray q_des(robot.n_joints);
for (int i = 0; i < robot.n_joints; ++i) {
    q_des(i) = q_enter(i) + alpha * (cfg.target(i) - q_enter(i));
}
```

Transition when either the duration has elapsed and the joint error is small, or
the timeout is reached:

```cpp
bool settled = joint_error(q_des, current_q(robot)) < cfg.settle_tol;
bool duration_done = env.data->time - t_enter >= cfg.duration;
bool timed_out = env.data->time - t_enter >= cfg.timeout;

if ((duration_done && settled) || timed_out) {
    advance_to_next_state();
}
```

### 12.6 Apply Torque Impedance

Use KDL gravity plus joint PD:

```cpp
KDL::JntArray q(robot.n_joints);
KDL::JntArray dq(robot.n_joints);
KDL::JntArray gravity(robot.n_joints);

for (int i = 0; i < robot.n_joints; ++i) {
    q(i)  = robot.jnt_pos_msr[i];
    dq(i) = robot.jnt_vel_msr[i];
}

dyn.JntToGravity(q, gravity);

for (int i = 0; i < robot.n_joints; ++i) {
    const double kp = 120.0;
    const double kd = 18.0;
    robot.jnt_trq_cmd[i] =
        gravity(i) + kp * (q_des(i) - q(i)) - kd * dq(i);
}
```

The gripper actuator is model-specific. For the Robotiq Menagerie model, examples
write the gripper command directly to its actuator control after `update()`:

```cpp
int gripper_act = mj_name2id(env.model, mjOBJ_ACTUATOR, "g_fingers_actuator");
if (gripper_act >= 0) {
    env.data->ctrl[gripper_act] = cfg.gripper_cmd;
}
```

### 12.7 Reset The Pick-Place Task

Reset must restore both the robot and task objects:

```cpp
const double cube_start[3] = { 0.35, 0.10, surface_z + kCubeHalf };

env.on_reset = [&](mj_kdl::ResetContext *ctx) {
    mj_kdl::set_joint_pos(&robot, q_home, false);
    mj_kdl::set_body_pose(ctx->model, ctx->data, "cube", cube_start);

    current_state = TaskState::HOME;
    state_index = 0;
    q_enter = q_home;
    t_enter = 0.0;
};
```

Because `reset(&env)` syncs registered robot command ports after this hook, the
first control step after reset starts from the reset pose without stale torque or
position commands.

### 12.8 Run With Viewer And Recorder

Start the Simulate UI:

```cpp
mj_kdl::Viewer viewer;
mj_kdl::init_window_sim(&viewer, &robot, "table pick-place");
mj_kdl::use_camera(&viewer, env.model, "task");

while (mj_kdl::step(&robot)) {
    mj_kdl::update(&robot);
    run_state_machine();
    apply_impedance_command();
}
```

Use `,` and `.` to slow down or speed up the wrapper real-time factor. In the
Simulation panel, the Recorder subsection can write an MP4 using `Current`,
`Free`, `Tracking`, or any compiled fixed camera.

### 12.9 What To Validate

Before treating the example as working, check:

- IK waypoints are reachable before the simulation loop starts.
- The cube is a free body and `set_body_pose()` can reset it.
- The gripper command closes enough to hold the cube but does not destabilize the arm.
- Joint torque limits are reasonable for the gains.
- The state machine can recover after pressing Reset in the Simulate UI.
- Headless mode runs a fixed number of steps and reports cube final position.

This is the architecture used by `ex_table_pick_place`: a small scene spec, a
reset hook, precomputed IK waypoints, a data-driven state table, and one control
law used consistently across states.

## 13. Build A Multi-Robot Scene

Use prefixes to disambiguate the second robot:

```cpp
scene.robots = {
    mj_kdl::RobotSpec{
        .path = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
        .pos  = { -0.7, 0.0, 0.0 },
        .attachments = { gripper },
    },
    mj_kdl::RobotSpec{
        .path   = mj_kdl_examples::menagerie_model("kinova_gen3/gen3.xml"),
        .prefix = "r2_",
        .pos    = { 0.7, 0.0, 0.0 },
        .attachments = { gripper },
    },
};
```

Then initialize two robot handles:

```cpp
mj_kdl::Robot left;
mj_kdl::Robot right;
mj_kdl::init_robot_from_mjcf(&left, env.model, env.data, "base_link", "bracelet_link", "", &tool);
mj_kdl::init_robot_from_mjcf(&right, env.model, env.data, "base_link", "bracelet_link", "r2_", &tool);
mj_kdl::env_add_robot(&env, &left);
mj_kdl::env_add_robot(&env, &right);
```

Each robot gets its own KDL chain and command ports, while both share the same
MuJoCo model/data.

## 14. Grow Into Task Examples

The included examples show how these pieces combine:

- `ex_table_scene`: table asset, primitive objects, sites, cameras, reset hook.
- `ex_pick`: IK waypoints, state machine, torque impedance.
- `ex_table_pick_place`: tabletop pick/place using table asset sites.
- `ex_table_pour`: gripper-held bottle asset, free particles, receiver asset.
- `ex_dual_arm`: two prefixed robots in one scene.
- `ex_record`: headless MP4 recording.

Read `../examples.md` for behavior summaries and expected outputs.

## 15. Modify A Running Scene

For occasional changes, use the scene add/remove helpers. They rebuild the model,
so this is for task setup and coarse changes, not per-frame object spawning.

Raw model/data form:

```cpp
mj_kdl::SceneObject obstacle{
    .name  = "obstacle",
    .shape = mj_kdl::Shape::CYLINDER,
    .size  = { 0.04, 0.12, 0.0 },                 // {radius, half_length, 0}
    .pos   = { 0.3, -0.2, 0.82 },
    .rgba  = { 0.6f, 0.6f, 0.6f, 1.0f },
    .fixed = true,                                // fixed obstacles tolerate mass = 0
};

mj_kdl::scene_add_object(&model, &data, &scene, obstacle);
// model/data pointers were replaced; reinitialize Robot handles.
```

`Env` form:

```cpp
mj_kdl::scene_add_object(&env, obstacle);
// env.model/env.data and registered Robot model/data pointers are updated.
```

After a rebuild, any cached MuJoCo IDs may be invalid. Recompute body IDs, joint
IDs, site names, and KDL solvers that depend on the old model.

## 16. Debugging Checklist

Use this checklist when a scene behaves incorrectly:

| Symptom | Check |
|---------|-------|
| Robot does not move in position mode | Model has actuators and `kdl_to_mj_ctrl[i] >= 0` |
| First step after reset jumps | Robot is registered with `env_add_robot()` and reset uses `reset(&env)` |
| KDL gravity is wrong with a tool | `ToolFrameSpec::tool_body` points at the tool subtree root |
| TCP frame is wrong | `ToolFrameSpec::tcp_site` names an authored MuJoCo site |
| Object asset site not found | Use `scene_object_site_name(object, "site")` to account for prefixes |
| Recorder fails | `BUILD_RECORDER=ON`, EGL available, ffmpeg installed, output path writable |
| Camera missing in recorder list | Camera must exist in the compiled `mjModel` (`get_camera_names(model)`) |

## 17. Development Checks

Run the default build:

```bash
cmake --build build --target mj_kdl_wrapper -j$(nproc)
```

Run clang-tidy on wrapper code:

```bash
clang-tidy -p build src/mj_kdl_wrapper.cpp
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

The vendored `src/simulate_ui/simulate.cc` is MuJoCo sample UI code with
small wrapper UI additions. It is intentionally excluded from clang-tidy style
cleanup so local changes stay reviewable against upstream.
