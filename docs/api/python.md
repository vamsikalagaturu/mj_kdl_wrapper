# Python Bindings API Guide {#page_api_python}

This page collects the Python wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated stubs in
`python/mj_kdl_wrapper/*.pyi`.

Coming from 0.2.x? Placement orientation moved from `.euler` to `.quat`
`[x, y, z, w]`; see [Migrating from 0.2.x](@ref sec_migrate_quat).

The Python package exposes the same scene, robot, reset, viewer, and recorder
concepts as the C++ wrapper. It owns MuJoCo `mjModel`/`mjData` through `Scene`
or `Env` and returns KDL values through the upstream `PyKDL` module.

The wheel bundles `PyKDL` as a top-level extension module built from the same
Orocos KDL fork as the wrapper. `import PyKDL` works after installation, but
`PyKDL` does not appear in `pip list` / `uv pip list` because it is not installed
as a separate Python distribution with its own `.dist-info`.

## Model Paths

`mj_kdl_wrapper.menagerie.model_path(name)` resolves bundled-example model
paths without hard-coding a checkout location. It checks overrides first, then
known local or cached MuJoCo Menagerie checkouts:

1. `MJ_KDL_MODEL` / `MJ_KDL_GRIPPER` - per-model file overrides.
2. `MJ_KDL_MENAGERIE` - a MuJoCo Menagerie checkout root.
3. The user cache `~/.cache/mj_kdl_wrapper/menagerie`, populated by
   `mj-kdl-fetch-menagerie`.

`menagerie.asset_path(rel)` resolves bundled assets (gripper, table, mug) the
same way: an optional per-asset env override, otherwise the user cache
`~/.cache/mj_kdl_wrapper/assets`. `mj-kdl-fetch-menagerie` populates both, and
the same cache backs the C++ examples (see the C++ guide).

**Overrides:** the example scripts wire these per-file env vars -- `MJ_KDL_MODEL`
(arm), `MJ_KDL_GRIPPER`, `MJ_KDL_TABLE`, `MJ_KDL_BOTTLE`, `MJ_KDL_RECEIVER`. Each
must point at an existing file or resolution raises a clear error.
`MJ_KDL_MENAGERIE` overrides the Menagerie checkout root.

For other MJCF sources, set the relevant environment variable or assign
`RobotSpec.path` directly.

## Load From MJCF

`SceneSpec` has no defaults for `timestep`, `add_floor`, or `add_skybox`.
Those are explicit scene choices. `Scene.build()` rejects `timestep <= 0`.
`spec.robots` may be empty; object-only scenes are valid.

```python
import mj_kdl_wrapper as mjk

spec = mjk.SceneSpec()
spec.timestep = 0.002
spec.add_floor = True
spec.add_skybox = True

robot_spec = mjk.RobotSpec()
robot_spec.path = mjk.menagerie.model_path("kinova_gen3")
spec.robots = [robot_spec]

scene = mjk.Scene.build(spec)
```

For an object-only scene, put MJCF or primitive objects in `spec.objects` and
leave `spec.robots` empty:

```python
cabinet = mjk.SceneObject()
cabinet.name = "cabinet"
cabinet.mjcf_path = mjk.menagerie.asset_path("cabinet/cabinet.xml")
cabinet.fixed = True

spec.objects = [cabinet]
scene = mjk.Scene.build(spec)
```

`Scene` owns the compiled MuJoCo model/data. Call `close()` when you want to
release native resources deterministically. Use `time()`, `timestep()`,
`step()`, and `step_n(n)` when you want to advance the scene without a `Robot`
control loop.

```python
scene.step_n(10)
print(scene.time(), scene.timestep())
scene.save_xml("combined_scene.xml")
scene.save_binary("combined_scene.mjb")
```

Set wrapper log verbosity globally when debugging scene construction:

```python
mjk.set_log_level(mjk.LogLevel.INFO)
assert mjk.get_log_level() == mjk.LogLevel.INFO
print(mjk.mujoco_version())
```

`mjk.__version__` is the Python package version. `mjk.__mujoco_version__` is the
MuJoCo version the extension was built against.

## Init A KDL Chain

```python
robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")

robot.ctrl_mode = mjk.CtrlMode.POSITION
robot.jnt_pos_cmd = [0.0] * robot.n_joints
robot.update()
robot.step()
```

After init, `joint_names`, `joint_limits`, and all joint port vectors are in KDL
chain order. Assignment to command ports must match `n_joints`; the bindings
raise `ValueError`/`RuntimeError` for wrong sizes or closed handles.

When a tool or gripper is attached, pass `ToolFrameSpec` so the KDL chain uses
the TCP site and includes the tool transform:

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base"
tool.tcp_site = "g_pinch"

robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
assert robot.has_tcp_frame
```

For a wrist force-torque sensor, attach the sensor MJCF first, then attach the
gripper to a site exported by that sensor asset:

```python
ft_sensor = mjk.AttachmentSpec()
ft_sensor.mjcf_path = mjk.menagerie.asset_path("ft_sensor.xml")
ft_sensor.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")

gripper = mjk.AttachmentSpec()
gripper.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml")
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "wrist_ft_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = mjk.menagerie.model_path("kinova_gen3")
robot_spec.attachments = [ft_sensor, gripper]
```

Then register the logical force-torque sensor on the same tool spec. MuJoCo
models one FT sensor as separate `<force>` and `<torque>` sensors; the wrapper
returns a `PyKDL.Wrench`.

```python
ft = mjk.ForceTorqueSensorSpec()
ft.name = "wrist_ft"          # resolves wrist_ft_force + wrist_ft_torque
ft.frame_site = "wrist_ft_site"

tool.ft_sensors = [ft]
robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
robot.update()
wrench = robot.ft_sensor("wrist_ft")
```

## Attach MJCF Bodies

`AttachTarget` is a tagged pair of `AttachKind` and an element name. The Kinova
GEN3 MJCF exports `pinch_site` on the bracelet, so a Robotiq gripper can attach
without manual pose offsets:

```python
gripper = mjk.AttachmentSpec()
gripper.mjcf_path = "assets/robotiq_2f85/2f85.xml"
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = mjk.menagerie.model_path("kinova_gen3")
robot_spec.attachments = [gripper]
```

Optional `pos` and `quat` on the attachment spec are composed with the parent
site pose, so small calibration offsets can stay local to the attachment.
`quat` is `[x, y, z, w]` and defaults to identity `[0, 0, 0, 1]`. Assign the
whole list -- these properties return a copy, so `gripper.pos[2] = 0.005` is
silently discarded:

```python
gripper.pos = [0.0, 0.0, 0.005]                    # +5 mm along the tool z
gripper.quat = [0.0, 0.0, 0.130526, 0.991445]      # +15 deg about the tool z
```

Chains are supported by appending multiple `AttachmentSpec` objects in order.
Each later attachment may reference a body, site, or frame added by an earlier
attachment.

## Multi-Robot Scene

Use prefixes to keep MuJoCo names distinct when loading the same MJCF more than
once. The prefix is also passed when creating the corresponding `Robot` handle.

```python
left = mjk.RobotSpec()
left.path = mjk.menagerie.model_path("kinova_gen3")
left.pos = [-0.5, 0.0, 0.0]

right = mjk.RobotSpec()
right.path = mjk.menagerie.model_path("kinova_gen3")
right.prefix = "r2_"
right.pos = [0.5, 0.0, 0.0]

spec.robots = [left, right]
scene = mjk.Scene.build(spec)

robot1 = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
robot2 = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", prefix="r2_")
```

## Table And Scene Objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so robots and
objects can be mounted to sites or bodies created earlier in the scene spec.
Build order is decorations, objects in declaration order, robots, then cameras.

For primitive objects, `shape`, `size`, `rgba`, `mass`, and `friction` are
required. For MJCF-backed objects, `mjcf_path` takes precedence and primitive
geometry fields are ignored at runtime.

```python
table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = "assets/table.xml"
table.pos = [0.0, 0.0, 0.7]
table.fixed = True

mount = mjk.scene_object_site_name(table, "table_top")

robot_spec = mjk.RobotSpec()
robot_spec.path = mjk.menagerie.model_path("kinova_gen3")
robot_spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, mount)

cube = mjk.SceneObject()
cube.name = "red_cube"
cube.shape = mjk.Shape.BOX
cube.size = [0.03, 0.03, 0.03]
cube.pos = [0.35, 0.10, 0.73]
cube.rgba = [1.0, 0.0, 0.0, 1.0]
cube.mass = 0.1
cube.condim = mjk.Condim.Torsional
cube.friction = [0.8, 0.02, 0.001]

spec.objects = [table, cube]
spec.robots = [robot_spec]
scene = mjk.Scene.build(spec)
```

MuJoCo restricts free joints to top-level bodies, so a non-fixed primitive with
a free joint must stay world-anchored.

## Cameras, Actuators, And Poses

Add fixed world cameras through `SceneSpec.cameras`. `pos` and `fovy` are
required; `quat` is `[x, y, z, w]` and defaults to identity `[0, 0, 0, 1]`.

```python
cam = mjk.CameraSpec()
cam.name = "overview"
cam.pos = [1.8, -2.0, 1.4]
cam.fovy = 45.0
spec.cameras = [cam]

scene = mjk.Scene.build(spec)
print(scene.camera_names())
```

`SimulateViewer.use_camera(name)` and `VideoRecorder.use_camera(name)` switch to
a fixed camera. Pass `""` to return to the free camera.

Use `body_frame()` and `site_frame()` to read world poses as `PyKDL.Frame`.
`Scene.set_body_pose()` teleports a free body; Python quaternions for this
method use `xyzw` order and are converted before calling MuJoCo.

```python
tcp = scene.site_frame("g_pinch")
scene.set_body_pose("red_cube", [0.45, 0.0, 0.75], [0.0, 0.0, 0.0, 1.0])
```

For named actuators that are not part of a `Robot` joint mapping, use the direct
actuator helpers:

```python
if scene.has_actuator("finger"):
    scene.set_actuator_ctrl("finger", 0.25)
    print(scene.actuator_ctrl("finger"))
```

## PyKDL Interop

The binding layer constructs standard PyKDL objects for KDL return types.
Downstream code can use the regular PyKDL solvers:

```python
import PyKDL as kdl

chain = robot.kdl_chain()
fk = kdl.ChainFkSolverPos_recursive(chain)
q = kdl.JntArray(robot.n_joints)
for i, value in enumerate(robot.jnt_pos_msr):
    q[i] = value

tcp = kdl.Frame()
fk.JntToCart(q, tcp)
```

`Robot.set_joint_pos()` and `Robot.fk_frame(q)` accept either Python sequences
or `PyKDL.JntArray`. `Scene.body_frame()` and `Scene.site_frame()` return
`PyKDL.Frame`.

`Robot.gravity_torques(gravity_z=-9.81)` is a convenience wrapper around
`KDL::ChainDynParam::JntToGravity()` using the robot's measured positions.
For full dynamics, get the chain with `kdl_chain()` and construct the PyKDL
solver you need.

## Control Loop

```python
robot.ctrl_mode = mjk.CtrlMode.TORQUE

while robot.step():
    robot.update()
    robot.jnt_trq_cmd = robot.gravity_torques()
```

`update()` reads MuJoCo state into `jnt_pos_msr`, `jnt_vel_msr`, and
`jnt_trq_msr`, then applies the command ports. In `POSITION` mode it writes
`jnt_pos_cmd` to actuator controls. In `TORQUE` mode it writes `jnt_trq_cmd` to
generalized forces. `Robot.step()` advances the owning scene data and returns
`False` only when the underlying simulation loop has been closed by an
interactive viewer.

Use `set_joint_pos(q, call_forward=True)` to seed joint state directly in KDL
order:

```python
robot.set_joint_pos([0.0] * robot.n_joints)
print(robot.fk_frame())
```

## Reset

`Env` is the resettable owner variant. It keeps registered robots synchronized
across resets and runtime scene rebuilds.

```python
env = mjk.Env.build(spec)
robot = env.create_robot("base_link", "bracelet_link")

def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos([0.0] * robot.n_joints, call_forward=False)

env.on_reset = on_reset

opts = mjk.ResetOptions()
opts.keyframe = 0
info = env.reset(opts)
```

`Env.reset()` restores MuJoCo state, calls the optional reset hook, re-seeds
registered robot command ports from measured state, and clears stale forces.
`ResetContext.options` and `ResetContext.info` expose the active reset request
inside the hook. `ResetOptions.use_keyframe = False` forces a default MuJoCo
reset instead of loading a keyframe.

`Env` mirrors the scene-level helpers for runtime state:

```python
env.set_actuator_ctrl("finger", 0.25)
frame = env.body_frame("red_cube")
env.save_xml("episode_start.xml")
```

## Runtime Add And Remove Objects

```python
cube = mjk.SceneObject()
cube.name = "cube"
cube.shape = mjk.Shape.BOX
cube.size = [0.02, 0.02, 0.02]
cube.pos = [0.4, 0.0, 0.02]
cube.rgba = [1.0, 0.5, 0.0, 1.0]
cube.mass = 0.1
cube.friction = [0.8, 0.02, 0.001]

scene.add_object(cube)
robot.update()
scene.remove_object("cube")
```

`Scene.add_object()`, `Scene.remove_object()`, `Env.add_object()`, and
`Env.remove_object()` rebuild the native MuJoCo model/data and rebind existing
Python `Robot` handles. Calling `Scene.close()` or `Env.close()` invalidates
dependent robot handles; using one raises `RuntimeError("robot is closed")`.

Use `Env.add_object()` / `Env.remove_object()` when the scene is resettable and
registered robots should remain synchronized with future resets.

## Headless Video Recording

```python
recorder = mjk.VideoRecorder.open_preset(
    scene,
    "sim.mp4",
    mjk.VideoResolution.R720p,
    fps=60,
)
recorder.set_free_camera(2.5, 135.0, -20.0, (0.0, 0.0, 0.7))

for _ in range(3000):
    robot.update()
    robot.step()
    recorder.record_frame()

recorder.close()
```

Use `VideoRecorder.open(scene_or_env, path, width, height, fps)` for explicit
frame sizes, or `open_preset()` for `VideoResolution` presets. `record_frame()`
captures the current state; step the scene or robot before each call.

The recorder camera list includes `Current`, `Free`, `Tracking`, robot MJCF
cameras, and cameras added through `SceneSpec.cameras`.

## Recording a Window

A `VideoRecorder` renders the scene a second time, which is what a viewpoint the
window is not showing costs. To record the view already on screen, ask the
viewer for it instead: the frame it has just drawn is read back and encoded, so
the recording costs a pixel copy rather than a render.

```python
viewer = mjk.SimulateViewer.open(robot, "MuJoCo")
viewer.start_recording("window.mp4", fps=30)   # 3D view only, no UI panels

while viewer.step():
    robot.update()

viewer.stop_recording()   # close() does this too
viewer.close()
```

`width`/`height` default to the view's own size; giving smaller ones scales the
frame on the GPU before it is read back, which is the cheaper way to record a
large window. Encoding runs in an `ffmpeg` subprocess, so it stays off the
control loop.

## Viewer Controls

`SimulateViewer.open(robot)` starts the custom MuJoCo Simulate UI and binds it
to a robot handle. For object-only scenes, pass the `Scene` instead:

```python
viewer = mjk.SimulateViewer.open(robot, "MuJoCo")
while viewer.step():
    robot.update()
viewer.close()

viewer = mjk.SimulateViewer.open(scene, "object scene")
while viewer.step():
    pass
viewer.close()
```

The UI exposes the same wrapper panels as the C++ viewer: `Frames`, `Trace`,
`Perturb`, `Recorder`, and `RTF`. Use `viewer.add_trace_segment()` for Python
trace overlays and `viewer.use_camera(name)` to switch to a named camera.
`viewer.realtime_factor` controls pacing; `1.0` is real time, `0.0` runs as fast
as the loop allows.
