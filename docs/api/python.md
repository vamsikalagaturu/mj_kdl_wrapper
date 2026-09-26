# Python Bindings API Guide {#page_api_python}

This page collects the Python wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated stubs in
`python/mj_kdl_wrapper/*.pyi`.

Coming from 0.2.x? Placement orientation moved from `.euler` to `.quat`
`[x, y, z, w]`; see [Migrating from 0.2.x](@ref sec_migrate_quat). Units, frames and what
persists between calls: [Conventions](@ref page_conventions).

A call that fails on its input raises `RuntimeError` whose message says why (the C++
`Status::error` text); a wrong-sized joint vector raises `ValueError`, and a spec field of
the wrong length (a 5-value `quat`, say) raises `TypeError`.

The Python package exposes the same scene, robot, reset, viewer, and recorder
concepts as the C++ wrapper. As in C++, one `Env` owns the MuJoCo
`mjModel`/`mjData`, its robots and its viewer; KDL values come back through the
upstream `PyKDL` module.

The wheel bundles `PyKDL` as a top-level extension module built from the same
Orocos KDL fork as the wrapper. `import PyKDL` works after installation, but
`PyKDL` does not appear in `pip list` / `uv pip list` because it is not installed
as a separate Python distribution with its own `.dist-info`.

## Model Paths

`mj_kdl_wrapper.menagerie.model_path(name)` resolves the models the examples use
(`kinova_gen3`, `robotiq_2f85`, `universal_robots_ur5e`, `universal_robots_ur10e`)
without hard-coding a checkout location, in this order:

1. The file named by `env_var`, when the caller passes one (`model_path("kinova_gen3",
   env_var="MJ_KDL_MODEL")`) and that variable is set.
2. `MJ_KDL_MENAGERIE` - a MuJoCo Menagerie checkout root.
3. The bundled assets in `~/.cache/mj_kdl_wrapper/assets`: a bundled model derived from
   Menagerie's (`kinova_gen3/gen3.xml`, with armature) replaces the upstream copy.
4. The Menagerie checkout in `~/.cache/mj_kdl_wrapper/menagerie`, populated by
   `mj-kdl-fetch-menagerie`.

`menagerie.asset_path(rel)` resolves bundled assets (gripper, table, mug, cabinet, F/T
sensor): an optional per-asset `env_var` override, otherwise the user cache
`~/.cache/mj_kdl_wrapper/assets`, which it refreshes from the installed package when it is
missing or out of date. `mj-kdl-fetch-menagerie` populates both caches, and the same cache
backs the C++ examples (see the C++ guide). It refuses a `--dest` that is neither empty nor a
Menagerie checkout, and a failed fetch leaves nothing behind.

**Overrides:** the example scripts pass these per-file env vars -- `MJ_KDL_MODEL`
(arm), `MJ_KDL_GRIPPER`, `MJ_KDL_TABLE`, `MJ_KDL_BOTTLE`, `MJ_KDL_RECEIVER`, `MJ_KDL_CABINET`,
`MJ_KDL_FT`. Each must point at an existing file or resolution raises a clear error.

For other MJCF sources, set the relevant environment variable or assign
`RobotSpec.path` directly.

## Load From MJCF

`SceneSpec` has no defaults for `timestep`, `add_floor`, or `add_skybox`.
Those are explicit scene choices; `Env.build()` raises if one is unset or `timestep <= 0`.
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

env = mjk.Env.build(spec)
```

For an object-only scene, put MJCF or primitive objects in `spec.objects` and
leave `spec.robots` empty:

```python
cabinet = mjk.SceneObject()
cabinet.name = "cabinet"
cabinet.mjcf_path = mjk.menagerie.asset_path("cabinet/cabinet.xml")
cabinet.fixed = True

spec.objects = [cabinet]
env = mjk.Env.build(spec)
```

`Env` owns the compiled MuJoCo model/data. Call `close()`, or use it as a context manager,
to release native resources deterministically. `step()` advances it whether or not it has
robots; `env.data.time` and `env.model.opt.timestep` report where it is. `build()`, `step()`, `pace()`,
`reset()`, `add_object()`, `remove_object()` and a recorder's `record_frame()` and `render_rgb()`
release the GIL while they run. `env.spec` is a read-only copy of the scene the `Env` runs,
objects added or removed since `build()` included; changing the copy changes nothing.

```python
with mjk.Env.build(spec) as env:
    env.step()
```

```python
import mujoco

for _ in range(10):
    env.step()
print(env.data.time, env.model.opt.timestep)
env.save_model_xml("combined_scene.xml")
mujoco.mj_saveModel(env.model, "combined_scene.mjb", None)
```

The log level is a threshold: messages at that level and above print. `INFO` (the default)
prints everything, `WARN` warnings and errors, `ERROR` errors only, `NONE` nothing:

```python
mjk.set_log_level(mjk.LogLevel.WARN)   # quiet the scene-construction progress
assert mjk.get_log_level() == mjk.LogLevel.WARN
print(mjk.__mujoco_version__)
```

`mjk.__version__` is the Python package version. `mjk.__mujoco_version__` is the
MuJoCo version the extension was built against.

## Init A KDL Chain

```python
robot = env.create_robot("base_link", "bracelet_link")

robot.ctrl_mode = mjk.CtrlMode.POSITION
robot.jnt_pos_cmd = [0.0] * robot.n_joints
env.update()
env.step()
```

`create_robot()` registers the robot with the `Env`, whose `update()` reads and
commands every registered robot. After init, `joint_names`, `joint_limits`, and
all joint port vectors are in KDL chain order, and the command ports hold the
current pose.

The ports (`jnt_pos_msr`, `jnt_vel_msr`, `jnt_trq_msr`, `jnt_pos_cmd`, `jnt_vel_cmd`,
`jnt_trq_cmd`, and the boolean `jnt_saturated`) read as read-only numpy copies. Write a port
by assigning the whole vector; writing one element raises, so a lost write cannot go unnoticed:

```python
q = robot.jnt_pos_cmd.copy()
q[0] += 0.1
robot.jnt_pos_cmd = q          # writes the port
robot.jnt_pos_cmd[0] = 0.1     # ValueError: assignment destination is read-only
```

A held copy stays what it was: it does not follow later `update()` or `reset()` calls. An
assignment must have `n_joints` values (`ValueError` otherwise), and a robot of a closed `Env`
raises `RuntimeError("robot is closed")`.

When a tool or gripper is attached, pass `ToolFrameSpec` so the KDL chain uses
the TCP site and includes the tool's inertia. `tool_body` is the tool's root body; for the
2F-85 that is `g_base_mount`, which carries mass too. `robot.tip_T_tcp` is the transform from
the chain tip to the TCP:

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base_mount"
tool.tcp_site = "g_pinch"

robot = env.create_robot("base_link", "bracelet_link", tool=tool)
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
robot = env.create_robot("base_link", "bracelet_link", tool=tool)
env.update()
wrench = robot.ft_sensor("wrist_ft")
```

When the chain comes from the same description that produced the MJCF, register it as given;
`joint_names` are the MuJoCo joints in chain order, and no tool inertia is lumped onto it:

```python
robot = env.create_robot_from_chain(chain, joint_names, tool=tool)
```

## Attach MJCF Bodies

`AttachTarget` is a tagged pair of `AttachKind` and an element name. The Kinova
GEN3 MJCF exports `pinch_site` on the bracelet, so a Robotiq gripper can attach
without manual pose offsets:

```python
gripper = mjk.AttachmentSpec()
gripper.mjcf_path = mjk.menagerie.asset_path("robotiq_2f85/2f85.xml")
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
once. `create_robot()`'s `prefix` is prepended to every name it resolves (base and tip
bodies, tool body, TCP site, F/T sensor names), so `create_robot("base_link",
"bracelet_link", "r2_", tool)` and the same call with `r2_`-prefixed names and no prefix
build the same robot.

```python
left = mjk.RobotSpec()
left.path = mjk.menagerie.model_path("kinova_gen3")
left.pos = [-0.5, 0.0, 0.0]

right = mjk.RobotSpec()
right.path = mjk.menagerie.model_path("kinova_gen3")
right.prefix = "r2_"
right.pos = [0.5, 0.0, 0.0]

spec.robots = [left, right]
env = mjk.Env.build(spec)

robot1 = env.create_robot("base_link", "bracelet_link")
robot2 = env.create_robot("r2_base_link", "r2_bracelet_link")
```

## Table And Scene Objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so robots and
objects can be mounted to sites or bodies created earlier in the scene spec.
Build order is decorations, objects in declaration order, robots, then cameras.

For primitive objects, `shape`, `size`, `rgba` and `friction` are required, and `mass`
unless the object is `fixed`. For MJCF-backed objects, `mjcf_path` takes precedence and
primitive geometry fields are ignored at runtime.

```python
table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = mjk.menagerie.asset_path("table.xml")
table.pos = [0.0, 0.0, 0.7]
table.fixed = True

mount = "table_top"   # the asset's own site name; SceneObject.prefix would prepend to it

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
env = mjk.Env.build(spec)
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

env = mjk.Env.build(spec)
print([env.model.camera(i).name for i in range(env.model.ncam)])
```

The viewer's and `VideoRecorder`'s `use_camera(name)` switch to a fixed camera.
Pass `""` to return to the free camera.

Use `body_frame()` and `site_frame()` to read world poses as `PyKDL.Frame`; the
kinematics are recomputed only when the state changed since they were last
computed. `Env.set_body_pose()` teleports a free body and zeroes its velocity; its quaternion
is `[x, y, z, w]` like every quaternion in the API.

```python
tcp = env.site_frame("g_pinch")
env.set_body_pose("red_cube", [0.45, 0.0, 0.75], [0.0, 0.0, 0.0, 1.0])
```

For named actuators that are not part of a `Robot` joint mapping, write `env.data` directly
(see [Direct MuJoCo access](@ref sec_py_direct_mujoco)); `env.model.actuator(name)` raises
`KeyError` when the actuator is missing:

```python
env.data.actuator("g_fingers_actuator").ctrl[0] = 0.25   # the "g_"-prefixed 2F-85's drive
print(env.data.actuator("g_fingers_actuator").ctrl[0])
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

`Robot.set_joint_pos()` accepts either a Python sequence or a `PyKDL.JntArray`.
`Env.body_frame()` and `Env.site_frame()` return `PyKDL.Frame`.

Gravity torques come from `ChainDynParam` on the same chain and `JntArray`:

```python
dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, env.spec.gravity_z))
g = kdl.JntArray(robot.n_joints)
dyn.JntToGravity(q, g)
```

## Control Loop

```python
robot.set_control_mode(mjk.CtrlMode.TORQUE)   # seeds the torque ports, no jump
dyn = kdl.ChainDynParam(robot.kdl_chain(), kdl.Vector(0.0, 0.0, env.spec.gravity_z))
q, g = kdl.JntArray(robot.n_joints), kdl.JntArray(robot.n_joints)

while env.data.time < 5.0 and env.step():
    env.update()
    for i, value in enumerate(robot.jnt_pos_msr):
        q[i] = value
    dyn.JntToGravity(q, g)
    robot.jnt_trq_cmd = [g[i] for i in range(robot.n_joints)]
```

`env.update()` reads MuJoCo state into every registered robot's `jnt_pos_msr`,
`jnt_vel_msr`, and `jnt_trq_msr`, then applies their command ports: the active
mode's command goes to that mode's actuator controls (`jnt_pos_cmd`,
`jnt_vel_cmd` or `jnt_trq_cmd`), clamped to `ctrlrange` and flagged in
`jnt_saturated`. `robot.joint_force_limits()` returns each joint's torque limit in the active
mode. `env.step()` advances one timestep, after which joint state and frames describe the same
instant; it returns `False` only when the viewer window has been closed, so a headless loop
needs its own end condition.

Use `set_joint_pos(q)` to seed joint state directly in KDL order:

```python
robot.set_joint_pos([0.0] * robot.n_joints)
env.update()                   # jnt_pos_msr now reads the seeded pose
```

## Reset

`Env.reset()` resets everything the `Env` holds and keeps registered robots
synchronized across resets and runtime scene rebuilds.

```python
env = mjk.Env.build(spec)
robot = env.create_robot("base_link", "bracelet_link")

def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos([0.0] * robot.n_joints)

env.on_reset = on_reset

opts = mjk.ResetOptions()
opts.keyframe = 0
info = env.reset(opts)
```

`Env.reset()` restores MuJoCo state, re-seeds every registered robot's ports and F/T
readings from the reset state (so the first command holds the pose), then calls the optional
reset hook, then reads the measurements back. A command the hook primes is kept; a POSITION
robot the hook moves needs its `jnt_pos_cmd` set there too. The Simulate UI's reset button
does the same. An exception the hook raises comes out of `reset()` (or of the `step()` that
ran a UI reset) after the measurements have been read back; `env.on_reset` reads `None` until
set, and `close()` drops it.
`ResetContext.options` and `ResetContext.info` are a copy of the reset request and its result,
safe to keep after the hook returns. `ResetOptions.use_keyframe = False` forces a default
MuJoCo reset instead of loading a keyframe.

`Env` also carries the runtime-state helpers:

```python
frame = env.body_frame("red_cube")
env.save_model_xml("episode_start.xml")
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

env.add_object(cube)
env.update()
env.remove_object("cube")
```

`Env.add_object()` and `Env.remove_object()` rebuild the native MuJoCo
model/data and rebind existing Python `Robot` handles, the viewer and open recorders.
`remove_object()` of a name the scene does not hold raises `RuntimeError` naming it. Calling
`Env.close()` invalidates dependent robot handles; using one raises
`RuntimeError("robot is closed")`.

## Direct MuJoCo Access {#sec_py_direct_mujoco}

`env.model` and `env.data` are the live `mujoco.MjModel` and `mujoco.MjData` the `Env` runs
on, so the `mujoco` package reads and writes them in place. `add_object()` and
`remove_object()` replace both: re-read them after either call rather than holding the old
objects. The viewer thread reads `env.data` too; call `mujoco` functions on it only while the
viewer is closed or paused.

```python
import mujoco
import numpy as np

print(env.data.time, env.model.opt.timestep)
env.data.actuator("g_fingers_actuator").ctrl[0] = 0.25   # KeyError if missing
env.data.body("red_cube").xfrc_applied[:] = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0]   # force, torque

f = np.zeros(6)
for i in range(env.data.ncon):
    mujoco.mj_contactForce(env.model, env.data, i, f)   # contact frame: normal, then tangents
```

## Headless Video Recording

```python
recorder = mjk.VideoRecorder.open_preset(
    env,
    "sim.mp4",
    mjk.VideoResolution.R720p,
    fps=60,
)
recorder.set_free_camera(2.5, 135.0, -20.0, (0.0, 0.0, 0.7))

for _ in range(3000):
    env.step()
    env.update()
    recorder.record_frame()

recorder.close()
```

Use `VideoRecorder.open(env, path, width, height, fps)` for explicit frame
sizes, or `open_preset()` for `VideoResolution` presets. `record_frame()`
captures the current state; step the `Env` before each call. A recorder is also a context
manager.

For frames in memory instead of a file, open an offscreen renderer; `render_rgb()` returns a
`(height, width, 3)` uint8 array, top row first, on any recorder:

```python
with mjk.VideoRecorder.open_offscreen(env, 640, 480) as rec:
    rgb = rec.render_rgb()
```

A recorder's `use_camera(name)` takes any fixed camera of the model (robot MJCF cameras and
`SceneSpec.cameras`); `""` returns to the free camera.

Recording is offscreen-only: a `VideoRecorder` renders on its own EGL context,
so the interactive window never pays for it. To record the view a fresh window
opens with, leave the recorder on its default free camera. The Simulate UI's Recorder panel
also offers its `Current` and `Tracking` views, all rendered offscreen.

## Viewer Controls

`env.open_viewer()` starts the custom MuJoCo Simulate UI on the `Env`, with or
without robots; `env.step()` then also honours its pause, perturbation and
recording, and `env.close()` closes the window:

```python
env.open_viewer("MuJoCo")
while env.data.time < 10.0 and env.step():   # ends by itself, or when the window closes
    env.update()
    env.pace()
env.close()
```

`env.viewer.key_pressed(key)` reports a held GLFW key code, and
`env.viewer.capture_key(key)` withholds a key from the UI's own bindings (arrows, space,
escape) so a controller can use it.

The UI exposes the same wrapper panels as the C++ viewer: `Frames`, `Trace`,
`Perturb`, `Recorder`, and `RTF`. Trace overlays, camera selection and the
real-time factor (`1.0` is real time, `0.0` runs as fast as the loop allows) go
through the `Env`'s viewer; see the stubs in `python/mj_kdl_wrapper/*.pyi` for
the exact method names.
