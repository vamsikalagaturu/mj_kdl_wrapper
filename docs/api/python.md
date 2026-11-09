# Python Bindings API Guide {#page_api_python}

This page collects the Python wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated stubs in
`python/mj_kdl_wrapper/*.pyi`.

Coming from 0.5? The viewer is `mj_kdl_wrapper.viewer.Viewer` and recording uses mujoco's
own renderers; see [Migrating to 0.6](@ref sec_migrate_mujoco4).
Coming from 0.4? The models are in `mjk.ASSETS_DIR`; see
[Migrating to 0.5](@ref sec_migrate_mj_kdl_wrapper).
Coming from 0.2.x? Placement orientation moved from `.euler` to `.quat`
`[x, y, z, w]`; see [Migrating from 0.2.x](@ref sec_migrate_quat). Units, frames and what
persists between calls: [Conventions](@ref page_conventions).

A call that fails on its input raises `RuntimeError` whose message says why (the C++
`Status::error` text); a wrong-sized joint vector raises `ValueError`, and a spec field of
the wrong length (a 5-value `quat`, say) raises `TypeError`.

The Python package exposes the same scene, robot, reset and overlay concepts as the C++
core library. As in C++, one `Env` owns the MuJoCo `mjModel`/`mjData` and its robots; KDL
values come back through the upstream `PyKDL` module. The wheel builds the core only:
`mj_kdl_wrapper.viewer.Viewer` shows an `Env` through `mujoco.studio`, and recording uses
mujoco's own renderers. Python >= 3.11, `mujoco==4.0.0`.

The wheel bundles `PyKDL` as a top-level extension module built from the same
Orocos KDL fork as the wrapper. `import PyKDL` works after installation, but
`PyKDL` does not appear in `pip list` / `uv pip list` because it is not installed
as a separate Python distribution with its own `.dist-info`.

## Model Paths

`mj_kdl_wrapper.ASSETS_DIR` is a `pathlib.Path` to the models installed with the package
(Kinova Gen3, Robotiq 2F-85, table, mug, cabinet, cube, F/T sensor, door latch); the
examples and tests load them as `str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")`. For other
MJCF sources (e.g. a MuJoCo Menagerie checkout), assign `RobotSpec.path` or `mjcf_path`
directly; a model that brings its own floor (Menagerie's `scene.xml`) needs `use_stage = False`,
or its `groundplane` texture collides with the stage's.

## Load From MJCF

`SceneSpec` has no defaults for `timestep` or `use_stage`.
Those are explicit scene choices; `Env.build()` raises if one is unset or `timestep <= 0`.
`spec.robots` may be empty; object-only scenes are valid.

`use_stage = True` builds the scene on a stage MJCF that supplies the floor, sky, haze and
lights: by default the bundled `assets/stage/default.xml` (an infinite checker floor plane, a
gradient skybox and two spot lights), or the file named by `spec.stage`. `use_stage = False`
builds on an empty world.

```python
import mj_kdl_wrapper as mjk

spec = mjk.SceneSpec()
spec.timestep = 0.002
spec.use_stage = True

robot_spec = mjk.RobotSpec()
robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
spec.robots = [robot_spec]

env = mjk.Env.build(spec)
```

For an object-only scene, put MJCF or primitive objects in `spec.objects` and
leave `spec.robots` empty:

```python
cabinet = mjk.SceneObject()
cabinet.name = "cabinet"
cabinet.mjcf_path = str(mjk.ASSETS_DIR / "cabinet/cabinet.xml")
cabinet.fixed = True

spec.objects = [cabinet]
env = mjk.Env.build(spec)
```

`Env` owns the compiled MuJoCo model/data. Call `close()`, or use it as a context manager,
to release native resources deterministically. `step()` advances it whether or not it has
robots; `env.data.time` and `env.model.opt.timestep` report where it is. `build()`, `step()`,
`reset()`, `add_object()` and `remove_object()` release the GIL while they run. `env.spec` is a read-only copy of the scene the `Env` runs,
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
ft_sensor.mjcf_path = str(mjk.ASSETS_DIR / "ft_sensor.xml")
ft_sensor.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")

gripper = mjk.AttachmentSpec()
gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "wrist_ft_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
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
gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
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
left.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
left.pos = [-0.5, 0.0, 0.0]

right = mjk.RobotSpec()
right.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
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
table.mjcf_path = str(mjk.ASSETS_DIR / "table.xml")
table.pos = [0.0, 0.0, 0.7]
table.fixed = True

mount = "table_top"   # the asset's own site name; SceneObject.prefix would prepend to it

robot_spec = mjk.RobotSpec()
robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
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

A renderer picks a fixed camera through its `mujoco.MjvCamera`: `type =
mujoco.mjtCamera.mjCAMERA_FIXED`, `fixedcamid = env.model.camera(name).id`.

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

while env.data.time < 5.0:
    env.update()
    for i, value in enumerate(robot.jnt_pos_msr):
        q[i] = value
    dyn.JntToGravity(q, g)
    robot.jnt_trq_cmd = [g[i] for i in range(robot.n_joints)]
    env.step()
```

`env.update()` reads MuJoCo state into every registered robot's `jnt_pos_msr`,
`jnt_vel_msr`, and `jnt_trq_msr`, then applies their command ports: the active
mode's command goes to that mode's actuator controls (`jnt_pos_cmd`,
`jnt_vel_cmd` or `jnt_trq_cmd`), clamped to `ctrlrange` and flagged in
`jnt_saturated`. `robot.joint_force_limits()` returns each joint's torque limit in the active
mode. `env.step()` advances one timestep, after which joint state and frames describe the same
instant; it never sleeps, so a loop needs its own end condition. With a viewer, call
`viewer.step()` instead (see Viewer Controls below).

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
robot the hook moves needs its `jnt_pos_cmd` set there too. The viewer's reset button
does the same. An exception the hook raises comes out of `reset()` (or of the `viewer.step()`
that ran a UI reset) after the measurements have been read back; `env.on_reset` reads `None` until
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
model/data and rebind existing Python `Robot` handles and scene slots; a renderer built on
the old `env.model` must be rebuilt.
`remove_object()` of a name the scene does not hold raises `RuntimeError` naming it. Calling
`Env.close()` invalidates dependent robot handles; using one raises
`RuntimeError("robot is closed")`.

## Direct MuJoCo Access {#sec_py_direct_mujoco}

`env.model` and `env.data` are the live `mujoco.MjModel` and `mujoco.MjData` the `Env` runs
on, so the `mujoco` package reads and writes them in place. `add_object()` and
`remove_object()` replace both: re-read them after either call rather than holding the old
objects. The viewer draws what each `viewer.sync()` publishes, never `env.data` itself.

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

The wheel has no recorder of its own; render `env.model` / `env.data` with mujoco's
renderers. `ex_table_pour.py --record` renders with `mujoco.rendering.filament` into a numpy
array and encodes it with `imageio[ffmpeg]` (`pip install "imageio[ffmpeg]"`):

```python
import imageio.v2 as imageio
import mujoco
import numpy as np
from mujoco import mjrf
from mujoco.rendering.filament import model_scene, renderer

r = renderer.Renderer(mjrf.Context(mjrf.ContextConfig()))
scene = model_scene.ModelScene(r.context, env.model, r.scene("scn"))
pixels = np.empty((720, 1280, 3), dtype=np.uint8)   # (height, width, 3), the frame in memory
r.target("out", pixels)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(env.model, cam)         # the view a fresh window opens with
writer = imageio.get_writer("sim.mp4", fps=60, codec="libx264", macro_block_size=1)

for _ in range(3000):
    env.update()
    env.step()
    scene.update(env.data)
    gl_cam = mujoco.mjv_camera2GLCamera(env.model, env.data, cam)
    r.render([renderer.View(scene="scn", target="out", camera=gl_cam)])
    writer.append_data(pixels)

writer.close()
```

## Overlays

`env.add_trace_segment(a, b, rgba=None)` and `env.add_overlay_arrow(from_, dir, length,
rgba=None)` append world-space geoms [m]; `env.clear_trace()` removes them all. The viewer
draws them; `env.overlay_geoms()` returns them as `mujoco.MjvGeom`s for a renderer's scene or
a viewer's `extra_geoms`.

```python
env.clear_trace()
env.add_trace_segment(prev_tcp, tcp, [1.0, 0.9, 0.1, 1.0])
```

## Viewer Controls

`mj_kdl_wrapper.viewer.Viewer(env, title)` opens a MuJoCo Studio window (`mujoco.studio`) on
the `Env`, with or without robots. The `Env` keeps stepping on the caller's thread; the window
runs on a thread of its own and draws what each `sync()` publishes. `viewer.step()` syncs
with the window (its perturbation, state edits, physics options and reset reach `env.data`),
calls `env.step()` unless the window is paused, then paces to the window's speed; it returns
`False` once the window is closed. A loop written for `env.step()` runs unchanged on it:

```python
from mj_kdl_wrapper.viewer import Viewer

viewer = Viewer(env, "MuJoCo") if gui else None
step = viewer.step if viewer else env.step
while env.data.time < 10.0:
    env.update()
    if not step():   # False once the window closes
        break
if viewer:
    viewer.close()
env.close()
```

`Viewer` is also a context manager. `viewer.sync()`, `viewer.pace()`, `viewer.paused`,
`viewer.speed` (percent of real time) and `viewer.is_running()` are there for a loop that
steps by itself. Unlike the C++ viewer it has no key or camera calls; the window's own panels
and mouse controls are mujoco's.
