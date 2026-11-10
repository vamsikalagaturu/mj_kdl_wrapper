# Python Bindings API Guide {#page_api_python}

This page collects the Python wrapper usage notes that are too detailed for the
README. For complete function signatures, see the generated stubs in
`python/mj_kdl_wrapper/*.pyi`.

Coming from 0.5? The viewer is `mjk.Viewer()` opened with `viewer.open(env, title)` and
paced with `viewer.pace()`; see [Migrating to 0.6](@ref page_migration). Units, frames and
what persists between calls: [Conventions](@ref page_conventions).

A call that fails on its input raises `RuntimeError` whose message says why (the C++
`Status::error` text); a wrong-sized joint vector raises `ValueError`, and a spec field of
the wrong length (a 5-value `quat`, say) raises `TypeError`.

The Python package exposes the same scene, robot, reset and overlay concepts as the C++
core library. As in C++, one `Env` owns the MuJoCo `mjModel`/`mjData` and its robots; KDL
values come back through the upstream `PyKDL` module. The wheel also binds the C++ viewer
and recorder: `mjk.Viewer` is the same MuJoCo Studio window as `mj_kdl::Viewer`, and
`mjk.VideoRecorder` the same Filament recorder as `mj_kdl::VideoRecorder`. Python >= 3.11,
`mujoco==4.0.0`.

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
`spec.robots` may be empty; object-only scenes are valid. `spec.speculative_contacts` (default
`False`) turns on MuJoCo's speculative contacts; `spec.integrator` (`mjk.Integrator`) and
`spec.solver` (`mjk.Solver`) keep the first robot's options at `Robot`, and cloth needs
`mjk.Integrator.Discrete` (see `ex_table_towel.py`).

`use_stage = True` builds the scene on a stage MJCF that supplies the floor, sky, haze and
lights: by default the bundled `assets/stage/default.xml` (an infinite checker floor plane, a
gradient skybox and two spot lights), or the file named by `spec.stage`. `use_stage = False`
builds on an empty world.

```python
import mj_kdl_wrapper as mjk

spec = mjk.SceneSpec()
spec.timestep = 0.001
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

`mjk.__version__` is the package version (`0.6.0`). `mjk.version()`, the C++
`mj_kdl::version()`, is the build's: the same for a build of the `v0.6.0` tag, else
`0.6.0-dev+<commit>`, with `.dirty` when tracked files had uncommitted changes. `mjk.__mujoco_version__` is the
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

With more than one robot, give each a prefix of its own, so MuJoCo names stay distinct and
every robot's are built the same way. `create_robot()`'s `prefix` is prepended to every name it resolves (base and tip
bodies, tool body, TCP site, F/T sensor names), so `create_robot("base_link",
"bracelet_link", "r2_", tool)` and the same call with `r2_`-prefixed names and no prefix
build the same robot.

```python
left = mjk.RobotSpec()
left.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
left.prefix = "r1_"
left.pos = [-0.5, 0.0, 0.0]

right = mjk.RobotSpec()
right.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
right.prefix = "r2_"
right.pos = [0.5, 0.0, 0.0]

spec.robots = [left, right]
env = mjk.Env.build(spec)

robot1 = env.create_robot("base_link", "bracelet_link", "r1_")
robot2 = env.create_robot("base_link", "bracelet_link", "r2_")
robot1.name = "left"    # the viewer's Robots section shows it; unnamed, the base body
robot2.name = "right"
```

## Table And Scene Objects

`SceneObject` and `RobotSpec` share the same `attach_to` field, so robots and
objects can be mounted to sites or bodies created earlier in the scene spec.
Build order is decorations, objects in declaration order, robots, then cameras.

For primitive objects, `shape` and `size` are required, and `mass` and `friction` unless the
object is `fixed`; an unset `rgba`, `mass` or `friction` takes MuJoCo's default. `Shape.MESH`
also needs `mesh_path`, with `size` as the mesh's scale; `Shape.ELLIPSOID` takes three radii.
For MJCF-backed objects, `mjcf_path` takes precedence and primitive geometry fields are
ignored at runtime.

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
instant; it never sleeps, so a loop needs its own end condition. With a viewer open, it
returns `False` once the window is closed, and `viewer.pace()` after it paces the loop (see
Viewer Controls below).

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
does the same. An exception the hook raises comes out of `reset()` (or of the `env.step()`
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
model/data and rebind existing Python `Robot` handles and scene slots; an open `Viewer` and
every `VideoRecorder` follow the new model, but a mujoco renderer built on the old `env.model`
must be rebuilt.
`remove_object()` of a name the scene does not hold raises `RuntimeError` naming it. Calling
`Env.close()` invalidates dependent robot handles; using one raises
`RuntimeError("robot is closed")`.

## Direct MuJoCo Access {#sec_py_direct_mujoco}

`env.model` and `env.data` are the live `mujoco.MjModel` and `mujoco.MjData` the `Env` runs
on, so the `mujoco` package reads and writes them in place. `add_object()` and
`remove_object()` replace both: re-read them after either call rather than holding the old
objects. The viewer draws a copy of the state each `env.step()` publishes, never `env.data`
itself.

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

`mjk.VideoRecorder` is the C++ recorder: Filament offscreen rendering on a thread of its own,
encoded to an H.264 MP4 by ffmpeg (on `PATH`), or kept in memory. No window is needed.
Recording a run looks like this:

```python
with mjk.VideoRecorder.open_preset(env, "sim.mp4", mjk.VideoResolution.R720p, 60) as vr:
    vr.set_free_camera(1.35, 145.0, -22.0, (0.05, 0.02, 0.88))   # distance, azimuth, elevation, lookat
    for i in range(6000):
        env.update()
        env.step()
        if i % 16 == 0:                # ~60 fps at a 1 ms timestep
            vr.record_frame()
```

`VideoRecorder.open(env, out_path, width=1280, height=720, fps=60)` takes an explicit size;
`VideoResolution` has `R360p`, `R480p`, `R720p`, `R1080p`, `R2K` and `R4K`.
`VideoRecorder.open_offscreen(env, width, height)` writes no file: `render_rgb()` returns the
`Env`'s current state as a `(height, width, 3)` `uint8` array, top row first.
`use_camera(name)` renders from a named model camera (`""` for the free camera). `close()`
or the context manager finishes the file.

## Overlays

`env.add_trace_segment(a, b, rgba=None)` and `env.add_overlay_arrow(from_, dir, length,
rgba=None)` append world-space geoms [m]; `env.clear_trace()` removes them all. The viewer
and every `VideoRecorder` draw them; `env.overlay_geoms()` returns them as `mujoco.MjvGeom`s
for a mujoco renderer of your own.

```python
env.clear_trace()
env.add_trace_segment(prev_tcp, tcp, [1.0, 0.9, 0.1, 1.0])
```

## Viewer Controls

`mjk.Viewer` is the C++ viewer (`mj_kdl::Viewer`), the same MuJoCo Studio window with the same
menus, panels, Step Back/Fwd history, Explorer, wrapper sections (`Robots`, `Frames`,
`Perturb`, `Recording`), camera views and key bindings; see
Viewer Controls in the [C++ API guide](@ref page_api_cpp). `viewer.open(env, title="MuJoCo")`
opens it on the `Env`, with or without robots, and raises `RuntimeError` when there is no
display. The `Env` keeps stepping on the caller's thread; the window runs on a thread of its
own and draws a copy of the state each `env.step()` publishes. Its pause, reset,
perturbation, edited controls, physics options and recording reach the `Env` inside
`env.step()`, which returns `False` once the window is closed. `viewer.pace()` sleeps out one
timestep at `viewer.realtime_factor` and does nothing while the window is closed, so the same
loop runs with and without a window:

```python
viewer = mjk.Viewer()
if gui:
    viewer.open(env, "MuJoCo")
while env.data.time < 10.0:
    env.update()
    if not env.step():   # False once the window closes
        break
    viewer.pace()
viewer.close()
env.close()
```

`Viewer` is also a context manager. `viewer.is_running()`, `viewer.realtime_factor` (1.0 real
time, 0.0 uncapped), `viewer.fps`, `use_camera(name)` and `set_free_camera(distance, azimuth,
elevation, lookat)` mirror the C++ calls. `key_pressed(key)` and `capture_key(key,
capture=True)` take `ImGuiKey` integer codes as in `imgui.h`, as in C++. Every setting of the
C++ `Viewer` is a property of the same name, set before `open()`: `paused`, `window_width`,
`window_height`, `min_panel_width`, `theme` (`mjk.ViewerTheme`), `ini_path`, `record_path`,
`record_resolution`, `record_fps`, `framing_frames`, `speed_window`, `plugin_max_geoms`,
`joint_limit_margin`, `joint_limit_highlight`, `manipulability_ellipsoid` and
`manipulability_scale`;
the C++ guide's Viewer Controls lists their defaults.
