# Python Tutorial {#page_tutorial_python}

This tutorial builds a Python simulation application in layers: create a scene,
initialize a KDL-backed robot handle, run position and torque control, add
attachments, objects, cameras, reset hooks, the viewer, recording, and a
small tabletop task structure.

Installation stays in the README. The snippets below assume `mj_kdl_wrapper`,
`PyKDL`, and the matching `mujoco` package import successfully; the models come from
the package's bundled assets, `mjk.ASSETS_DIR`.

## How To Read This Tutorial

The Python API has the same layers as the C++ wrapper:

| Layer | Type | Responsibility |
|-------|------|----------------|
| Scene description | `SceneSpec`, `RobotSpec`, `AttachmentSpec`, `SceneObject`, `CameraSpec` | Describe what should be compiled into MuJoCo |
| Runtime owner | `Env` | Own the compiled MuJoCo model/data, the robots and the overlay geoms |
| Robot control handle | `Robot` | KDL chain, measured ports, command ports |
| Visualization/recording | `Viewer`, `VideoRecorder` | The C++ MuJoCo Studio window and Filament offscreen recorder (MP4 or RGB frames) |
| KDL interop | `PyKDL` | FK, IK, dynamics solvers over the wrapper-built chain |

Most applications follow this flow:

1. Build a `SceneSpec`.
2. Call `Env.build(spec)`.
3. Create one or more `Robot` handles with `env.create_robot()`.
4. Install `env.on_reset` if task state must be restored.
5. Optionally create a `Viewer()` and `open(env, title)` it.
6. In the loop: `env.update()`, compute commands, write command ports, then `env.step()`
   (`False` once the window is closed) and `viewer.pace()` (a no-op without a window).
7. Close the viewer, the recorder and the environment when done.

## Complete Runnable Script

The code below is a complete interactive script. Saved as `python_tutorial_demo.py` and run
from any directory in an environment where `mj_kdl_wrapper` is installed, it builds a table
scene, initializes a tool-aware KDL robot, opens the viewer, and runs gravity
compensation.
Pass `--headless` to run the same controller without a window for CI or remote
machines without a display.

Later sections break this same structure into focused excerpts.

```python
from __future__ import annotations

import math
import argparse

import PyKDL as kdl

import mj_kdl_wrapper as mjk


HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
SURFACE_Z = 0.7
CUBE_HALF = 0.025


def joints(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q


def make_gripper() -> mjk.AttachmentSpec:
    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    gripper.prefix = "g_"
    return gripper


def make_table() -> mjk.SceneObject:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = str(mjk.ASSETS_DIR / "table.xml")
    table.pos = [0.0, 0.0, SURFACE_Z]
    table.fixed = True
    return table


def make_cube() -> mjk.SceneObject:
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [CUBE_HALF, CUBE_HALF, CUBE_HALF]
    cube.pos = [0.35, 0.10, SURFACE_Z + CUBE_HALF]
    cube.rgba = [0.1, 0.25, 1.0, 1.0]
    cube.mass = 0.1
    cube.condim = mjk.Condim.Torsional
    cube.friction = [0.8, 0.02, 0.001]
    return cube


def build_env() -> tuple[mjk.Env, mjk.Robot]:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.use_stage = True

    table = make_table()
    spec.objects = [table, make_cube()]

    arm = mjk.RobotSpec()
    arm.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
    arm.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "table_top")
    arm.attachments = [make_gripper()]
    spec.robots = [arm]

    camera = mjk.CameraSpec()
    camera.name = "task"
    camera.pos = [0.1, -0.9, 1.45]
    # extrinsic XYZ (35, 0, 5) deg: tilt down, yaw slightly right
    camera.quat = [0.300420, 0.013117, 0.041601, 0.952809]
    camera.fovy = 45.0
    spec.cameras = [camera]

    env = mjk.Env.build(spec)

    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base_mount"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def run_controller(env: mjk.Env, robot: mjk.Robot, dyn: kdl.ChainDynParam) -> None:
    env.update()
    g = kdl.JntArray(robot.n_joints)
    dyn.JntToGravity(joints(robot.jnt_pos_msr), g)
    robot.jnt_trq_cmd = [g[i] for i in range(robot.n_joints)]
    env.data.actuator("g_fingers_actuator").ctrl[0] = 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without opening the viewer")
    parser.add_argument("--duration", type=float, default=2.0, help="run duration in seconds")
    args = parser.parse_args()

    env, robot = build_env()
    viewer = mjk.Viewer()
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx: mjk.ResetContext) -> None:
            robot.set_joint_pos(HOME_POSE)
            env.set_body_pose("cube", [0.35, 0.10, SURFACE_Z + CUBE_HALF])

        env.on_reset = on_reset
        env.reset()

        chain = robot.kdl_chain()
        fk = kdl.ChainFkSolverPos_recursive(chain)
        dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, env.spec.gravity_z))

        def tcp_xyz() -> list[float]:
            frame = kdl.Frame()
            fk.JntToCart(joints(robot.jnt_pos_msr), frame)
            return [frame.p.x(), frame.p.y(), frame.p.z()]

        start_xyz = tcp_xyz()

        if not args.headless:
            viewer.open(env, "python tutorial")
        end_time = env.data.time + args.duration
        # Ends after the duration either way; closing the window ends it early.
        while env.data.time < end_time:
            run_controller(env, robot, dyn)
            if not env.step():
                break
            viewer.pace()

        env.update()
        end_xyz = tcp_xyz()
        drift = math.sqrt(sum((b - a) ** 2 for a, b in zip(start_xyz, end_xyz)))
        cameras = [env.model.camera(i).name for i in range(env.model.ncam)]
        print(f"MuJoCo {mjk.__mujoco_version__}")
        print(f"joints: {robot.n_joints}")
        print(f"cameras: {' '.join(cameras)}")
        print(f"EE drift: {drift:.6f} m")
    finally:
        viewer.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 1. Start With One Robot Scene

`SceneSpec.timestep` and `use_stage` are required. They are explicit choices,
not defaults the wrapper invents. `use_stage = True` builds on a stage MJCF
(floor, sky, haze, lights): the bundled `assets/stage/default.xml` unless
`spec.stage` names another; `use_stage = False` builds on an empty world.
`floor_z` (default `0.0`) is added to the height of every plane on the stage's
worldbody, for a scene whose world frame sits above ground level.

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

`Env` owns native MuJoCo resources. Call `env.close()` when you want deterministic
cleanup, or use the `Env` as a context manager:

```python
with mjk.Env.build(spec) as env:
    ...
```

A call that fails raises `RuntimeError` with the reason, e.g. an unset `timestep` or an
unknown body name.

## 2. Initialize A Robot

`Robot` is the handle for one controllable articulation. It stores KDL chain
metadata, joint names/limits, measured ports, and command ports.
`create_robot()` registers it with the `Env`, whose `update()` reads and
commands it from then on.

```python
robot = env.create_robot("base_link", "bracelet_link")
print(robot.n_joints)
print(robot.joint_names)
```

All joint vectors are in KDL chain order:

- `jnt_pos_msr`, `jnt_vel_msr`, `jnt_trq_msr` are measured ports.
- `jnt_pos_cmd` is used in `CtrlMode.POSITION`.
- `jnt_vel_cmd` is used in `CtrlMode.VELOCITY`.
- `jnt_trq_cmd` is used in `CtrlMode.TORQUE`.

Each read returns a read-only numpy copy. Write a port by assigning the whole vector
(`robot.jnt_pos_cmd = q`); `robot.jnt_pos_cmd[0] = x` raises `ValueError` instead of being
silently lost. Take `.copy()` to edit a port's values before assigning them back.

## 3. Run Position Control

In position mode, write `jnt_pos_cmd`. `env.update()` reads measured state for
every robot and applies their command ports to MuJoCo.

```python
robot.ctrl_mode = mjk.CtrlMode.POSITION

end_time = env.data.time + 2.0
while env.data.time < end_time:
    env.step()
    env.update()
    robot.jnt_pos_cmd = list(robot.jnt_pos_msr)
```

`env.step()` advances the `Env` by one MuJoCo timestep; afterwards joint state
and frames describe the same instant. `env.data.time` and `env.model.opt.timestep` expose
the runtime clock, with or without robots; `env.model` / `env.data` are the live
`mujoco.MjModel` / `mujoco.MjData` (see the Python API guide).

## 4. Add KDL Gravity Compensation

Gravity torques and FK come from the standard `PyKDL` solvers over the wrapper-built chain,
fed a `JntArray` of the measured positions:

```python
import PyKDL as kdl

def joints(values) -> kdl.JntArray:
    q = kdl.JntArray(len(values))
    for i, value in enumerate(values):
        q[i] = value
    return q

chain = robot.kdl_chain()
dyn = kdl.ChainDynParam(chain, kdl.Vector(0.0, 0.0, env.spec.gravity_z))
fk = kdl.ChainFkSolverPos_recursive(chain)

def gravity() -> list[float]:
    g = kdl.JntArray(robot.n_joints)
    dyn.JntToGravity(joints(robot.jnt_pos_msr), g)
    return [g[i] for i in range(robot.n_joints)]

robot.ctrl_mode = mjk.CtrlMode.TORQUE

while env.data.time < end_time:
    env.step()
    env.update()
    robot.jnt_trq_cmd = gravity()

tcp = kdl.Frame()
fk.JntToCart(joints(robot.jnt_pos_msr), tcp)
```

The later snippets reuse `joints()`, `fk` and `gravity()`.

## 5. Attach A Gripper Or Tool

Attachments are MJCF assets attached under a body, site, or frame in the
accumulated robot spec. They are applied in order.

```python
gripper = mjk.AttachmentSpec()
gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
robot_spec.attachments = [gripper]

spec.robots = [robot_spec]
env = mjk.Env.build(spec)
```

If the model has no useful mount site, attach by body name and add offsets.
Assign the whole list -- `pos` and `quat` return a copy, so item assignment
such as `gripper.pos[2] = -0.061525` is silently discarded:

```python
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Body, "bracelet_link")
gripper.pos = [0.0, 0.0, -0.061525]
gripper.quat = [1.0, 0.0, 0.0, 0.0]  # 180 deg about x, [x, y, z, w]
```

Tell KDL about the attached tool when creating the robot:

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base_mount"
tool.tcp_site = "g_pinch"

robot = env.create_robot("base_link", "bracelet_link", tool=tool)
assert robot.has_tcp_frame
```

`tool_body` identifies the attached subtree whose inertia should be included in
the KDL chain; for the 2F-85 it is the root body `g_base_mount`, whose mass `g_base` would
leave out. `tcp_site` becomes the terminal frame for FK and task-space code.
If the scene includes an attached FT sensor asset, add a `ForceTorqueSensorSpec`
to `tool.ft_sensors`; `robot.ft_sensor(name)` returns a `PyKDL.Wrench`.

## 6. Add Tables, Objects, And Asset Sites

`SceneObject` supports primitive objects and MJCF-backed assets. MJCF assets use
`mjcf_path`; primitive objects require `shape`, `size`, `rgba` and `friction`, and `mass`
unless `fixed`.

```python
table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = str(mjk.ASSETS_DIR / "table.xml")
table.pos = [0.0, 0.0, 0.7]
table.fixed = True

cube = mjk.SceneObject()
cube.name = "cube"
cube.shape = mjk.Shape.BOX
cube.size = [0.025, 0.025, 0.025]
cube.pos = [0.35, 0.10, 0.725]
cube.rgba = [0.1, 0.25, 1.0, 1.0]
cube.mass = 0.1
cube.condim = mjk.Condim.Torsional
cube.friction = [0.8, 0.02, 0.001]

spec.objects = [table, cube]
```

Objects are compiled before robots, in declaration order. That lets a robot
mount to a site exported by a previous object:

```python
mount = "table_top"   # the asset's own site name; SceneObject.prefix would prepend to it
robot_spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, mount)
```

After build, read authored object sites as frames:

```python
env = mjk.Env.build(spec)
world_t_table_top = env.site_frame("table_top")
surface_z = world_t_table_top.p.z()
```

Free objects must stay world-anchored because MuJoCo restricts free joints to
top-level bodies.

`spec.robots` can be empty. An object-only `Env` opens the viewer the same way:

```python
env = mjk.Env.build(spec)
viewer = mjk.Viewer()
viewer.open(env, "object scene")
```

## 7. Add Cameras

Add fixed scene cameras with `CameraSpec`. `pos` and `fovy` are required;
`quat` is `[x, y, z, w]` and defaults to identity `[0, 0, 0, 1]`.

```python
camera = mjk.CameraSpec()
camera.name = "task"
camera.pos = [0.1, -0.9, 1.45]
# extrinsic XYZ (35, 0, 5) deg: tilt down, yaw slightly right
camera.quat = [0.300420, 0.013117, 0.041601, 0.952809]
camera.fovy = 45.0
spec.cameras = [camera]
```

List compiled cameras after build. `use_camera(name)` switches a viewer or a recorder to one;
`use_camera("")` returns to the free camera:

```python
print([env.model.camera(i).name for i in range(env.model.ncam)])
viewer.use_camera("task")
```

## 8. Write Reset Hooks

`Env.reset()` resets MuJoCo, re-seeds every registered robot's ports from the reset state (so
stale commands do not hit the first post-reset step), runs your hook, then reads the
measurements back. A command the hook primes is kept; a POSITION robot the hook moves needs
its `jnt_pos_cmd` set there too. The viewer's reset button does the same. The hook gets a
`ResetContext` copy it may keep; an exception it raises comes out of `reset()` (or the
`env.step()` that ran a UI reset) once the measurements are read back.

```python
home = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
cube_start = [0.35, 0.10, 0.725]

def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos(home)
    env.set_body_pose("cube", cube_start)
    task_state["name"] = "HOME"
    task_state["step"] = 0

env.on_reset = on_reset

opts = mjk.ResetOptions()
opts.keyframe = 0
opts.use_keyframe = True
info = env.reset(opts)
```

Use reset hooks for robot pose, object pose, controller state, randomization,
and task state. Avoid hiding reset logic in the control loop.

## 9. Use The Viewer

`mjk.Viewer` is the C++ viewer: the same MuJoCo Studio window, menus, panels and keys as a
C++ program gets. `viewer.open(env, title)` opens it on the `Env`, with or without robots,
and raises `RuntimeError` without a display. The window runs on a thread of its own; your
loop still owns controller logic and its end condition. The window's pause, reset,
perturbation and edits reach the `Env` inside `env.step()`, which returns `False` once the
window is closed; `viewer.pace()` then sleeps out one timestep at the window's speed.

```python
viewer = mjk.Viewer()
viewer.open(env, "task")
while env.data.time < 10.0:
    env.update()
    robot.jnt_trq_cmd = gravity()
    if not env.step():
        break
    viewer.pace()
viewer.close()
```

`viewer.realtime_factor` is `pace()`'s speed (1.0 real time, 0.0 uncapped).
`key_pressed(key)` and `capture_key(key)` take `ImGuiKey` integer codes as in `imgui.h`;
`set_free_camera(distance, azimuth, elevation, lookat)` places the free camera. `Viewer` is
also a context manager.

### Draw A Live Trajectory Trace Overlay

Trace helpers on the `Env` draw line segments in world frame; the viewer draws them. Clear
them each frame and add the segments you want visible:

```python
trace = []
orange = [1.0, 0.5, 0.1, 1.0]

while env.data.time < 10.0:
    env.update()
    frame = kdl.Frame()
    fk.JntToCart(joints(robot.jnt_pos_msr), frame)
    trace.append([frame.p.x(), frame.p.y(), frame.p.z()])
    trace = trace[-1024:]

    env.clear_trace()
    for a, b in zip(trace, trace[1:]):
        env.add_trace_segment(a, b, orange)

    robot.jnt_trq_cmd = gravity()
    if not env.step():
        break
    viewer.pace()
```

`add_trace_segment()` and `add_overlay_arrow()` silently drop geoms once the overlay is
full (8192 geoms), so keep the trace bounded. `env.overlay_geoms()` returns them as
`mujoco.MjvGeom`s for a renderer of your own.

## 10. Record Video

`mjk.VideoRecorder` is the C++ recorder: Filament offscreen rendering, encoded to MP4 by
ffmpeg (on `PATH`), with no window needed; `ex_table_pour.py --record` uses it. Record after
stepping the state you want captured.

```python
with mjk.VideoRecorder.open_preset(env, "episode.mp4", mjk.VideoResolution.R720p, 60) as vr:
    vr.use_camera("task")
    for i in range(3000):
        env.update()
        robot.jnt_trq_cmd = gravity()
        env.step()
        if i % 8 == 0:                 # 60 fps at a 2 ms timestep
            vr.record_frame()
```

`VideoRecorder.open(env, out_path, width, height, fps)` takes an explicit size.
`VideoRecorder.open_offscreen(env, width, height)` writes no file; its `render_rgb()` returns
the current frame as a `(height, width, 3)` `uint8` numpy array.

## 11. Build A Tabletop Task Structure

A pick-place task is the same pieces assembled with a small state machine:

1. Build a scene with an arm, gripper, table asset, cube, and camera.
2. Read the table site to compute reliable tabletop coordinates.
3. Build PyKDL FK/IK/dynamics solvers.
4. Define a reset hook and state table.
5. Run torque impedance against state-specific joint targets.

### 11.1 Build The Scene

```python
spec = mjk.SceneSpec()
spec.timestep = 0.002
spec.use_stage = True

table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = str(mjk.ASSETS_DIR / "table.xml")
table.pos = [0.0, 0.0, 0.7]
table.fixed = True

cube = mjk.SceneObject()
cube.name = "cube"
cube.shape = mjk.Shape.BOX
cube.size = [0.025, 0.025, 0.025]
cube.pos = [0.35, 0.10, 0.725]
cube.rgba = [0.1, 0.25, 1.0, 1.0]
cube.mass = 0.1
cube.condim = mjk.Condim.Torsional
cube.friction = [0.8, 0.02, 0.001]
spec.objects = [table, cube]

gripper = mjk.AttachmentSpec()
gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

arm = mjk.RobotSpec()
arm.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
arm.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "table_top")
arm.attachments = [gripper]
spec.robots = [arm]

camera = mjk.CameraSpec()
camera.name = "task"
camera.pos = [0.1, -0.9, 1.45]
# extrinsic XYZ (35, 0, 5) deg: tilt down, yaw slightly right
camera.quat = [0.300420, 0.013117, 0.041601, 0.952809]
camera.fovy = 45.0
spec.cameras = [camera]

env = mjk.Env.build(spec)
```

### 11.2 Initialize Tool And Solvers

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base_mount"
tool.tcp_site = "g_pinch"
robot = env.create_robot("base_link", "bracelet_link", tool=tool)
robot.ctrl_mode = mjk.CtrlMode.TORQUE

chain = robot.kdl_chain()
fk = kdl.ChainFkSolverPos_recursive(chain)
ik_vel = kdl.ChainIkSolverVel_pinv(chain)
```

Use `robot.joint_limits` to construct joint-limit arrays for a PyKDL IK solver
when the solver requires them.

### 11.3 State Machine Shape

Keep state data separate from controller math:

```python
GRIPPER_CLOSED = 0.82  # the bundled 2F-85's ctrl is its driver angle [rad]

plan = [
    {"name": "HOME", "target": q_home, "duration": 1.0, "timeout": 2.5, "gripper": 0.0},
    {"name": "PICK", "target": q_pick, "duration": 1.5, "timeout": 3.5, "gripper": 0.0},
    {"name": "CLOSE", "target": q_pick, "duration": 0.8, "timeout": 1.5, "gripper": GRIPPER_CLOSED},
    {"name": "LIFT", "target": q_lift, "duration": 2.0, "timeout": 4.0, "gripper": GRIPPER_CLOSED},
]
```

On state entry, capture the measured joint position. During the state,
interpolate to the target and apply impedance:

```python
q_enter = list(robot.jnt_pos_msr)
t_enter = env.data.time

def interpolate(target, duration):
    alpha = min(max((env.data.time - t_enter) / duration, 0.0), 1.0)
    return [q0 + alpha * (q1 - q0) for q0, q1 in zip(q_enter, target)]

def apply_impedance(q_des):
    cmd = []
    for q, dq, q_target, g in zip(robot.jnt_pos_msr, robot.jnt_vel_msr, q_des, gravity()):
        cmd.append(g + 120.0 * (q_target - q) - 18.0 * dq)
    robot.jnt_trq_cmd = cmd
```

The bundled Robotiq gripper's actuator is controlled directly by name through `env.data`:

```python
env.data.actuator("g_fingers_actuator").ctrl[0] = state["gripper"]
```

### 11.4 Reset And Validate

```python
def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos(q_home)
    env.set_body_pose("cube", [0.35, 0.10, 0.725])
    task["index"] = 0
    task["entered_at"] = 0.0

env.on_reset = on_reset
env.reset()
```

Before treating a task as working, validate:

- IK waypoints solve before the simulation loop starts.
- Free objects can be reset with `set_body_pose()`.
- The gripper command holds the object without destabilizing the arm.
- The state machine recovers after a viewer reset.
- Headless and GUI loops follow the same controller path.

## 12. Build A Multi-Robot Scene

With more than one robot, give each its own prefix, so every robot's names are built the same
way:

```python
left = mjk.RobotSpec()
left.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
left.prefix = "r1_"
left.pos = [-0.7, 0.0, 0.0]
left.attachments = [gripper]

right = mjk.RobotSpec()
right.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
right.prefix = "r2_"
right.pos = [0.7, 0.0, 0.0]
right.attachments = [gripper]

spec.robots = [left, right]
env = mjk.Env.build(spec)

left_robot = env.create_robot("base_link", "bracelet_link", "r1_", tool=tool)
right_robot = env.create_robot("base_link", "bracelet_link", "r2_", tool=tool)
left_robot.name = "left"
right_robot.name = "right"
```

Each robot gets its own KDL chain and command ports while sharing the same
`Env`; one `env.update()` reads and commands both. `prefix` is prepended to every name the
call resolves (bodies, tool body, TCP site, F/T sensors), so one `ToolFrameSpec` serves both
arms, as in `ex_rnea_pick_place.py`; passing already-prefixed names (`"r2_base_link"`) with no
prefix is the same. `name` labels each robot in the viewer.

## 13. Modify A Running Scene

`Env.add_object()` and `Env.remove_object()` rebuild the native model/data. Use them for task setup and coarse changes, not per-frame spawning.

```python
obstacle = mjk.SceneObject()
obstacle.name = "obstacle"
obstacle.shape = mjk.Shape.BOX
obstacle.size = [0.05, 0.05, 0.20]
obstacle.pos = [0.25, -0.25, 0.90]
obstacle.rgba = [0.8, 0.2, 0.2, 1.0]
obstacle.mass = 0.5
obstacle.friction = [0.8, 0.02, 0.001]

env.add_object(obstacle)
env.update()
env.remove_object("obstacle")
```

Existing Python `Robot` handles follow the new model; a renderer built on the old
`env.model` must be rebuilt. A closed `Env` or `Robot` raises `RuntimeError` instead of leaving dangling native
pointers.

## 14. Grow Into The Examples

The Python examples mirror the C++ ones:

- `ex_gravity_comp`: single-arm gravity compensation.
- `ex_joint_ctrl`: a POSITION mode motion, then a VELOCITY mode motion back.
- `ex_table_pick_place`: IK waypoints, gripper command, phase table, table asset sites, a push
  on the arm mid-carry through `xfrc_applied`.
- `ex_table_pour`: gripper-held bottle asset and receiver; `--record` writes an MP4.
- `ex_rnea_pick_place`: two prefixed arms at one table with RNEA computed torque, free objects,
  scene cameras.
- `ex_achd_pick_place`, `ex_achd_table_slide`: ACHD torque control.
- `ex_admittance_ft`: F/T admittance around an RNEA task-space inner loop.

They live in `python/mj_kdl_wrapper/examples/` (installed as `mj_kdl_wrapper.examples`) and
end by themselves. They run headless by default and accept `--gui`.
