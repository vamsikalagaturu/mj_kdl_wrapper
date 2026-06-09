# Python Tutorial

This tutorial builds a Python simulation application in layers: create a scene,
initialize a KDL-backed robot handle, run position and torque control, add
attachments, objects, cameras, reset hooks, the Simulate UI, recording, and a
small tabletop task structure.

Installation stays in the README. The snippets below assume `mj_kdl_wrapper`,
`PyKDL`, and the matching `mujoco` package import successfully, and that the
MuJoCo Menagerie models used by the examples are available through
`mjk.menagerie.model_path()`.

## How To Read This Tutorial

The Python API has the same layers as the C++ wrapper:

| Layer | Type | Responsibility |
|-------|------|----------------|
| Scene description | `SceneSpec`, `RobotSpec`, `AttachmentSpec`, `SceneObject`, `CameraSpec` | Describe what should be compiled into MuJoCo |
| Runtime owner | `Scene` or `Env` | Own the compiled MuJoCo model/data |
| Robot control handle | `Robot` | KDL chain, joint maps, measured ports, command ports |
| Visualization/recording | `SimulateViewer`, `VideoRecorder` | Interactive Simulate UI and offscreen MP4 recording |
| KDL interop | `PyKDL` | FK, IK, dynamics solvers over the wrapper-built chain |

Use `Scene` for simple one-off simulations. Use `Env` when you need reset hooks
or registered robots that stay synchronized across resets and scene rebuilds.

Most applications follow this flow:

1. Build a `SceneSpec`.
2. Call `Env.build(spec)` or `Scene.build(spec)`.
3. Create one or more `Robot` handles.
4. Install `env.on_reset` if task state must be restored.
5. In the loop: `update()`, compute commands, write command ports, then `step()`.
6. Close viewer/recorder and environment when done.

## Complete Runnable Script

The code below is a complete interactive script. If you save it as
`python_tutorial_demo.py` in the repository root and run it from an environment
where `mj_kdl_wrapper` is installed, it builds a table scene, initializes a
tool-aware KDL robot, opens the Simulate UI, and runs gravity compensation.
Pass `--headless` to run the same controller without a window for CI or remote
machines without a display.

Later sections break this same structure into focused excerpts.

```python
from __future__ import annotations

import math
import argparse
from pathlib import Path

import mj_kdl_wrapper as mjk


HOME_POSE = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
TABLE_PATH = Path("assets/table.xml")
SURFACE_Z = 0.7
CUBE_HALF = 0.025


def require_path(path: str | Path, label: str) -> str:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return str(resolved)


def make_gripper() -> mjk.AttachmentSpec:
    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = require_path("assets/robotiq_2f85/2f85.xml", "gripper model")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
    gripper.prefix = "g_"
    return gripper


def make_table() -> mjk.SceneObject:
    table = mjk.SceneObject()
    table.name = "table"
    table.mjcf_path = require_path(TABLE_PATH, "table asset")
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
    spec.add_floor = True
    spec.add_skybox = True

    table = make_table()
    spec.objects = [table, make_cube()]

    arm = mjk.RobotSpec()
    arm.path = require_path(mjk.menagerie.model_path("kinova_gen3"), "arm model")
    arm.attach_to = mjk.AttachTarget(
        mjk.AttachKind.Site,
        mjk.scene_object_site_name(table, "table_top"),
    )
    arm.attachments = [make_gripper()]
    spec.robots = [arm]

    camera = mjk.CameraSpec()
    camera.name = "task"
    camera.pos = [0.1, -0.9, 1.45]
    camera.euler = [35.0, 0.0, 5.0]
    camera.fovy = 45.0
    spec.cameras = [camera]

    env = mjk.Env.build(spec)

    tool = mjk.ToolFrameSpec()
    tool.tool_body = "g_base"
    tool.tcp_site = "g_pinch"
    robot = env.create_robot("base_link", "bracelet_link", tool=tool)
    return env, robot


def run_controller(env: mjk.Env, robot: mjk.Robot) -> None:
    robot.update()
    robot.jnt_trq_cmd = robot.gravity_torques(env.spec.gravity_z)
    if env.has_actuator("g_fingers_actuator"):
        env.set_actuator_ctrl("g_fingers_actuator", 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without opening the Simulate UI")
    parser.add_argument("--duration", type=float, default=2.0, help="headless run duration in seconds")
    args = parser.parse_args()

    env, robot = build_env()
    try:
        robot.ctrl_mode = mjk.CtrlMode.TORQUE

        def on_reset(ctx: mjk.ResetContext) -> None:
            robot.set_joint_pos(HOME_POSE, call_forward=False)
            env.set_body_pose("cube", [0.35, 0.10, SURFACE_Z + CUBE_HALF])

        env.on_reset = on_reset
        env.reset()

        robot.update()
        start = robot.fk_frame().p
        start_xyz = [start.x(), start.y(), start.z()]

        if args.headless:
            end_time = env.time() + args.duration
            while env.time() < end_time:
                run_controller(env, robot)
                robot.step()
        else:
            viewer = mjk.SimulateViewer.open(robot, "python tutorial")
            viewer.use_camera("task")
            try:
                while viewer.is_running():
                    run_controller(env, robot)
                    if not viewer.step():
                        break
            finally:
                viewer.close()

        end = robot.fk_frame().p
        end_xyz = [end.x(), end.y(), end.z()]
        drift = math.sqrt(sum((b - a) ** 2 for a, b in zip(start_xyz, end_xyz)))
        print(f"MuJoCo {mjk.mujoco_version()}")
        print(f"joints: {robot.n_joints}")
        print(f"cameras: {' '.join(env.camera_names())}")
        print(f"EE drift: {drift:.6f} m")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 1. Start With One Robot Scene

`SceneSpec.timestep`, `add_floor`, and `add_skybox` are required. They are
explicit choices, not defaults the wrapper invents.

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

`Env` owns native MuJoCo resources. Call `env.close()` when you want deterministic
cleanup. For simple scripts, a `try/finally` block keeps that explicit:

```python
try:
    ...
finally:
    env.close()
```

## 2. Initialize A Robot

`Robot` is the handle for one controllable articulation. It stores KDL chain
metadata, joint names/limits, measured ports, and command ports.

```python
robot = env.create_robot("base_link", "bracelet_link")
print(robot.n_joints)
print(robot.joint_names)
```

All joint vectors are in KDL chain order:

- `jnt_pos_msr`, `jnt_vel_msr`, `jnt_trq_msr` are measured ports.
- `jnt_pos_cmd` is used in `CtrlMode.POSITION`.
- `jnt_trq_cmd` is used in `CtrlMode.TORQUE`.

## 3. Run Position Control

In position mode, write `jnt_pos_cmd`. `update()` reads measured state and
applies the command ports to MuJoCo.

```python
robot.ctrl_mode = mjk.CtrlMode.POSITION
robot.update()
robot.jnt_pos_cmd = list(robot.jnt_pos_msr)

end_time = env.time() + 2.0
while env.time() < end_time:
    robot.update()
    robot.jnt_pos_cmd = list(robot.jnt_pos_msr)
    robot.step()
```

`robot.step()` advances the owning scene by one MuJoCo timestep. For scenes
without a robot control loop, `env.time()` and `env.timestep()` still expose
the runtime clock.

## 4. Add KDL Gravity Compensation

The Python binding exposes a convenience gravity helper for the common case:

```python
robot.ctrl_mode = mjk.CtrlMode.TORQUE

while env.time() < end_time:
    robot.update()
    robot.jnt_trq_cmd = robot.gravity_torques(env.spec.gravity_z)
    robot.step()
```

For other KDL solvers, use the standard `PyKDL` module and the wrapper-built
chain:

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

`robot.fk_frame()` is available for simple FK reads and returns a `PyKDL.Frame`.

## 5. Attach A Gripper Or Tool

Attachments are MJCF assets attached under a body, site, or frame in the
accumulated robot spec. They are applied in order.

```python
gripper = mjk.AttachmentSpec()
gripper.mjcf_path = "assets/robotiq_2f85/2f85.xml"
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

robot_spec = mjk.RobotSpec()
robot_spec.path = mjk.menagerie.model_path("kinova_gen3")
robot_spec.attachments = [gripper]

spec.robots = [robot_spec]
env = mjk.Env.build(spec)
```

If the model has no useful mount site, attach by body name and add offsets:

```python
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Body, "bracelet_link")
gripper.pos[2] = -0.061525
gripper.euler[0] = 180.0
```

Tell KDL about the attached tool when creating the robot:

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base"
tool.tcp_site = "g_pinch"

robot = env.create_robot("base_link", "bracelet_link", tool=tool)
assert robot.has_tcp_frame
```

`tool_body` identifies the attached subtree whose inertia should be included in
the KDL chain. `tcp_site` becomes the terminal frame for FK and task-space code.

## 6. Add Tables, Objects, And Asset Sites

`SceneObject` supports primitive objects and MJCF-backed assets. MJCF assets use
`mjcf_path`; primitive objects require `shape`, `size`, `rgba`, `mass`, and
`friction`.

```python
table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = "assets/table.xml"
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
mount = mjk.scene_object_site_name(table, "table_top")
robot_spec.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, mount)
```

After build, read authored object sites as frames:

```python
env = mjk.Env.build(spec)
world_t_table_top = env.site_frame(mjk.scene_object_site_name(table, "table_top"))
surface_z = world_t_table_top.p.z()
```

Free objects must stay world-anchored because MuJoCo restricts free joints to
top-level bodies.

## 7. Add Cameras

Add fixed scene cameras with `CameraSpec`. `pos` and `fovy` are required;
`euler` defaults to identity.

```python
camera = mjk.CameraSpec()
camera.name = "task"
camera.pos = [0.1, -0.9, 1.45]
camera.euler = [35.0, 0.0, 5.0]
camera.fovy = 45.0
spec.cameras = [camera]
```

List compiled cameras after build. A viewer or recorder can switch to any
compiled fixed camera after it is opened:

```python
print(env.camera_names())
```

Pass `""` to return to the free camera.

## 8. Write Reset Hooks

`Env.reset()` resets MuJoCo, runs your hook, forwards dynamics, then syncs
registered robot command ports so stale commands do not hit the first post-reset
step.

```python
home = [0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708]
cube_start = [0.35, 0.10, 0.725]

def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos(home, call_forward=False)
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

## 9. Use The Simulate UI

`SimulateViewer.open(robot)` starts the custom Simulate UI. The viewer owns the
window; your loop still owns controller logic.

```python
viewer = mjk.SimulateViewer.open(robot, "task")
viewer.use_camera("task")

try:
    while viewer.is_running():
        robot.update()
        robot.jnt_trq_cmd = robot.gravity_torques(env.spec.gravity_z)
        if not viewer.step():
            break
finally:
    viewer.close()
```

The viewer exposes the same wrapper panels as C++: `Frames`, `Trace`, `Perturb`,
`Recorder`, and `RTF`. `viewer.realtime_factor` controls pacing; `1.0` is
real time and `0.0` runs as fast as possible.

### Draw A Live Trajectory Trace Overlay

Trace helpers draw line segments in world frame. Clear the user scene each frame
and add the segments you want visible:

```python
trace = []
orange = [1.0, 0.5, 0.1, 1.0]

while viewer.is_running():
    robot.update()
    frame = robot.fk_frame()
    trace.append([frame.p.x(), frame.p.y(), frame.p.z()])
    trace = trace[-1024:]

    viewer.clear_trace()
    for a, b in zip(trace, trace[1:]):
        viewer.add_trace_segment(a, b, orange)

    robot.jnt_trq_cmd = robot.gravity_torques(env.spec.gravity_z)
    if not viewer.step():
        break
```

`add_trace_segment()` silently drops segments once the user-scene geometry
buffer is full, so keep the trace bounded.

## 10. Record Video

Headless recording uses `VideoRecorder`. Call `record_frame()` after stepping
the simulation state you want captured.

```python
recorder = mjk.VideoRecorder.open_preset(
    env,
    "episode.mp4",
    mjk.VideoResolution.R720p,
    fps=60,
)
recorder.use_camera("task")

try:
    for _ in range(3000):
        robot.update()
        robot.jnt_trq_cmd = robot.gravity_torques(env.spec.gravity_z)
        robot.step()
        recorder.record_frame()
finally:
    recorder.close()
```

Use `VideoRecorder.open(env, path, width, height, fps)` when you need explicit
frame dimensions instead of a preset.

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
spec.add_floor = True
spec.add_skybox = True

table = mjk.SceneObject()
table.name = "table"
table.mjcf_path = "assets/table.xml"
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
gripper.mjcf_path = "assets/robotiq_2f85/2f85.xml"
gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")
gripper.prefix = "g_"

arm = mjk.RobotSpec()
arm.path = mjk.menagerie.model_path("kinova_gen3")
arm.attach_to = mjk.AttachTarget(
    mjk.AttachKind.Site,
    mjk.scene_object_site_name(table, "table_top"),
)
arm.attachments = [gripper]
spec.robots = [arm]

camera = mjk.CameraSpec()
camera.name = "task"
camera.pos = [0.1, -0.9, 1.45]
camera.euler = [35.0, 0.0, 5.0]
camera.fovy = 45.0
spec.cameras = [camera]

env = mjk.Env.build(spec)
```

### 11.2 Initialize Tool And Solvers

```python
tool = mjk.ToolFrameSpec()
tool.tool_body = "g_base"
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
plan = [
    {"name": "HOME", "target": q_home, "duration": 1.0, "timeout": 2.5, "gripper": 0.0},
    {"name": "PICK", "target": q_pick, "duration": 1.5, "timeout": 3.5, "gripper": 0.0},
    {"name": "CLOSE", "target": q_pick, "duration": 0.8, "timeout": 1.5, "gripper": 255.0},
    {"name": "LIFT", "target": q_lift, "duration": 2.0, "timeout": 4.0, "gripper": 255.0},
]
```

On state entry, capture the measured joint position. During the state,
interpolate to the target and apply impedance:

```python
q_enter = list(robot.jnt_pos_msr)
t_enter = env.time()

def interpolate(target, duration):
    alpha = min(max((env.time() - t_enter) / duration, 0.0), 1.0)
    return [q0 + alpha * (q1 - q0) for q0, q1 in zip(q_enter, target)]

def apply_impedance(q_des):
    robot.update()
    gravity = robot.gravity_torques(env.spec.gravity_z)
    cmd = []
    for q, dq, q_target, g in zip(robot.jnt_pos_msr, robot.jnt_vel_msr, q_des, gravity):
        cmd.append(g + 120.0 * (q_target - q) - 18.0 * dq)
    robot.jnt_trq_cmd = cmd
```

The Robotiq Menagerie gripper actuator is controlled directly by name:

```python
if env.has_actuator("g_fingers_actuator"):
    env.set_actuator_ctrl("g_fingers_actuator", state["gripper"])
```

### 11.4 Reset And Validate

```python
def on_reset(ctx: mjk.ResetContext) -> None:
    robot.set_joint_pos(q_home, call_forward=False)
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

Use prefixes to disambiguate the second robot:

```python
left = mjk.RobotSpec()
left.path = mjk.menagerie.model_path("kinova_gen3")
left.pos = [-0.7, 0.0, 0.0]
left.attachments = [gripper]

right = mjk.RobotSpec()
right.path = mjk.menagerie.model_path("kinova_gen3")
right.prefix = "r2_"
right.pos = [0.7, 0.0, 0.0]
right.attachments = [gripper]

spec.robots = [left, right]
env = mjk.Env.build(spec)

left_robot = env.create_robot("base_link", "bracelet_link", tool=tool)
right_robot = env.create_robot("base_link", "bracelet_link", prefix="r2_", tool=tool)
```

Each robot gets its own KDL chain and command ports while sharing the same
MuJoCo scene.

## 13. Modify A Running Scene

`Scene.add_object()`, `Env.add_object()`, `remove_object()` rebuild the native
model/data. Use them for task setup and coarse changes, not per-frame spawning.

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
robot.update()
env.remove_object("obstacle")
```

Existing Python `Robot` handles are rebound automatically. A closed `Scene`,
`Env`, or `Robot` raises `RuntimeError` instead of leaving dangling native
pointers.

## 14. Grow Into The Examples

The included Python examples mirror the C++ examples:

- `ex_gravity_comp`: single-arm gravity compensation.
- `ex_table_scene`: table asset, primitive objects, cameras, reset hook.
- `ex_pick`: IK waypoints, gripper command, state machine.
- `ex_table_pick_place`: tabletop pick/place using table asset sites.
- `ex_table_pour`: gripper-held bottle asset and receiver.
- `ex_dual_arm`: two prefixed robots in one scene.
- `ex_record`: headless MP4 recording.

Read the corresponding files in `../../python/examples/` when you want complete,
runnable versions of the patterns above.
