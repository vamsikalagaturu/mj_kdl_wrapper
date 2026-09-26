# Conventions {#page_conventions}

What the numbers in the API mean, and what persists between calls. The same holds in C++ and
Python, except the scene slots (`bind_scene_*()`), which only C++ has.

## Units

SI throughout: metres, radians, seconds, kilograms, newtons, newton-metres. The exceptions are
the camera angles people set by eye: `CameraSpec::fovy` and the `set_free_camera()` azimuth and
elevation are in degrees.

## Quaternions

Every quaternion in the API is `[x, y, z, w]` (the KDL and ROS order), identity
`{ 0, 0, 0, 1 }`: the `quat` of `RobotSpec`, `AttachmentSpec`, `SceneObject`, `SiteSpec` and
`CameraSpec`, and `set_body_pose()`. MuJoCo's `[w, x, y, z]` stays inside the wrapper; you only
meet it when you read `mjData::qpos` of a free joint yourself.

## Frames

| Value | Frame |
|---|---|
| `get_body_frame()`, `get_site_frame()`, free-body slot `pose` | world |
| `SceneSpec::floor_z`, trace segments | world |
| spec `pos`/`quat` | the parent they attach to (world, body, site or frame) |
| KDL chain, `robot.chain` / `robot.kdl_chain()` solvers | the chain's base body |
| `Robot::tip_T_tcp` | chain tip to the TCP site |
| wrench slot `wrench` | world axes, applied at the body's centre of mass (`xfrc_applied`) |
| F/T `wrench` | the MuJoCo `<force>`/`<torque>` sensors' site; `frame_site` names a frame to read with `get_site_frame()` / `env.site_frame()` |

## Joint ports

All port vectors are in KDL chain order (`joint_names`).

| Port | Meaning |
|---|---|
| `jnt_pos_msr` | `qpos` [rad] |
| `jnt_vel_msr` | `qvel` [rad/s] |
| `jnt_trq_msr` | `qfrc_actuator` [N m]: only the active mode's actuators produce force, so this is the drive torque in every mode |
| `jnt_pos_cmd`, `jnt_vel_cmd` | joint setpoints [rad], [rad/s] for POSITION, VELOCITY |
| `jnt_trq_cmd` | joint torque [N m] for TORQUE; `update()` divides by the actuator gear |
| `jnt_saturated` | the last command was clamped to the actuator's `ctrlrange` |

`joint_force_limits()` gives each joint's torque limit in the active mode. In Python the
ports read as read-only numpy copies; assign a whole vector to write one.

## Scene actuator slots

`SceneActuatorSlot::command` is in the actuator's own ctrl units, before its gear: the 2F-85's
ctrl is its driver joint angle (0 open, 0.82 rad closed), and a motor with gear 4 turns
command 1 into 4 N m. `update()` clamps it to `ctrlrange` and sets `saturated`.

## Physics options

`<option>` is global to a scene, so it comes from one place: the first robot's own MJCF, with
`SceneSpec::timestep` and `gravity_z` on top. MuJoCo drops the `<option>` of every attachment
and later robot with an "Attach conflict" warning; the 2F-85's `cone="elliptic"
impratio="10"` is one of them. `env.model->opt` can be changed after `init_env()`; a rebuild
recompiles the model and drops that change.

## Stepping

`step()` is MuJoCo's `mj_step()` split as `mj_step2()` then `mj_step1()`. Afterwards `qpos`,
body and site frames and position/velocity sensors describe the same, new instant; force and
acceleration sensors describe the step just taken. `step()` never sleeps (`pace_realtime()`
does) and returns `false` only when the viewer window closes, so a headless loop needs its own
end condition.

## What persists

MuJoCo keeps `ctrl`, `xfrc_applied` and `qfrc_applied` until something overwrites them. The
wrapper writes only what it owns:

- `ctrl` of each registered robot's active-mode actuators and of each bound actuator slot, every
  `update()`;
- `xfrc_applied` of each bound wrench slot, every `update()` (a zero wrench clears it), and of
  the body the viewer's perturbation drags;
- never `qfrc_applied`.

`reset()` resets MuJoCo, re-seeds every robot port and scene slot (commands hold the reset pose,
wrench slots zero), runs `Env::on_reset`, then reads the measurements back. A command the hook
primes is kept.

A rebuild (`scene_add_object()` / `scene_remove_object()`) is not a reset: the time,
`qpos`/`qvel` (by joint name and type) and `act`/`ctrl` (by actuator name) carry over, and every
port command is kept.
