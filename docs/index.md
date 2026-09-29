# Documentation {#page_documentation}

- [C++ Tutorial](@ref page_tutorial_cpp)
- [Python Tutorial](@ref page_tutorial_python)
- [C++ API Guide](@ref page_api_cpp)
- [Python Bindings API Guide](@ref page_api_python)
- [Examples](@ref page_examples)
- [Conventions: units, frames, quaternions, what persists](@ref page_conventions)
- [Torque Control and Tool Inertia](@ref page_howto_torque_control)
- [Importing a URDF Robot](@ref page_howto_urdf)
- [Loop Pacing and the Real-Time Factor](@ref page_howto_pacing)

## Migrating to 0.4 {#sec_migrate_env}

The `Env` now owns everything the loop touches: the model/data, the robots, the scene slots
(`env.scene`) and the viewer (`env.viewer`). The loop and its calls take the `Env`:

| 0.3.x | 0.4 |
|-------|-----|
| `init_robot_from_mjcf(&r, model, data, ...)` + `env_add_robot(&env, &r)` | `init_robot_from_mjcf(&r, &env, ...)` (registers it) |
| `init_window_sim(&viewer, &robot)` / `(&viewer, m, d)` | `open_viewer(&env, title)` |
| `step(&robot)`, `step_n()`, `step(&viewer, m, d)` | `step(&env)` |
| `update(&robot)`, `read_measurements()`, `apply_commands()` | `update(&env)` (every robot and scene slot) |
| `pace_realtime(&robot)` / `(&viewer, m)` | `pace_realtime(&env)` |
| `get_body_frame(m, d, ...)`, `get_site_frame`, `set_body_pose(m, d, ...)` | same names, taking `&env` |
| `init_scene_state()`, `read/apply/rebind_scene_state()` | `bind_scene_*(&env.scene, ...)`; `update(&env)` and rebuilds handle the rest |
| `scene_add_object(&m, &d, &spec, ...)` | `scene_add_object(&env, ...)` |
| `record_frame(&vr, m, d)`, `render_rgb(&vr, m, d, out)` | `record_frame(&vr, &env)`, `render_rgb(&vr, &env, out)` |
| `set_joint_pos(&r, q, call_forward)`, `mark_kinematics_*()` | `set_joint_pos(&r, q)`; frames notice state changes themselves |
| `init_window()`, `render()`, `cleanup(&viewer)` | removed; `cleanup(&env)` closes the viewer |

`reset(&env)` now re-seeds every robot's ports and F/T readings and every scene slot, then runs
`on_reset`, and the Simulate UI's reset button does the same. A `Robot`'s MuJoCo index maps are
private.

Also changed in 0.4:

| Change | What to do |
|--------|------------|
| Calls that can fail return `mj_kdl::Status` instead of `bool` | `if (!s)` still works; print `s.error` |
| Spec string fields (`path`, `prefix`, `mjcf_path`, `tool_body`, ...) are `std::string` | drop `.c_str()`; empty means not set |
| `set_body_pose()` takes `[x, y, z, w]` (was `[w, x, y, z]`) | reorder the quaternion |
| Control modes are MuJoCo actuator groups (`RobotSpec::modes`, `set_control_mode()`) | nothing is written to `qfrc_applied` any more |
| Unset required numbers (`timestep`, primitive `size`/`mass`/`friction`/`rgba`, camera `pos`/`fovy`) fail the build | set them |
| Earlier docs used `tool_body = "g_base"` for the 2F-85, which leaves the mount's mass out of KDL | use `"g_base_mount"` |
| Python: joint ports read as read-only numpy arrays | assign whole vectors; `port[i] = x` raises |
| Python: `Env.save_xml()` -> `save_model_xml()`, `Robot.tip_to_tcp` -> `tip_T_tcp` | rename; there are no aliases |
| `get_camera_names()` removed | loop `i < model->ncam` with `mj_id2name(model, mjOBJ_CAMERA, i)` |
| `get_joint_position()` / `get_joint_velocity()` removed | a `bind_scene_joint()` slot's `position`/`velocity`, or `model`/`data` directly |
| `realtime_factor_of()` removed | read `Viewer::realtime_factor` |
| `use_camera(VideoRecorder*)` / `set_free_camera(VideoRecorder*)` removed (the `Viewer` overloads stay) | write `vr.cam`: `type = mjCAMERA_FIXED`, `fixedcamid = mj_name2id(model, mjOBJ_CAMERA, name)`; or `mjv_defaultFreeCamera()` |
| `scene_object_site_name()` removed (C++ and Python) | the asset's own site name; objects are no longer auto-prefixed: set `SceneObject::prefix` (Python `SceneObject.prefix`) to prefix them |
| `attach_to_spec()`, `add_floor_to_spec()`, `add_skybox_to_spec()`, `add_objects_to_spec()`, `compile_and_make_data()`, `ensure_plugins_loaded()` are internal | go through `SceneSpec` and `build_scene()`; to run on your own pair, set `Env::adopt` |
| Python: `Env.time()`, `timestep()`, `camera_names()`, `save_binary()`, `mjk.mujoco_version()` removed | `env.data.time`, `env.model.opt.timestep`, `[env.model.camera(i).name for i in range(env.model.ncam)]`, `mujoco.mj_saveModel(env.model, path, None)`, `mjk.__mujoco_version__` |
| Python: `Env.actuator_ctrl()`, `set_actuator_ctrl()`, `has_actuator()`, `set_body_wrench()` removed | `env.data.actuator(n).ctrl[0]` (read or assign), `env.model.actuator(n)` (`KeyError` when missing), `env.data.body(n).xfrc_applied[:] = [*f, *t]` |
| Python: `Robot.fk_frame()`, `gravity_torques()` removed | PyKDL `ChainFkSolverPos_recursive` / `ChainDynParam` on `robot.kdl_chain()` with a `JntArray` of `robot.jnt_pos_msr` |
| Python: `Robot.ft_sensor_frame(name)` removed | `env.site_frame(<the sensor's frame_site>)` |
| `find_ft_sensor()` removed | search `robot.ft_sensors` (in `ToolFrameSpec::ft_sensors` order) by `.name` |
| MJCF `SceneObject`s are no longer prefixed with `name + "_"` | use the asset's own element names; set `SceneObject::prefix` when an asset is used twice |
| `LogLevel` is a threshold, `INFO` < `WARN` < `ERROR` < `NONE`, default `INFO` (it was a verbosity where `ERROR` printed everything) | `set_log_level(WARN)` now prints warnings and errors; the default prints what it did before |
| `LOG_INFO()` / `LOG_WARN()` / `LOG_ERROR()` | `MJ_LOG_INFO()` / `MJ_LOG_WARN()` / `MJ_LOG_ERROR()` |
| The header no longer includes `<GLFW/glfw3.h>` | include it yourself for the `GLFW_KEY_*` codes of `key_pressed()` / `capture_key()` |
| `init_env()` dropped an `on_reset` set before it | it keeps it, like `adopt` |
| `scene_add_object()` / `scene_remove_object()` started from a fresh `mjData` at `qpos0` | the time, `qpos`/`qvel` (by joint name and type) and `act`/`ctrl` (by actuator name) carry over into the new model; call `reset(&env)` for the old behaviour |
| A rebuild whose robot or F/T sensor rebinding failed left the `Env` half rebound | it is left unchanged and the call returns the error |
| `scene_add_object()` / `scene_remove_object()` could be called from `on_reset` | they fail there; change the scene outside the hook |
| A `VideoRecorder` kept rendering the old model after a rebuild | it builds its render context at the first frame and follows rebuilds |
| `init_offscreen()` / `init_video_recorder()` on an open recorder leaked it | they free it first |
| `open_viewer()` without a usable display could abort on the render thread | it returns an error `Status` when no window can be created |
| Simulate UI: a reset was detected from time jumping back (also on API reset, rebuild, history scrub, the Load key) | only an explicit UI Reset runs the reset path and `on_reset`; scrubbing and the Load key no longer reset |
| Simulate UI: the right arrow single-stepped MuJoCo behind the wrapper; a dropped model file was loaded | the right arrow no longer single-steps; drag-and-drop loading is gone (build scenes with `SceneSpec`) |
| CMake: `find_package(mj_kdl_wrapper 0.3)` accepted any 0.x (`SameMajorVersion`) | the minor version must match (`SameMinorVersion`); ask for `0.4` |
| Install: licenses went to `lib/`, `image_io.hpp` and `camera_ros.hpp` were always installed | licenses in `share/doc/mj_kdl_wrapper`; only the headers of what was built |
| New: pkg-config file `mj_kdl_wrapper.pc` | `pkg-config --cflags --libs mj_kdl_wrapper`; see the [Standalone Installation Guide](install/standalone.md) |
| Python: `Scene` removed | `Env.build(spec)`; its frame, pose, object and save calls are on `Env` |
| Python: `Robot.from_scene(scene, ...)` | `env.create_robot(...)` or `env.create_robot_from_chain(...)` |
| Python: `Robot.update()`, `step()`, `pace()`, `step_n()`, `Scene.step()`, `step_n()` | `env.update()`, `env.step()`, `env.pace()`; loop for several steps |
| Python: `SimulateViewer.open(robot_or_scene, title)`, `.close()`, `.step()`, `.pace()`, `.step_n()` | `env.open_viewer(title)`; `env.viewer` has `is_running()` and the trace, camera and key calls; `env.close()` closes it |
| Python: `Robot.set_joint_pos(q, call_forward)` | `set_joint_pos(q)` |
| Python: `VideoRecorder.open(scene, ...)` / `open_preset(scene, ...)` | pass the `Env`: `open(env, ...)` |
| Python: `Env.spec` was a writable copy whose changes did nothing | it is read-only; change the scene with `add_object()` / `remove_object()` |
| Python: `Env.on_reset` got a `ResetContext` that dangled once the call returned | it gets a copy, safe to keep |
| Python: `menagerie.model_path()` knew only `kinova_gen3` and `robotiq_2f85` | also `universal_robots_ur5e` and `universal_robots_ur10e` |

## Migrating from 0.3.1 {#sec_migrate_pacing}

`step()` no longer sleeps. Pacing moved out of the physics call and into `pace_realtime()`, so a
loop that owns its own timing is no longer fought by a hidden sleep inside `step()`. A windowed
run that should track wall time needs one `pace_realtime(&env)` call per iteration; headless
runs are unaffected, because pacing only ever happened when a viewer existed. See
[Loop Pacing and the Real-Time Factor](@ref page_howto_pacing).

## Migrating from 0.2.x {#sec_migrate_quat}

0.3.0 removes the `euler` placement field from `RobotSpec`, `AttachmentSpec`,
`SceneObject` and `CameraSpec`, and from `attach_child()`. Placement orientation
is now the quaternion `quat`, in `[x, y, z, w]` order, with identity
`{ 0, 0, 0, 1 }`. There is no compatibility shim: code setting `euler` no longer
compiles.

`quat` is the library's own convention. MuJoCo's `[w, x, y, z]` ordering stays
inside the wrapper and is never exposed through these specs.

C++, a 180-degree flip about x:

```cpp
// 0.2.x
cam.euler = { 180.0, 0.0, 0.0 };
// 0.3.0
cam.quat = { 1.0, 0.0, 0.0, 0.0 };
```

Python, the same rotation:

```python
# 0.2.x
cam.euler = [180.0, 0.0, 0.0]
# 0.3.0
cam.quat = [1.0, 0.0, 0.0, 0.0]
```

For angles without an exact quaternion, convert with SciPy --
`Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()` returns
`[x, y, z, w]` directly.
