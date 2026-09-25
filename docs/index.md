# Documentation {#page_documentation}

- [C++ Tutorial](@ref page_tutorial_cpp)
- [Python Tutorial](@ref page_tutorial_python)
- [C++ API Guide](@ref page_api_cpp)
- [Python Bindings API Guide](@ref page_api_python)
- [Examples](@ref page_examples)
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
| `get_body_frame(m, d, ...)`, `get_site_frame`, `get_joint_*`, `set_body_pose(m, d, ...)` | same names, taking `&env` |
| `init_scene_state()`, `read/apply/rebind_scene_state()` | `bind_scene_*(&env.scene, ...)`; `update(&env)` and rebuilds handle the rest |
| `scene_add_object(&m, &d, &spec, ...)` | `scene_add_object(&env, ...)` |
| `record_frame(&vr, m, d)`, `render_rgb(&vr, m, d, out)` | `record_frame(&vr, &env)`, `render_rgb(&vr, &env, out)` |
| `set_joint_pos(&r, q, call_forward)`, `mark_kinematics_*()` | `set_joint_pos(&r, q)`; frames notice state changes themselves |
| `init_window()`, `render()`, `cleanup(&viewer)` | removed; `cleanup(&env)` closes the viewer |

`reset(&env)` now re-seeds every robot's ports and F/T readings and every scene slot, and the
Simulate UI's reset button does the same. A `Robot`'s MuJoCo index maps are private.

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
