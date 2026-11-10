# Migrating to 0.6 {#page_migration}

## From 0.5 {#sec_migrate_mujoco4}

MuJoCo 4.0.0, built from source; the viewer and the recorder are optional libraries of their own:

| 0.5 | 0.6 |
|-----|-----|
| MuJoCo 3.14.0 release tarball downloaded into `~/.cache/mj_kdl_wrapper/mujoco-*` (`MJ_KDL_FETCH_MUJOCO`, `MJ_KDL_MUJOCO_URL`, sha256 pin) | `vcs import < mj_kdl_wrapper.repos` checks out MuJoCo 4.0.0 into `third_party/mujoco`; CMake builds it. `MJ_KDL_MUJOCO_DIR` names a MuJoCo source checkout, not an install |
| apt `libglfw3-dev libgl-dev` | not needed; `libegl-dev` for render, the X11/Wayland/GL headers (`libgl1-mesa-dev libwayland-dev libxkbcommon-dev libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libfontconfig-dev`) for the viewer |
| One library `mj_kdl_wrapper` with the Simulate UI and the EGL recorder | core `mj_kdl_wrapper` (no GL, no window), `mj_kdl_wrapper::render` (`render.hpp`) and `mj_kdl_wrapper::viewer` (`viewer.hpp`); `MJ_KDL_WITH_RENDER` / `MJ_KDL_WITH_VIEWER` (`AUTO`/`ON`/`OFF`, the viewer needs render); `find_package(mj_kdl_wrapper COMPONENTS render viewer)`; pkg-config covers the core only |
| `BUILD_RECORDER`, `SHOW_EQUALITY_PANEL`, `SHOW_GROUP_PANEL` | removed |
| `VideoRecorder`, `init_video_recorder()`, `init_offscreen()`, `record_frame()`, `render_rgb()` in `mj_kdl_wrapper.hpp`, rendered through EGL | the same calls in `render.hpp`, link `mj_kdl_wrapper::render`; rendered with MuJoCo's Filament renderer |
| `image_io.hpp`, `write_png_rgb()` | removed |
| `env.viewer`, `open_viewer(&env, title)` | the caller owns it: `mj_kdl::Viewer viewer;` then `open_viewer(&viewer, &env, title)`, a MuJoCo Studio window; `cleanup(&viewer)` or `cleanup(&env)` closes it |
| `pace_realtime(&env)`, `is_running(&env.viewer)`, `use_camera(&env.viewer, ...)`, `set_free_camera(&env.viewer, ...)` | pass `&viewer` |
| `key_pressed()` / `capture_key()` take `GLFW_KEY_*` codes | `ImGuiKey_*` codes from `<imgui.h>` |
| `clear_trace(&env.viewer)`, `add_trace_segment(&env.viewer, ...)`, `add_overlay_arrow(&env.viewer, ...)`, no-ops headless | take `&env`; the viewer and every recorder draw them; `overlay_geoms(&env)` copies them |
| -- | `step()` applies the viewer's pause, reset, perturbation, edited controls (they win over `update()` until a reset) and physics options; components follow an `Env` through `Env::hooks` (`EnvHooks`) |
| `SceneSpec::add_floor`, `SceneSpec::add_skybox` (both required) | `SceneSpec::use_stage` (required; set it to the old `add_floor`, drop `add_skybox`): `true` builds on a stage MJCF with floor, sky, haze and lights, `false` on an empty world. Optional `SceneSpec::stage` names the stage file; empty means the bundled `assets/stage/default.xml`. `floor_z` is added to the height of every plane on the stage's worldbody. Python alike |
| Python >= 3.10, `mujoco==3.14.0` | Python >= 3.12, `mujoco==4.0.0`; the wheel builds the core, render and viewer from the `third_party/mujoco` source (`vcs import` and the viewer's apt headers first), against pip `mujoco-core==4.0.0` |
| Python `env.open_viewer(title)`, `env.viewer`, `env.pace()`, `mjk.Viewer` | the caller owns it, as in C++ (the same Studio window): `viewer = mjk.Viewer()`, `viewer.open(env, title)` (raises without a display); `env.step()` (`False` once closed), `viewer.pace()`; `viewer.close()`. `key_pressed()` / `capture_key()` take `ImGuiKey` integer codes; `use_camera()`, `set_free_camera()`, `realtime_factor`, `fps` |
| Python `env.viewer.clear_trace()`, `add_trace_segment()` | `env.clear_trace()`, `env.add_trace_segment(a, b, rgba)`, `env.add_overlay_arrow(from_, dir, length, rgba)`, `env.overlay_geoms()` |
| Python `mjk.VideoRecorder`, `mjk.VideoResolution` | the C++ Filament recorder: `VideoRecorder.open(env, out_path, width, height, fps)`, `open_preset(env, out_path, resolution, fps)`, `open_offscreen(env, width, height)`; `record_frame()`, `render_rgb()` (an `(h, w, 3)` `uint8` array), `use_camera()`, `set_free_camera()`, `close()`; ffmpeg on `PATH` for MP4 |
| Primitive `rgba` required; `friction` required when `fixed` | optional: unset takes MuJoCo's default |
