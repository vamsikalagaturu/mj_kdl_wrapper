# CLAUDE.md

## What this is

- C++ library bridging MuJoCo 4.0 physics with Orocos KDL kinematics and dynamics; Python bindings via pybind11 return PyKDL types.
- Primary target: Kinova GEN3 7-DOF arm, optionally with a Robotiq 2F-85 gripper.
- A plain C/C++ CMake package; the only ROS part is the optional `camera_ros` target (`MJ_KDL_WITH_ROS`, default `AUTO`).
- Three libraries: core `mj_kdl_wrapper` (MuJoCo + KDL, no GL or window), `mj_kdl_wrapper::render` (Filament offscreen recorder, `MJ_KDL_WITH_RENDER`) and `mj_kdl_wrapper::viewer` (Studio window, `MJ_KDL_WITH_VIEWER`); both options AUTO/ON/OFF, default AUTO; the viewer needs render.

## Build

- MuJoCo 4.0.0 is built from source in `third_party/mujoco` (`MJ_KDL_MUJOCO_DIR`) as an ExternalProject under `build/mujoco`: its CMake assumes it is the top-level project. Its static Filament and Studio archives are imported with the dependency list of upstream's `python/mujoco/CMakeLists.txt`.
- The release tarball is not used: it ships neither Filament nor the Studio libraries as libraries.
- apt: `libeigen3-dev ffmpeg`, plus `libegl-dev` for render and `libx11-dev` (and the X/Wayland headers SDL2 uses) and `libfontconfig-dev` for the viewer; ffmpeg only for recording.
- CMake builds the secorolab Orocos KDL fork from `third_party/orocos_kinematics_dynamics` (`MJ_KDL_OROCOS_KDL_DIR`); it never clones it; system `liborocos-kdl` is never used.
- `vcs import < mj_kdl_wrapper.repos` (vcstool, pip or apt) checks out the fork (branch `vereshchagin-driver-weighting`) and MuJoCo (tag `4.0.0`).
- `cmake/Versions.cmake` holds `MJ_KDL_MUJOCO_VERSION`, which must match the `.repos` tag (CMake validates `mjVERSION_HEADER`), and googletest.
- Filament and Studio's fonts are staged in `build/mujoco_assets` and installed to `<libdir>/mujoco_assets`, next to the render library that reads them (`register_asset_providers()` in `src/render.cpp`).
- Always build with all flags and pass the tests before calling a task done:
  ```bash
  vcs import < mj_kdl_wrapper.repos
  cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DBUILD_DOCS=ON
  cmake --build build --parallel $(nproc)
  cmake --build build --target docs
  ctest --test-dir build --output-on-failure
  ```

## Tests

- All: `ctest --test-dir build --output-on-failure`.
- One binary: `./build/test/test_init`.
- `test_render` and `test_viewer` exist only when their components are built; the C++ examples (and their `ex_*_headless` tests) need the viewer.
- Tests load the bundled models from `assets/` (`MJ_KDL_ASSETS_DIR`) and self-skip only when one is missing there.

## Python bindings

- Package under `python/`, built by scikit-build-core (`pyproject.toml` sets `-DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF`).
- A wheel build (`SKBUILD`) links the pip `mujoco-core` package's `libmujoco` (a build requirement) and builds render and viewer (`pyproject.toml` forces them ON) from MuJoCo's Filament and Studio archives, compiled from `third_party/mujoco`; the process loads one libmujoco.
- Python views and records through the same C++ code: `mjk.Viewer` (`open(env, title)`, then `env.step()` and `viewer.pace()`) and `mjk.VideoRecorder` bind `mj_kdl::Viewer` and `mj_kdl::VideoRecorder`. Never add a second, Python-side UI: C++ and Python must not differ.
- Install: `vcs import < mj_kdl_wrapper.repos` then `uv pip install .` into `.venv` (`uv venv .venv --python 3.12`); tests: `.venv/bin/pytest -q python/tests`. Python >= 3.11 (MuJoCo 4.0 dropped 3.10).
- `python/mj_kdl_wrapper/examples/ex_*.py` mirror `src/examples/ex_*.cpp`, run headless by default, take `--gui`, and end by themselves.
- The wheel maps only `mj_kdl_wrapper` -> `python/mj_kdl_wrapper`; CMake installs the repo-root `assets/` into it. Never map `assets/` in the wheel: the editable install would put the workspace `src/` on `sys.path`.
- Python examples and tests load models as `str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")`; `ASSETS_DIR` is the packaged `assets/`.

## Formatting and linting

- `clang-format --style=file -i <file>` (the pre-commit hook runs it); `clang-tidy -p build <file>`.
- Column limit 100; indent 4, continuations 2; see `.clang-format` and `.clang-tidy`.

## Architecture

- Core: `include/mj_kdl_wrapper/mj_kdl_wrapper.hpp`, `src/mj_kdl_wrapper.cpp`. Render: `render.hpp`, `src/render.cpp`. Viewer: `viewer.hpp`, `src/viewer.cpp`, built on Studio's components (`studio/hal`, `studio/render`, `studio/ux`), never on `LaunchStudio`/`App`.
- Key types (namespace `mj_kdl`):
  - `Status`: returned by every call that can fail on its input; `if (!s)` then `s.error`.
  - `RobotSpec`: MJCF path, position, `quat` `[x, y, z, w]`, prefix, `CtrlModeSpec`s, `AttachmentSpec` chain; empty strings mean unset.
  - `AttachmentSpec`: attachment MJCF, attach body, pose, prefix, contact exclusions.
  - `SceneSpec`: robots, objects (`SceneObject`, MJCF-backed included), timestep, gravity.
  - `Env`: owns `mjModel*`/`mjData*`, registered `Robot`s, `env.scene`, the reset hook, overlay geoms and `env.hooks` (`EnvHooks` of the components following it); never copied or moved.
  - `Robot`: derives from `RobotPorts`; holds the KDL chain, joint names and limits, F/T sensors; MuJoCo index maps are private (`_impl`).
  - `SceneState` (`env.scene`): joints, free bodies, wrenches and actuators outside any `Robot`, bound with `bind_scene_*()`.
  - `Viewer` (viewer): owned by the caller; `open_viewer(&viewer, &env)` runs a window thread that draws a copy of the state `step()` publishes; its edits reach the Env inside `step()` (`EnvHooks::on_step`). Keys are ImGuiKey codes.
  - `VideoRecorder` (render): Filament offscreen rendering to MP4 (`record_frame()`) or a buffer (`render_rgb()`), on a thread of its own, from a state snapshot.
- Conventions (units, frames, xyzw quaternions, F/T sign, what persists): `docs/conventions.md`.
- Usage flow: `init_env()` -> `init_robot_from_mjcf()` -> optional `open_viewer()` -> loop `update()`, write ports, `step()`, `pace_realtime()` -> `reset()` -> `cleanup()`.
- The core is single-threaded: only the thread that steps an Env touches it; the viewer and recorders work from copies (overlay calls are the thread-safe exception).
- `update(&env)`: reads `qpos`/`qvel`/`qfrc_actuator` into the measured ports and the F/T wrenches, writes the active mode's command to its actuators' `ctrl`, then samples and applies scene slots.
- Each control mode is an actuator group toggled through `opt.disableactuator` (`docs/howto/torque_control.md`).
- `step(&env)` is `mj_step2` then `mj_step1`, so frames and sensors after it describe the new state.
- Reset by construction: each part's runtime state is one struct (`ForceTorqueReading`, `Scene*Reading`, `Scene*Command`) that `reset()` assigns afresh, via a `reset_part()` overload per part.
- Exception: `RobotPorts` is rewritten in place by `seed_ports()`, because callers hold pointers into it; a new port field needs a line there.
- New runtime state belongs in one of those structs.
- `build_scene()` starts from the stage MJCF (`SceneSpec::stage`; the bundled `assets/stage/default.xml`, compiled into the core, when empty; an empty world when `use_stage` is false), merges MJCF through `mjSpec`, then injects objects, sites and cameras.
- `scene_add_object` / `scene_remove_object` rebuild the model and re-resolve robots and scene slots; the viewer and recorders follow through `EnvHooks::on_rebuild`.
- Bundled `assets/` (the only models used): Gen3 with armature, 2F-85 (ctrl is the driver angle 0..0.82 rad; tool frame starts at `g_base_mount`), table, cabinet, mug, cube, door latch, F/T sensor.
- C++ examples and tests resolve them with `mj_kdl_examples::asset()` / `find_asset()` (`src/examples/example_paths.hpp`) from the source `assets/`; nothing is fetched.
- `cmake --install` copies `assets/` and the two model licenses into `~/.cache/mj_kdl_wrapper/assets` (skipped for wheels); motion-spec's generated controllers read them there.

## Branching

- Work on `dev` or a feature branch off it; PRs into `dev` may squash.
- `main` is protected (admins included): no direct push, no force-push or deletion; merging needs a PR with `build`, `test`, `docs`, `bindings (3.11/3.12)` and `colcon (jazzy/lyrical)` passing (the 3.10 bindings check goes with MuJoCo 4.0; the protection rule must follow). `deploy-docs` only runs on releases and is not required.

## Versioning and releases

- `MJ_KDL_VERSION` in `cmake/Versions.cmake` is the only version: `project(VERSION)`, the CMake package config, `mj_kdl_wrapper.pc` and `pyproject.toml` (scikit-build-core regex provider) read it.
- Only the Python extension gets it as a define (`MJ_KDL_WRAPPER_VERSION`, exposed as `mj_kdl_wrapper.__version__`); the C++ library has no version define or call.
- The CMake package matches within a minor version (`SameMinorVersion`): 0.x minors are breaking.
- `dev` always carries the next version, never the last released one.
- Cutting `vX.Y.Z`:
  - check `MJ_KDL_VERSION` already reads `X.Y.Z`;
  - open a PR `dev` -> `main` with all required checks passing;
  - merge it with a merge commit (`gh pr merge --merge`), never squash, so `dev` can fast-forward afterwards;
  - on `main`, fast-forward, `git tag -a vX.Y.Z`, push the tag; never tag `dev`;
  - publish the GitHub release for the tag, which deploys the docs to <https://mj-kdl-wrapper.vamsi.sh/>;
  - fast-forward `dev` to `main`, bump `MJ_KDL_VERSION` to the next patch, commit on `dev`.
- If `dev` cannot fast-forward to `main`, something landed on `main` outside the release PR: find out what before resolving.

## Code style

- `/** ... */` for Doxygen comments on public declarations in the header.
- `//` for single-line comments, standalone or trailing.
- `/* ... */` only for multi-line block comments in the implementation.
- No border lines (`//---`, `// ===`, `// ***`).
- ASCII only in comments and string literals: `->`, `<-`, `<->`, `-`, `...` instead of arrows, dashes or ellipses.
