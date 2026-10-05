# CLAUDE.md

## What this is

- C++ library bridging MuJoCo 3.14 physics with Orocos KDL kinematics and dynamics; Python bindings via pybind11 return PyKDL types.
- Primary target: Kinova GEN3 7-DOF arm, optionally with a Robotiq 2F-85 gripper.
- A plain C/C++ CMake package; the only ROS part is the optional `camera_ros` target (`MJKDL_WITH_ROS`, default `AUTO`).

## Build

- MuJoCo 3.14.0 is downloaded into `~/.cache/mjkdl/mujoco-3.14.0` (`MJKDL_FETCH_MUJOCO=ON`); `-DMJKDL_MUJOCO_DIR=...` uses an existing install; no system paths are searched.
- apt: `libeigen3-dev libglfw3-dev libgl-dev libegl-dev ffmpeg` (EGL and ffmpeg only for the recorder).
- CMake builds the secorolab Orocos KDL fork from `third_party/orocos_kinematics_dynamics` (`MJKDL_OROCOS_KDL_DIR`); it never clones it; system `liborocos-kdl` is never used.
- `vcs import < mjkdl.repos` (vcstool, pip or apt) checks the fork out there, tracking branch `vereshchagin-driver-weighting`.
- `cmake/Versions.cmake` pins the MuJoCo tarball (sha256) and googletest; CMake validates `mjVERSION_HEADER`. KDL comes from `mjkdl.repos`.
- Always build with all flags and pass the tests before calling a task done:
  ```bash
  vcs import < mjkdl.repos
  cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DBUILD_DOCS=ON
  cmake --build build --parallel $(nproc)
  cmake --build build --target docs
  ctest --test-dir build --output-on-failure
  ```

## Tests

- All: `ctest --test-dir build --output-on-failure`.
- One binary: `./build/test/test_init`.
- Viewer tests (gtest `DISABLED_`): `./build/test/test_scene_state --gtest_also_run_disabled_tests --gtest_filter='*Viewer*'`.
- Tests load the bundled models from `assets/` (`MJKDL_ASSETS_DIR`) and self-skip only when one is missing there.

## Python bindings

- Package under `python/`, built by scikit-build-core (`pyproject.toml` sets `-DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF`).
- Install: `vcs import < mjkdl.repos` then `uv pip install .`; tests: `pytest -q python/tests`.
- `python/mjkdl/examples/ex_*.py` mirror `src/examples/ex_*.cpp`, run headless by default, take `--gui`, and end by themselves.
- The wheel maps only `mjkdl` -> `python/mjkdl`; CMake installs the repo-root `assets/` into it. Never map `assets/` in the wheel: the editable install would put the workspace `src/` on `sys.path`.
- Python examples and tests load models as `str(mjkdl.ASSETS_DIR / "kinova_gen3/gen3.xml")`; `ASSETS_DIR` is the packaged `assets/`.

## Formatting and linting

- `clang-format --style=file -i <file>` (the pre-commit hook runs it); `clang-tidy -p build <file>`.
- Column limit 100; indent 4, continuations 2; see `.clang-format` and `.clang-tidy`.

## Architecture

- One header `include/mjkdl/mjkdl.hpp`, one implementation `src/mjkdl.cpp`; `src/simulate_ui/` is the Simulate UI fork behind `open_viewer()`.
- Key types (namespace `mjkdl`):
  - `Status`: returned by every call that can fail on its input; `if (!s)` then `s.error`.
  - `RobotSpec`: MJCF path, position, `quat` `[x, y, z, w]`, prefix, `CtrlModeSpec`s, `AttachmentSpec` chain; empty strings mean unset.
  - `AttachmentSpec`: attachment MJCF, attach body, pose, prefix, contact exclusions.
  - `SceneSpec`: robots, objects (`SceneObject`, MJCF-backed included), timestep, gravity.
  - `Env`: owns `mjModel*`/`mjData*`, registered `Robot`s, `env.scene`, `env.viewer`, the reset hook; never copied or moved.
  - `Robot`: derives from `RobotPorts`; holds the KDL chain, joint names and limits, F/T sensors; MuJoCo index maps are private (`_impl`).
  - `SceneState` (`env.scene`): joints, free bodies, wrenches and actuators outside any `Robot`, bound with `bind_scene_*()`.
  - `Viewer`: the Simulate UI opened by `open_viewer()`.
  - `VideoRecorder`: headless EGL rendering to MP4 (`record_frame()`) or a buffer (`render_rgb()`).
- Conventions (units, frames, xyzw quaternions, F/T sign, what persists): `docs/conventions.md`.
- Usage flow: `init_env()` -> `init_robot_from_mjcf()` -> optional `open_viewer()` -> loop `step()`, `update()`, write ports -> `reset()` -> `cleanup()`.
- `update(&env)`: reads `qpos`/`qvel`/`qfrc_actuator` into the measured ports and the F/T wrenches, writes the active mode's command to its actuators' `ctrl`, then samples and applies scene slots.
- Each control mode is an actuator group toggled through `opt.disableactuator` (`docs/howto/torque_control.md`).
- `step(&env)` is `mj_step2` then `mj_step1`, so frames and sensors after it describe the new state.
- Reset by construction: each part's runtime state is one struct (`ForceTorqueReading`, `Scene*Reading`, `Scene*Command`) that `reset()` assigns afresh, via a `reset_part()` overload per part.
- Exception: `RobotPorts` is rewritten in place by `seed_ports()`, because callers hold pointers into it; a new port field needs a line there.
- New runtime state belongs in one of those structs.
- `build_scene()` merges MJCF through `mjSpec`, then injects floor, skybox, objects, sites and cameras.
- `scene_add_object` / `scene_remove_object` rebuild the model and re-resolve robots, scene slots and the viewer.
- Bundled `assets/` (the only models used): Gen3 with armature, 2F-85 (ctrl is the driver angle 0..0.82 rad; tool frame starts at `g_base_mount`), table, cabinet, mug, cube, door latch, F/T sensor.
- C++ examples and tests resolve them with `mjkdl_examples::asset()` / `find_asset()` (`src/examples/example_paths.hpp`) from the source `assets/`; nothing is fetched.
- `cmake --install` copies `assets/` and the two model licenses into `~/.cache/mjkdl/assets` (skipped for wheels); motion-spec's generated controllers read them there.

## Branching

- Work on `dev` or a feature branch off it; PRs into `dev` may squash.
- `main` is protected (admins included): no direct push, no force-push or deletion; merging needs a PR with `build`, `test`, `docs`, `bindings (3.10/3.11/3.12)` and `colcon (jazzy/lyrical)` passing. `deploy-docs` only runs on releases and is not required.

## Versioning and releases

- `MJKDL_VERSION` in `cmake/Versions.cmake` is the only version: `project(VERSION)`, the CMake package config, `mjkdl.pc` and `pyproject.toml` (scikit-build-core regex provider) read it.
- Only the Python extension gets it as a define (`MJKDL_VERSION`, exposed as `mjkdl.__version__`); the C++ library has no version define or call.
- The CMake package matches within a minor version (`SameMinorVersion`): 0.x minors are breaking.
- `dev` always carries the next version, never the last released one.
- Cutting `vX.Y.Z`:
  - check `MJKDL_VERSION` already reads `X.Y.Z`;
  - open a PR `dev` -> `main` with all required checks passing;
  - merge it with a merge commit (`gh pr merge --merge`), never squash, so `dev` can fast-forward afterwards;
  - on `main`, fast-forward, `git tag -a vX.Y.Z`, push the tag; never tag `dev`;
  - publish the GitHub release for the tag, which deploys the docs to <https://mj-kdl-wrapper.vamsi.sh/>;
  - fast-forward `dev` to `main`, bump `MJKDL_VERSION` to the next patch, commit on `dev`.
- If `dev` cannot fast-forward to `main`, something landed on `main` outside the release PR: find out what before resolving.

## Code style

- `/** ... */` for Doxygen comments on public declarations in the header.
- `//` for single-line comments, standalone or trailing.
- `/* ... */` only for multi-line block comments in the implementation.
- No border lines (`//---`, `// ===`, `// ***`).
- ASCII only in comments and string literals: `->`, `<-`, `<->`, `-`, `...` instead of arrows, dashes or ellipses.
