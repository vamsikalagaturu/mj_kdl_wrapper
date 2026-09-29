# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A C++ library bridging **MuJoCo 3.14** physics simulation with **KDL** (Kinematics and Dynamics Library) for robot kinematics/dynamics. The primary target is the Kinova GEN3 7-DOF arm with optional Robotiq 2F-85 gripper support. The same API is exposed to Python via pybind11 bindings (returning PyKDL types).

## Build

Requires: MuJoCo 3.14.0, downloaded into the user cache `~/.cache/mj_kdl_wrapper/mujoco-3.14.0` by default (`MJ_KDL_FETCH_MUJOCO=ON`; override with `-DMJ_KDL_MUJOCO_DIR=...` to use an existing install, no system paths are searched), apt packages `libeigen3-dev libglfw3-dev libgl-dev libegl-dev ffmpeg` (EGL and ffmpeg only for the recorder), and the secorolab Orocos KDL fork. CMake fetches and builds the KDL fork at the commit pinned in `cmake/Versions.cmake` (`MJ_KDL_OROCOS_KDL_GIT_SHA`); system `liborocos-kdl` is not used. The MuJoCo tarball (sha256) and the Menagerie commit are pinned there too; `scripts/check_versions.py` checks every copy of the pins. Older MuJoCo releases are not supported; CMake validates the expected `mjVERSION_HEADER` from `cmake/Versions.cmake`.

**Always build with all flags and verify tests pass before considering any task complete:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DMJ_KDL_FETCH_MENAGERIE=ON -DBUILD_DOCS=ON
cmake --build build --parallel $(nproc)
cmake --build build --target docs
ctest --test-dir build --output-on-failure
```

## Tests

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run a single test binary
./build/test/test_init

# Run the opt-in tests that open a Simulate window (gtest DISABLED_ prefix)
./build/test/test_scene_state --gtest_also_run_disabled_tests --gtest_filter='*Viewer*'
```

All tests self-skip when Menagerie is absent; `-DMJ_KDL_FETCH_MENAGERIE=ON` populates the user cache.

## Python bindings

The Python package lives under `python/` and is built with scikit-build-core (config in `pyproject.toml`, which sets `-DBUILD_PYTHON_BINDINGS=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF`).

```bash
# editable/dev install (builds the C++ extension)
uv pip install .

# run the Python test suite
pytest -q python/tests
```

- `python/mj_kdl_wrapper/` -- the bindings package (`menagerie.py`, `fetch_examples.py`, type stubs).
- `python/mj_kdl_wrapper/examples/ex_*.py` -- Python counterparts of the C++ `src/examples/ex_*.cpp`, one each. They run headless by default and accept `--gui`. Every example ends by itself: `--gui` runs the headless sequence with the viewer open.

**Packaging (examples + assets ship in the wheel):** the examples live inside the package, so `tool.scikit-build.wheel.packages` maps only `mj_kdl_wrapper` -> `python/mj_kdl_wrapper`. The repo-root `assets/` is installed into the wheel by CMake (`install(DIRECTORY assets/ DESTINATION mj_kdl_wrapper/assets)`), never mapped: a wheel mapping to a directory outside the package makes the editable install add every parent it needs to reach it, and for `assets/` that reached the workspace `src/`, putting every sibling repository on `sys.path`. Two console scripts populate a user's working directory:

- `mj-kdl-fetch-menagerie` (`menagerie:main`) -- clones the MuJoCo Menagerie into cache and copies bundled assets to `~/.cache/mj_kdl_wrapper/assets`.
- `mj-kdl-fetch-examples` (`fetch_examples:main`) -- copies the bundled `examples/` out (default `./mj_kdl_wrapper_examples/examples`); the examples read assets from the user cache.

**Asset resolution in examples:** Python and C++ helpers resolve Menagerie models and bundled assets from env overrides or the user cache. `mj-kdl-fetch-menagerie` populates both `menagerie/` and `assets/` under the cache.

## Formatting and Linting

```bash
# Format a file (pre-commit hook does this automatically)
clang-format --style=file -i src/mj_kdl_wrapper.cpp

# Run clang-tidy
clang-tidy -p build src/mj_kdl_wrapper.cpp
```

Column limit is 100. Indentation is 4 spaces (continuations 2). See `.clang-format` and `.clang-tidy` for full configuration.

## Architecture

**Single header, single implementation:**
- `include/mj_kdl_wrapper/mj_kdl_wrapper.hpp` -- all public types and function declarations
- `src/mj_kdl_wrapper.cpp` -- all implementation (~3200 lines)
- `src/simulate_ui/` -- the MuJoCo Simulate UI fork behind `open_viewer()`

**Key types (all in the `mj_kdl` namespace):**

- `Status` -- returned by every call that can fail on its input; `if (!s)` then `s.error` says why
- `RobotSpec` -- MJCF path, position, orientation (`quat`, `[x, y, z, w]`), prefix, control modes (`CtrlModeSpec`), optional `AttachmentSpec` chain; string fields are `std::string`, empty = not set
- `AttachmentSpec` -- attachment MJCF, attach body, position/orientation, prefix, contact exclusions
- `SceneSpec` -- aggregates robots, objects (`SceneObject`, including MJCF-backed assets), timestep, gravity
- `Env` -- owns `mjModel*`/`mjData*`, the registered `Robot`s, the scene slots (`env.scene`), the viewer (`env.viewer`) and the reset hook; not copied or moved
- `Robot` -- derives from `RobotPorts` (measured/commanded joint ports, `CtrlMode`); holds the KDL chain, joint names/limits, F/T sensors; its MuJoCo index maps are private (`_impl`)
- `SceneState` (`env.scene`) -- joints, free bodies, wrenches and actuators outside any `Robot`, bound with `bind_scene_*()`
- `Viewer` -- the Simulate UI of an `Env`, opened by `open_viewer()`
- `VideoRecorder` -- headless EGL rendering, to an MP4 through ffmpeg (`record_frame()`) or to a buffer (`render_rgb()`)

Conventions (units, frames, xyzw quaternions, what persists): `docs/conventions.md`.

**Typical usage flow:**

```
MJCF files
    |
    v
init_env()              -- build_scene() into an Env
    |
    v
init_robot_from_mjcf(&robot, &env, ...)   -- KDL chain; registers the robot
    |
    +-- open_viewer(&env)   -- optional Simulate UI
    +-- loop: step(&env); update(&env); compute commands into the ports
    +-- reset(&env)         -- resets MuJoCo, on_reset, re-seeds every robot and slot
    +-- cleanup(&env)
```

**Control cycle (`update(&env)`):** for every registered robot, reads `qpos`/`qvel`/`qfrc_actuator` into `jnt_pos_msr` / `jnt_vel_msr` / `jnt_trq_msr` and the F/T wrenches, then writes the active mode's command (`jnt_pos_cmd` / `jnt_vel_cmd` / `jnt_trq_cmd`) to that mode's actuators' `data->ctrl`; then samples and applies the scene slots. Each mode is an actuator group toggled via `opt.disableactuator` (see `docs/howto/torque_control.md`). `step(&env)` is `mj_step2` then `mj_step1`, so frames and sensors after it describe the new state.

**Reset by construction:** each part's runtime state is one struct (`RobotPorts`, `ForceTorqueReading`, `Scene*Reading` / `Scene*Command`) that `reset()` assigns afresh; parts go through `reset_parts()`, which requires a `reset_part()` overload per part at compile time. New runtime state belongs in one of those structs.

**Scene patching:** `build_scene()` merges MJCF files using `mjSpec` (MuJoCo's programmatic spec API), then injects floor, skybox, objects, sites and cameras. Runtime add/remove (`scene_add_object` / `scene_remove_object`) rebuilds the `Env`'s model and re-resolves its robots, scene slots and viewer.

**Bundled dependencies:**
- user cache `~/.cache/mj_kdl_wrapper/menagerie/` -- MuJoCo Menagerie at the pinned commit, fetched by `mj-kdl-fetch-menagerie` or CMake
- `assets/` -- bundled models (Gen3 with armature, 2F-85 whose ctrl is the driver angle 0..0.82 rad, table, cabinet, mug, cube, door latch, F/T sensor); a 2F-85 tool frame starts at `g_base_mount`, the body that carries the mount's mass

## Branching and releases

Development happens on `dev` (or feature branches off it). **`main` is protected** -- direct pushes are blocked (this applies to admins too), force-push and deletion are disabled, and merging requires a PR with all CI checks passing (`build`, `test`, `docs`, `bindings (3.10/3.11/3.12)`, `colcon (jazzy/lyrical)`; `deploy-docs` is deliberately not required since it only runs on releases). Do not attempt to commit or push directly to `main`.

To land work: branch off `dev`, open a PR into `dev`; squash is fine there. To cut a release, follow the ordered checklist in [AGENTS.md](AGENTS.md) -- the release PR is merged with a merge commit, which is what lets `dev` fast-forward to `main` afterwards instead of diverging from it.

**Versioning.** The version lives in three manual places that must stay in sync and read the same numeric string: `cmake/Versions.cmake` (`MJ_KDL_VERSION`), `pyproject.toml` (`version`) and `package.xml` (`<version>`); `scripts/check_versions.py` checks that they agree. `dev` always carries the *next* version, never the last released tag's number -- e.g. after releasing `0.1.0`, bump all three files on `dev` to `0.1.1`. When cutting that release the files already read `0.1.1`, so just merge `dev` -> `main` and tag `v0.1.1`; then bump `dev` to the following version. `project(VERSION)` takes it from `cmake/Versions.cmake`, so the CMake package config and `mj_kdl_wrapper.pc` carry it; only the Python extension gets it as a compile define (`MJ_KDL_WRAPPER_VERSION`, surfaced as `mj_kdl_wrapper.__version__`). The C++ library has no version define or call. The CMake package matches only within a minor version (`SameMinorVersion`): 0.x minors are breaking.

**Docs/GitHub Pages deploy only on releases.** `docs.yml` builds docs on every push/PR (CI check) but only uploads the Pages artifact and deploys when `github.event_name == 'release'`. The `github-pages` environment allows deployments from the `main` branch and from `v*` tags. Publishing a release is what refreshes <https://mj-kdl-wrapper.vamsi.sh/>.

## Code Style

### Comments

- Use `/** ... */` (JavaDoc) for Doxygen documentation comments on public API declarations (structs, enums, functions) in the header.
- Use `//` for all single-line comments (standalone or inline/trailing).
- Use `/* ... */` only for multi-line block comments in the implementation.
- Never use border lines (`//---`, `// ===`, `// ***`, etc.) to delimit sections.
- Use only ASCII characters in comments and string literals.
  No Unicode arrows (→ ← ↔), dashes (— –), ellipses (…), or other non-ASCII symbols.
  Use ASCII equivalents: `->`, `<-`, `<->`, `-`, `...`
