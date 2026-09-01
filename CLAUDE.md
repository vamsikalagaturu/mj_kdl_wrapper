# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A C++ library bridging **MuJoCo 3.9** physics simulation with **KDL** (Kinematics and Dynamics Library) for robot kinematics/dynamics. The primary target is the Kinova GEN3 7-DOF arm with optional Robotiq 2F-85 gripper support. The same API is exposed to Python via pybind11 bindings (returning PyKDL types).

## Build

Requires: MuJoCo 3.9.0, downloaded into the user cache `~/.cache/mj_kdl_wrapper/mujoco-3.9.0` by default (`MJ_KDL_FETCH_MUJOCO=ON`; override with `-DMJ_KDL_MUJOCO_DIR=...` to use an existing install, no system paths are searched), apt packages `libglfw3-dev libgl-dev`, and the secorolab Orocos KDL fork. CMake fetches and builds the KDL fork by default; system `liborocos-kdl` is not used. Older MuJoCo releases are not supported; CMake validates the expected `mjVERSION_HEADER` from `cmake/Versions.cmake`.

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
./build/test_init

# Run with GUI (headless -> interactive)
./build/test_mjcf_trq_ctrl --gui
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
- `python/mj_kdl_wrapper/examples/ex_*.py` -- Python counterparts of the C++ `src/examples/ex_*.cpp`. They run headless by default and accept `--gui`.

**Packaging (examples + assets ship in the wheel):** the examples live inside the package, so `tool.scikit-build.wheel.packages` maps only `mj_kdl_wrapper` -> `python/mj_kdl_wrapper`. The repo-root `assets/` is installed into the wheel by CMake (`install(DIRECTORY assets/ DESTINATION mj_kdl_wrapper/assets)`), never mapped: a wheel mapping to a directory outside the package makes the editable install add every parent it needs to reach it, and for `assets/` that reached the workspace `src/`, putting every sibling repository on `sys.path`. Two console scripts populate a user's working directory:

- `mj-kdl-fetch-menagerie` (`menagerie:main`) -- clones the MuJoCo Menagerie into cache and copies bundled assets to `~/.cache/mj_kdl_wrapper/assets`.
- `mj-kdl-fetch-examples` (`fetch_examples:main`) -- copies the bundled `examples/` and `assets/` out as sibling dirs (default `./mj_kdl_wrapper_examples`).

**Asset resolution in examples:** Python and C++ helpers resolve Menagerie models and bundled assets from env overrides or the user cache. `mj-kdl-fetch-menagerie` populates both `menagerie/` and `assets/` under the cache.

## Formatting and Linting

```bash
# Format a file (pre-commit hook does this automatically)
clang-format --style=file -i src/mj_kdl_wrapper.cpp

# Run clang-tidy
clang-tidy -p build src/mj_kdl_wrapper.cpp
```

Column limit is 100. Indentation is 2 spaces. See `.clang-format` and `.clang-tidy` for full configuration.

## Architecture

**Single header, single implementation:**
- `include/mj_kdl_wrapper/mj_kdl_wrapper.hpp` -- all public types and function declarations
- `src/mj_kdl_wrapper.cpp` -- all implementation (~1400 lines)

**Key types (all in the `mj_kdl` namespace):**

- `RobotSpec` -- MJCF path, position, orientation, prefix, optional `AttachmentSpec` chain
- `AttachmentSpec` -- attachment MJCF, attach body, position/orientation, prefix, contact exclusions
- `SceneSpec` -- aggregates robots, objects (`SceneObject`, including MJCF-backed assets), timestep, gravity
- `Robot` -- runtime handle: holds `mjModel*`, `mjData*`, KDL chain, joint index maps, measured/commanded joint ports, `CtrlMode`
- `Viewer` -- GLFW window + MuJoCo render context

**Typical usage flow:**

```
MJCF files
    |
    v
build_scene()
    |
    v
mjModel*, mjData*  (compiled MuJoCo simulation)
    |
    v
init_robot_from_mjcf()
    |
    v
Robot  (KDL chain + joint index maps into MuJoCo arrays)
    |
    +-- update()           -- read sensors, apply commands (call every control step)
    +-- step() / step_n()  -- advance simulation
    +-- init_window_sim() + tick()  -- interactive event loop
```

**Control cycle (`update()`):** reads `qpos`/`qvel`/`qfrc_actuator` from MuJoCo into `jnt_pos_msr` / `jnt_vel_msr` / `jnt_trq_msr`, then writes commands back according to `ctrl_mode`: POSITION writes `jnt_pos_cmd` to `data->ctrl`; TORQUE writes `jnt_trq_cmd` to `data->qfrc_applied` (and neutralises position actuators by zeroing their ctrl error).

**Index maps inside `Robot`:** `kdl_to_mj_qpos`, `kdl_to_mj_dof`, `kdl_to_mj_ctrl` translate between KDL joint ordering and MuJoCo array indices. These are built during `init_robot_from_mjcf()` and are the reason multi-robot and gripper scenes work correctly even when joint ordering differs.

**Scene patching:** `build_scene()` merges MJCF files using `mjSpec` (MuJoCo's programmatic spec API), then calls `patch_mjcf_*` helpers to inject floor, skybox, table, and objects. Runtime add/remove (`scene_add_object` / `scene_remove_object`) re-compiles the spec in place and updates all existing `Robot` handles.

**Bundled dependencies:**
- user cache `~/.cache/mj_kdl_wrapper/menagerie/` -- MuJoCo Menagerie fetched by `mj-kdl-fetch-menagerie` or CMake

## Branching and releases

Development happens on `dev` (or feature branches off it). **`main` is protected** -- direct pushes are blocked (this applies to admins too), force-push and deletion are disabled, and merging requires a PR with all CI checks passing (`build`, `test`, `docs`, `bindings (3.10/3.11/3.12)`, `colcon (jazzy/lyrical)`; `deploy-docs` is deliberately not required since it only runs on releases). Do not attempt to commit or push directly to `main`.

To land work: branch off `dev`, open a PR into `dev`; squash is fine there. To cut a release, follow the ordered checklist in [AGENTS.md](AGENTS.md) -- the release PR is merged with a merge commit, which is what lets `dev` fast-forward to `main` afterwards instead of diverging from it.

**Versioning.** The version lives in two manual places that must stay in sync and read the same numeric string: `cmake/Versions.cmake` (`MJ_KDL_VERSION`) and `pyproject.toml` (`version`). `dev` always carries the *next* version, never the last released tag's number -- e.g. after releasing `0.1.0`, bump both files on `dev` to `0.1.1`. When cutting that release the files already read `0.1.1`, so just merge `dev` -> `main` and tag `v0.1.1`; then bump `dev` to the following version. The C++ build exposes the version via the `MJ_KDL_WRAPPER_VERSION` compile define (CMakeLists.txt), surfaced in Python as `mj_kdl_wrapper.__version__`.

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
