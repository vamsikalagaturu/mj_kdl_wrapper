# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A C++ library bridging **MuJoCo 3.8** physics simulation with **KDL** (Kinematics and Dynamics Library) for robot kinematics/dynamics. The primary target is the Kinova GEN3 7-DOF arm with optional Robotiq 2F-85 gripper support.

## Build

Requires: MuJoCo 3.8.0 at `/opt/mujoco-3.8.0` (override with `-DMUJOCO_ROOT=...`), and apt packages `liborocos-kdl-dev libglfw3-dev libgl-dev`. Older MuJoCo releases are not supported; CMake validates `mjVERSION_HEADER == 3008000`.

**Always build with all flags and verify tests pass before considering any task complete:**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DFETCH_MENAGERIE=ON -DBUILD_DOCS=ON
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

All tests self-skip when `third_party/menagerie` is absent (requires `-DFETCH_MENAGERIE=ON`).

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
- `SceneSpec` -- aggregates robots, optional table (`TableSpec`), objects (`SceneObject`), timestep, gravity
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

**Control cycle (`update()`):** reads `qpos`/`qvel`/`qfrc_actuator` from MuJoCo into `jnt_pos_msr` / `jnt_vel_msr` / `jnt_trq_msr`, then writes `jnt_pos_cmd` / `jnt_vel_cmd` / `jnt_trq_cmd` back to MuJoCo actuators according to `ctrl_mode` (POSITION, VELOCITY, or TORQUE).

**Index maps inside `Robot`:** `kdl_to_mj_qpos`, `kdl_to_mj_dof`, `kdl_to_mj_ctrl` translate between KDL joint ordering and MuJoCo array indices. These are built during `init_robot_from_mjcf()` and are the reason multi-robot and gripper scenes work correctly even when joint ordering differs.

**Scene patching:** `build_scene()` merges MJCF files using `mjSpec` (MuJoCo's programmatic spec API), then calls `patch_mjcf_*` helpers to inject floor, skybox, table, and objects. Runtime add/remove (`scene_add_object` / `scene_remove_object`) re-compiles the spec in place and updates all existing `Robot` handles.

**Bundled dependencies:**
- `third_party/menagerie/` -- MuJoCo Menagerie (optional, fetched via CMake FetchContent)

## Code Style

### Comments

- Use `/** ... */` (JavaDoc) for Doxygen documentation comments on public API declarations (structs, enums, functions) in the header.
- Use `//` for all single-line comments (standalone or inline/trailing).
- Use `/* ... */` only for multi-line block comments in the implementation.
- Never use border lines (`//---`, `// ===`, `// ***`, etc.) to delimit sections.
- Use only ASCII characters in comments and string literals.
  No Unicode arrows (→ ← ↔), dashes (— –), ellipses (…), or other non-ASCII symbols.
  Use ASCII equivalents: `->`, `<-`, `<->`, `-`, `...`
