# Mujoco KDL Wrapper

A C++ library bridging [MuJoCo 3.9](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

## Screenshots

<table>
<tr>
  <td align="center"><img src="docs/screenshots/ex_gravity_comp.png" width="380"/><br/><b>ex_gravity_comp</b> &mdash; Single arm, KDL gravity compensation</td>
  <td align="center"><img src="docs/screenshots/ex_table_scene.png" width="380"/><br/><b>ex_table_scene</b> &mdash; Arm + table + scene objects</td>
</tr>
<tr>
  <td align="center"><img src="docs/screenshots/ex_pick.png" width="380"/><br/><b>ex_pick</b> &mdash; Pick-and-place with Robotiq 2F-85</td>
  <td align="center"><img src="docs/screenshots/ex_dual_arm.png" width="380"/><br/><b>ex_dual_arm</b> &mdash; Dual arm + grippers</td>
</tr>
</table>

## Features

- **Unified scene builder** -- `build_scene()` accepts MJCF files and builds floor, skybox, primitive objects, asset-backed objects, and cameras via `mjSpec` with no intermediate XML files
- **Ordered attachment chains** -- `AttachmentSpec` attaches any MJCF body (mount, FT sensor, gripper, arm on a mobile base) under any named body, site, or frame via a tagged `AttachTarget`; chains of arbitrary length are applied in declaration order
- **Relative placement everywhere** -- `RobotSpec.attach_to` and `SceneObject.attach_to` accept the same tagged `AttachTarget`, so a robot can be mounted on a site exported by a prior scene object (e.g. a tabletop site) without hand-threading world-frame heights
- **Multi-robot scenes** -- place multiple robots with independent KDL chains in one shared simulation via `SceneSpec::robots`
- **Runtime environments** -- `Env` owns model/data, registered robots, and reset hooks for task-specific object/controller state
- **KDL chain from model** -- `init_robot_from_mjcf()` builds a KDL chain directly from a compiled MuJoCo model
- **Control ports** -- `update()` reads `qpos`/`qvel`/`qfrc_actuator` into `*_msr` and applies `*_cmd` in POSITION or TORQUE mode
- **Interactive viewer** -- `init_window_sim()` + `step()` gives your code the control loop while the MuJoCo simulate UI runs in a background render thread
- **Viewer debug panels** -- `Frames` (per-body/site coordinate triads), `Trace` (end-effector trail), and `Perturb` (point-and-drag force/torque on a selected body) sections in the simulate UI
- **Overlay geometry** -- `clear_trace()` / `add_trace_segment()` draw your own lines (e.g. a live end-effector trajectory trace) into the simulate UI's user scene; no-ops in headless mode
- **Interactive and headless recording** -- Simulate UI recorder controls plus `VideoRecorder` for EGL offscreen MP4 recording

## Install

Ubuntu/Debian instructions. Only MuJoCo 3.9.0 is supported (CMake checks
`mjVERSION_HEADER` and stops if `MUJOCO_ROOT` points at another release).

### System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev \
  ffmpeg doxygen
```

### MuJoCo 3.9.0

Install it once and point `MUJOCO_ROOT` at it, or pass `-DMJ_KDL_FETCH_MUJOCO=ON`
to let CMake download it.

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/3.9.0/mujoco-3.9.0-linux-x86_64.tar.gz
sudo tar -xzf mujoco-3.9.0-linux-x86_64.tar.gz -C /opt/
export MUJOCO_ROOT=/opt/mujoco-3.9.0
```

### C++ library, examples, and tests

```bash
git clone https://github.com/secorolab/mj_kdl_wrapper.git
cd mj_kdl_wrapper
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

- The build always downloads and builds the pinned secorolab Orocos KDL fork
  (`feature/achd_fixed_joint`, the only KDL this project uses; no system
  `liborocos-kdl` is consulted). Override the source with
  `-DMJ_KDL_OROCOS_KDL_GIT_REPOSITORY=...` / `-DMJ_KDL_OROCOS_KDL_GIT_TAG=...`.
- `-DFETCH_MENAGERIE=ON` downloads the Kinova GEN3 / Robotiq assets the examples
  and tests use into `third_party/menagerie/`.
- Add `-DMJ_KDL_FETCH_MUJOCO=ON` if you did not set `MUJOCO_ROOT`.

Install it for use from another CMake project via `find_package(mj_kdl_wrapper)`
(add `-DCMAKE_INSTALL_PREFIX="$HOME/.local"` at configure time for a user-local
install):

```bash
cmake --install build
```

### Python Install

Install into an active Python 3.10+ environment. The build bundles the native
dependencies (MuJoCo, the secorolab Orocos KDL fork, and PyKDL); see the
[Python bindings guide](docs/PYTHON_BINDINGS.md) for what it does and how model
paths are resolved.

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"  # from GitHub
uv pip install .                                                            # from a checkout
```

Fetch the MuJoCo Menagerie models the examples use (requires `git`), verify, and
run an example from a checkout:

```bash
mj-kdl-fetch-menagerie
python -c "import PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mujoco.mj_versionString(), mjk.mujoco_version())"
python python/examples/ex_gravity_comp.py
```

The `python/examples/` scripts are not shipped in the wheel, so run them from a
checkout. Model resolution, environment variables, and using other model sources
are documented in the [Python bindings guide](docs/PYTHON_BINDINGS.md).

### Generate Documentation

```bash
cmake -B build -DBUILD_DOCS=ON
cmake --build build --target docs
```

Open `build/docs/html/index.html`. The generated docs include the C++ headers,
C++ examples, Markdown guides, Python stubs, and Python examples. The docs
target also generates `build/docs/kdl.tag` and `build/docs/html/kdl/` from the
installed Orocos KDL headers so KDL types and common solver calls link locally.

### CMake Options

| Flag | Default | Description |
|------|---------|-------------|
| `MJ_KDL_FETCH_MUJOCO=ON` | OFF | Download the supported MuJoCo release if `MUJOCO_ROOT` is not set |
| `FETCH_MENAGERIE=ON` | OFF | Download MuJoCo Menagerie robot models into `third_party/menagerie/` |
| `BUILD_RECORDER=ON` | ON | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `BUILD_EXAMPLES=ON` | ON | Build the `src/examples/ex_*` programs |
| `BUILD_TESTS=ON` | ON | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS=ON` | OFF | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `SHOW_EQUALITY_PANEL=ON` | OFF | Show the Simulate UI `Equality` section (hidden by default) |
| `SHOW_GROUP_PANEL=ON` | OFF | Show the Simulate UI `Group enable` section (hidden by default) |

> [!NOTE]
> Once the build succeeds, follow the [Tutorial](docs/TUTORIAL.md) to start
> building scenes, adding robots, KDL control, reset hooks, and more.

## API

- [C++ API guide](docs/CPP_API.md)
- [Python bindings guide](docs/PYTHON_BINDINGS.md)
- Generated C++ and Python API reference: `build/docs/html/index.html`

## Examples

The example catalog lives in [src/examples/README.md](src/examples/README.md).
Every C++ `src/examples/ex_*.cpp` example has a same-name Python counterpart in
`python/examples/`.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

See [test/README.md](test/README.md) for the full list of tests.

## More Documentation

- [Full tutorial](docs/TUTORIAL.md)
- [Torque control notes](docs/HOWTO_torque_control.md)
- [URDF to MJCF notes](docs/HOWTO_urdf_to_mjcf.md)
- [Examples guide](src/examples/README.md)

## Assets

| Path | Description |
|------|-------------|
| `third_party/menagerie/kinova_gen3/gen3.xml` | Kinova GEN3 7-DOF arm (MuJoCo Menagerie) |
| `third_party/menagerie/robotiq_2f85/2f85.xml` | Robotiq 2F-85 gripper (MuJoCo Menagerie) |
