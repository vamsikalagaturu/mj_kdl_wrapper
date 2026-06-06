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
- **Dynamics probes** -- `test/urdf_solver_probe.cpp` uses the bundled Kinova GEN3 URDF to check ACHD fixed-joint outputs and compare URDF-vs-MuJoCo RNEA torques
- **Interactive viewer** -- `init_window_sim()` + `step()` gives your code the control loop while the MuJoCo simulate UI runs in a background render thread
- **Viewer debug panels** -- `Frames` (per-body/site coordinate triads), `Trace` (end-effector trail), and `Perturb` (point-and-drag force/torque on a selected body) sections in the simulate UI
- **Overlay geometry** -- `clear_trace()` / `add_trace_segment()` draw your own lines (e.g. a live end-effector trajectory trace) into the simulate UI's user scene; no-ops in headless mode
- **Interactive and headless recording** -- Simulate UI recorder controls plus `VideoRecorder` for EGL offscreen MP4 recording

## Setup

The commands below assume Ubuntu or Debian-style packages and install MuJoCo
3.9.0 at `/opt/mujoco-3.9.0`. Set `MUJOCO_ROOT` if you install it elsewhere.

Only MuJoCo 3.9.0 is supported. CMake checks `mjVERSION_HEADER` and stops at
configure time if `MUJOCO_ROOT` points to any other MuJoCo release.

### System Packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev \
  ffmpeg doxygen
```

`libeigen3-dev` is required to build the secorolab Orocos KDL fork from source
for the Python wheel.

The Python install builds Orocos KDL and PyKDL from
`https://github.com/secorolab/orocos_kinematics_dynamics.git` at
`feature/achd_fixed_joint`, so distro `liborocos-kdl-dev` and `python3-pykdl`
are not required for the wheel. The secorolab fork is the only KDL the wheel
uses; no system KDL is consulted.

### MuJoCo 3.9.0

```bash
wget https://github.com/google-deepmind/mujoco/releases/download/3.9.0/mujoco-3.9.0-linux-x86_64.tar.gz
sudo tar -xzf mujoco-3.9.0-linux-x86_64.tar.gz -C /opt/
export MUJOCO_ROOT=/opt/mujoco-3.9.0
```

Persist `MUJOCO_ROOT` in your shell profile if `/opt/mujoco-3.9.0` is not the
default path you want CMake to use.

### Clone The Repo

```bash
git clone https://github.com/secorolab/mj_kdl_wrapper.git
cd mj_kdl_wrapper
```

### Orocos KDL From Source (Required)

The C++ build requires the secorolab Orocos KDL fork at
`feature/achd_fixed_joint`; the distro `liborocos-kdl-dev` does not provide the
ACHD fixed-joint solver this project depends on. Build and install the fork,
then point CMake at it.

```bash
git clone --branch feature/achd_fixed_joint \
  https://github.com/secorolab/orocos_kinematics_dynamics.git
cd orocos_kinematics_dynamics
cmake orocos_kdl -B build_kdl \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=~/ws/install \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF
cmake --build build_kdl -j$(nproc)
cmake --install build_kdl
```

Then add `-DCMAKE_PREFIX_PATH=~/ws/install` to the CMake configure command.

The Python install handles this automatically and does not need this step; see
[Python Install](#python-install).

## Build And Install

### Build C++ Library, Examples, Tests, And Docs

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCH_MENAGERIE=ON \
  -DBUILD_DOCS=ON
cmake --build build --parallel $(nproc)
```

`FETCH_MENAGERIE=ON` downloads the Kinova GEN3 and Robotiq assets used by the
examples and tests into `third_party/menagerie/`.

Run the C++ smoke tests:

```bash
ctest --test-dir build --output-on-failure
./build/src/examples/ex_gravity_comp --headless
```

### Install The C++ Package

Install the C++ package if another CMake project should use
`find_package(mj_kdl_wrapper)`:

```bash
sudo cmake --install build
```

For a user-local install, configure with an explicit prefix first:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DFETCH_MENAGERIE=ON \
  -DBUILD_DOCS=ON \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build --parallel $(nproc)
cmake --install build
```

### Python Install

Once the system packages above are installed, install directly from GitHub into
your active Python 3.10+ environment:

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"
```

The Python build downloads the matching native MuJoCo release automatically if
`MUJOCO_ROOT` does not already point to a MuJoCo 3.9.0 install. It also builds
Orocos KDL and PyKDL from the pinned `feature/achd_fixed_joint` branch above,
bundles their shared libraries into the wheel, and installs the matching
official `mujoco` Python package from the MuJoCo version in
`cmake/MuJoCoVersion.cmake`. It does not build the C++ examples/tests; use the
CMake build above for those.

For a local checkout, use:

```bash
uv pip install .
```

Verify the environment:

```bash
python -c "import PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mujoco.mj_versionString(), mjk.mujoco_version())"
python python/examples/basic_scene.py
```

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
| `BUILD_RECORDER=ON` | ON | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `FETCH_MENAGERIE=ON` | OFF | Download MuJoCo Menagerie robot models |
| `BUILD_TESTS=ON` | ON | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS=ON` | OFF | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `SHOW_EQUALITY_PANEL=ON` | OFF | Show the Simulate UI `Equality` section (hidden by default) |
| `SHOW_GROUP_PANEL=ON` | OFF | Show the Simulate UI `Group enable` section (hidden by default) |

The repo also carries `third_party/kinova/GEN3_URDF_V12.urdf` for KDL parser
diagnostics. The MuJoCo model remains sourced from Menagerie.

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
