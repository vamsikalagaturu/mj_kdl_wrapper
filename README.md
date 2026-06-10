# Mujoco KDL Wrapper

A C++ library bridging [MuJoCo](https://github.com/google-deepmind/mujoco) physics simulation with [KDL](https://github.com/orocos/orocos_kinematics_dynamics) for robot kinematics and dynamics.

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

- **Scene builder** -- compose MJCF robots, grippers, objects, and cameras into one
  MuJoCo scene via `mjSpec`, with ordered attachment chains and relative placement.
- **Multi-robot** -- multiple robots with independent KDL chains in one simulation.
- **KDL from the model** -- builds the KDL chain directly from the compiled MuJoCo model.
- **Control** -- POSITION / TORQUE ports plus KDL FK, IK, RNEA, and ACHD solvers.
- **Runtime environments** -- `Env` with reset hooks for task setup and replay.
- **Interactive viewer** -- MuJoCo simulate UI with Frames / Trace / Perturb panels
  and overlay lines.
- **Recording** -- interactive and headless EGL + ffmpeg MP4 capture.
- **Python bindings** -- the same API from Python via `pip install`, returning PyKDL types.

## Install

Ubuntu/Debian instructions. Supported dependency versions are listed below.
CMake checks `mjVERSION_HEADER` and stops if `MJ_KDL_MUJOCO_DIR` points at an
unsupported MuJoCo release.

Pick the workflow that matches how you consume the library. The standalone
workflows need no ROS; the ROS 2 section is entirely separate.

Standalone (no ROS):

| Use case | Build system | Section |
|----------|--------------|---------|
| C++ | `cmake` / `find_package` | [C++ (CMake)](#c-cmake) |
| Python | `pip install` (single command) | [Python](#python) |

ROS 2:

| Use case | Build system | Section |
|----------|--------------|---------|
| ROS 2 C++ | `colcon` (no separate package) | [ROS 2 C++](#ros-2-c) |
| ROS 2 Python | `colcon` + venv `--system-site-packages` | [ROS 2 Python](#ros-2-python) |

### Dependency Versions

| Dependency | Version / source | Notes |
|------------|------------------|-------|
| MuJoCo | `3.9.0` from `cmake/Versions.cmake` | Native library and pinned `mujoco` Python package must match |
| Orocos KDL | secorolab fork, `feature/achd_fixed_joint` | Built from source; system `liborocos-kdl` is not used |
| CMake | `>=3.16` | Required to configure the C++ build |
| C++ compiler | C++20-capable | `CMAKE_CXX_STANDARD` is set to 20 |
| Python | `>=3.10` | Required for the Python package |
| scikit-build-core | `>=0.11.2` | Python build backend |
| pybind11 | `>=2.13` | Python binding build dependency |

### System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev \
  ffmpeg doxygen
```

### C++ (CMake)

A standard CMake project. The default build is self-contained: it downloads
MuJoCo, clones and builds the Orocos KDL fork, and (with the menagerie flag)
fetches the robot models - no system MuJoCo or KDL is used.

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMJ_KDL_FETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

#### Install

```bash
cmake --install build           # add -DCMAKE_INSTALL_PREFIX="$HOME/.local" to configure first
```

The install is self-contained: the Orocos KDL fork (library, headers, and its
CMake package config) is installed into the same prefix and the wrapper is
rpath'd to `$ORIGIN`, so other projects in the prefix can `find_package(orocos_kdl)`
directly and `find_package(mj_kdl_wrapper)` pulls KDL in transitively. No build
tree or `LD_LIBRARY_PATH` is needed at runtime. (MuJoCo is not bundled; consumers
resolve it from `MJ_KDL_MUJOCO_DIR` / the pip package.)

#### Where the dependencies come from

Each dependency is fetched by default but can point at something you already have.
See the full list in [CMake Options](#cmake-options); the common ones:

| To... | Set |
|-------|-----|
| Use an existing MuJoCo install | `-DMJ_KDL_MUJOCO_DIR=/opt/mujoco-3.9.0` |
| Clone the KDL fork somewhere specific | `-DMJ_KDL_OROCOS_KDL_DIR=~/src/orocos_kinematics_dynamics` |
| Reuse a prebuilt KDL by prefix (no clone/rebuild/bundle) | `-DMJ_KDL_OROCOS_KDL_INSTALL_DIR=$HOME/.local` |
| Consume KDL via its CMake package on `CMAKE_PREFIX_PATH` | `-DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON` |
| Keep bundled KDL out of the install prefix | `-DMJ_KDL_INSTALL_BUNDLED_KDL=OFF` |
| Choose build / install locations | `cmake -B <build-dir> -DCMAKE_INSTALL_PREFIX=<prefix>` |

#### One shared KDL across several projects

When multiple projects need KDL, build the fork once into a shared prefix and
point everyone at it, so there is exactly one `liborocos-kdl` (avoids duplicate
copies and rebuilds):

```bash
# 1. Build the fork once into the shared prefix
cmake -S <kdl-src>/orocos_kdl -B build/orocos_kdl \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build/orocos_kdl --parallel $(nproc) && cmake --install build/orocos_kdl

# 2. Build the wrapper (and any sibling) against that shared KDL
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  -DMJ_KDL_OROCOS_KDL_INSTALL_DIR=$HOME/.local
cmake --build build --parallel $(nproc) && cmake --install build
```

By default the wrapper bundles its own KDL, which is correct for a single
project; switch to the shared prefix only when several projects link KDL (see
[ROS 2](#ros-2-jazzy--colcon) for the colcon version of this).

### Python

Install into an active supported Python environment. The build bundles the
native dependencies (MuJoCo, the secorolab Orocos KDL fork, and PyKDL); see the
[Python Bindings API Guide](docs/api/python.md) for what it does and how model
paths are resolved.

PyKDL is bundled inside the `mj-kdl-wrapper` wheel as a top-level extension
module. It imports as `PyKDL`, but it does not appear as a separate package in
`pip list` / `uv pip list`.

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"  # from GitHub
uv pip install .                                                            # from a checkout
```

This is an isolated, self-contained build independent of any C++ build tree: it
recompiles the wrapper, builds and bundles its own KDL + PyKDL, and pins
`mujoco==3.9.0`. Build options (all optional):

| To... | Add |
|-------|-----|
| Editable install (dev) | `uv pip install -e . --config-settings=editable.rebuild=true` |
| Put the build dir outside the source tree | `--config-settings=build-dir=/path/build_py/{wheel_tag}` |
| Reuse a prebuilt KDL+PyKDL prefix (skip bundling) | `--config-settings=cmake.define.MJ_KDL_OROCOS_KDL_INSTALL_DIR=/prefix` |

The shared-KDL option needs a prefix that also ships `PyKDL`; a C++-only install
prefix has KDL but no `PyKDL`, so the default (bundle) is right for standalone use.

Fetch the MuJoCo Menagerie models the examples use (requires `git`), verify, and
run an example from a checkout:

```bash
mj-kdl-fetch-menagerie
python -c "import PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mujoco.mj_versionString(), mjk.mujoco_version())"
python python/examples/ex_gravity_comp.py
```

Use `--dest` to choose the Menagerie checkout location:

```bash
mj-kdl-fetch-menagerie --dest /path/to/menagerie
export MJ_KDL_MENAGERIE=/path/to/menagerie
```

The `python/examples/` scripts are not shipped in the wheel, so run them from a
checkout. Model resolution, environment variables, and using other model sources
are documented in the [Python Bindings API Guide](docs/api/python.md).

### ROS 2 (Jazzy / colcon)

A plain CMake project (not `ament_cmake`); colcon builds it via the bundled
`package.xml` (build type `cmake`). It is not registered in the ament index, so
`ros2 pkg list` will not show it - expected; consumers still use
`find_package(mj_kdl_wrapper)`.

Build the secorolab KDL as its own workspace package so the overlay has one shared
`liborocos-kdl` (the distro KDL and `python3-pykdl` use the same SONAME, and two
copies in one process is unsafe). Workspace layout:

```
ros2_ws/src/
  orocos_kinematics_dynamics/   # secorolab fork -> orocos_kdl package
  mj_kdl_wrapper/
```

Install the [system packages](#system-packages) in the environment first.

#### ROS 2 C++

Build the KDL fork first, then the wrapper against it:

```bash
cd ~/ros2_ws && source /opt/ros/jazzy/setup.bash
colcon build --packages-select orocos_kdl --cmake-args -DENABLE_TESTS=OFF
source install/setup.bash
colcon build --packages-select mj_kdl_wrapper \
  --cmake-args -DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
```

The wrapper links the shared `orocos_kdl` and bundles no KDL of its own; other
packages consume it with `find_package(orocos_kdl)` / `find_package(mj_kdl_wrapper)`.

#### ROS 2 Python

Create the venv with `--system-site-packages` so it can import the system `rclpy`,
then install the wheel:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages ~/ros2_ws/.venv-ros
source ~/ros2_ws/.venv-ros/bin/activate
pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"
python -c "import rclpy, PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mjk.mujoco_version())"
```

The wheel's bundled `PyKDL` takes precedence over the system `python3-pykdl` on
`sys.path`.

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

Paths / sources (override to use your own):

| Flag | Default | Description |
|------|---------|-------------|
| `MJ_KDL_MUJOCO_DIR` | `/opt/mujoco-${MJ_KDL_MUJOCO_VERSION}` | Existing MuJoCo install to use |
| `MJ_KDL_FETCH_MUJOCO` | `ON` | Download MuJoCo when `MJ_KDL_MUJOCO_DIR` is not present |
| `MJ_KDL_MUJOCO_URL` | (release) | MuJoCo archive URL to download |
| `MJ_KDL_FETCH_OROCOS_KDL` | `ON` | Clone and build the secorolab Orocos KDL fork (the only KDL used) |
| `MJ_KDL_OROCOS_KDL_GIT_REPOSITORY` | secorolab fork | Orocos KDL git source to build |
| `MJ_KDL_OROCOS_KDL_GIT_TAG` | `feature/achd_fixed_joint` | Orocos KDL branch/tag to build |
| `MJ_KDL_OROCOS_KDL_DIR` | `third_party/orocos_kinematics_dynamics` | Fork source/clone destination; built in place if already present, else cloned here when fetch is ON. Point elsewhere (e.g. `~/test/src`) to clone/build there |
| `MJ_KDL_OROCOS_KDL_INSTALL_DIR` | (empty) | Pre-installed Orocos KDL prefix to consume (skips building and bundling the fork; for ROS 2 / a single shared workspace KDL) |
| `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` | `OFF` | Consume Orocos KDL via `find_package(orocos_kdl)` on `CMAKE_PREFIX_PATH` (colcon overlay or any prefix); skips building and bundling the fork |
| `MJ_KDL_FETCH_MENAGERIE` | `OFF` | Download MuJoCo Menagerie models |
| `MJ_KDL_MENAGERIE_DIR` | `third_party/menagerie` | Menagerie location / `MJ_KDL_FETCH_MENAGERIE` destination |

Build toggles:

| Flag | Default | Description |
|------|---------|-------------|
| `BUILD_RECORDER` | `ON` | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `BUILD_EXAMPLES` | `ON` | Build the `src/examples/ex_*` programs |
| `BUILD_TESTS` | `ON` | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS` | `OFF` | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `MJ_KDL_INSTALL_BUNDLED_KDL` | `ON` | Install the built Orocos KDL fork (lib, headers, CMake config) into the prefix so the install is self-contained and sibling packages can `find_package(orocos_kdl)`. No effect with `MJ_KDL_OROCOS_KDL_INSTALL_DIR` |
| `SHOW_EQUALITY_PANEL` | `OFF` | Show the Simulate UI `Equality` section |
| `SHOW_GROUP_PANEL` | `OFF` | Show the Simulate UI `Group enable` section |

> [!NOTE]
> Once the build succeeds, follow the [C++ tutorial](docs/tutorials/cpp.md) or
> [Python tutorial](docs/tutorials/python.md) to start building scenes, adding
> robots, KDL control, reset hooks, and more.

## API

- [C++ API guide](docs/api/cpp.md)
- [Python Bindings API Guide](docs/api/python.md)
- Generated C++ and Python API reference: `build/docs/html/index.html`

## Examples

The example catalog lives in [docs/examples.md](docs/examples.md).
Every C++ `src/examples/ex_*.cpp` example has a same-name Python counterpart in
`python/examples/`.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

See [test/README.md](test/README.md) for the full list of tests.

## More Documentation

- [C++ tutorial](docs/tutorials/cpp.md)
- [Python tutorial](docs/tutorials/python.md)
- [Torque control notes](docs/howto/torque_control.md)
- [URDF to MJCF notes](docs/howto/urdf_to_mjcf.md)
- [Examples guide](docs/examples.md)

## Assets

| Path | Description |
|------|-------------|
| `third_party/menagerie/kinova_gen3/gen3.xml` | Kinova GEN3 7-DOF arm (MuJoCo Menagerie) |
| `assets/robotiq_2f85/2f85.xml` | Local Robotiq 2F-85 gripper asset used by examples/tests |
| `assets/table.xml` | Table asset with authored `table_top` site |
| `assets/mug.xml`, `assets/mug_table.xml` | Pouring example assets |
