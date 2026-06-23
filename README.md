# Mujoco KDL Wrapper

[![build](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/build.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/build.yml)
[![tests](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/tests.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/tests.yml)
[![docs](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/docs.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/docs.yml)
[![python](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/python.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/python.yml)
[![ros2](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/ros2.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/ros2.yml)

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

Ubuntu/Debian. Requires MuJoCo 3.9.0 (auto-downloaded), CMake >= 3.16, a C++20
compiler, and (for Python) Python >= 3.10. The secorolab Orocos KDL fork is built
from source; the system KDL is never used.

Full details and every flag are in the installation guides:

- [Standalone Installation Guide](docs/install/standalone.md) -- C++ and Python, no ROS
- [ROS 2 Installation Guide](docs/install/ros2.md) -- colcon, C++ and Python

### System packages

```bash
sudo apt install cmake g++ git python3-dev python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev ffmpeg
```

### C++ (CMake)

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
# pin a release with --branch v0.1.0 (see Releases for the latest tag)
cd mj_kdl_wrapper

# configure (downloads MuJoCo, clones and builds the KDL fork, fetches Menagerie models)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMJ_KDL_FETCH_MENAGERIE=ON

# compile
cmake --build build --parallel $(nproc)

# optional install; self-contained, bundles KDL into the prefix
cmake --install build

# check: run the pick-and-place example
./build/src/examples/ex_pick
```

To share one KDL across several projects, set up a `ws/` folder, build the fork
once into `ws/install`, and point the wrapper (and any sibling) at it instead of
bundling its own:

```bash
# Workspace folder with the two source trees
mkdir -p ws && cd ws
git clone -b feature/achd_fixed_joint \
  https://github.com/secorolab/orocos_kinematics_dynamics.git
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git

# 1. Build the KDL fork once into ws/install
cmake -S orocos_kinematics_dynamics/orocos_kdl -B build/orocos_kdl \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" -DENABLE_TESTS=OFF
cmake --build build/orocos_kdl --parallel $(nproc)
cmake --install build/orocos_kdl

# 2. Build the wrapper against that shared KDL (bundles none of its own)
cmake -S mj_kdl_wrapper -B build/mj_kdl_wrapper \
  -DCMAKE_INSTALL_PREFIX="$PWD/install" -DMJ_KDL_OROCOS_KDL_INSTALL_DIR="$PWD/install"
cmake --build build/mj_kdl_wrapper --parallel $(nproc)
cmake --install build/mj_kdl_wrapper
```

The [standalone guide](docs/install/standalone.md) covers tests, custom
MuJoCo/KDL, sharing one KDL across projects, and all CMake options.

### Python

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"
# pin a release by appending @v0.1.0 to the URL (see Releases for the latest tag)

# fetch the MuJoCo Menagerie models and bundled assets into the user cache
mj-kdl-fetch-menagerie
```

Installing without a `@tag` tracks the `main` branch, which only advances at
releases - so the default command above already installs the latest release.

Bundles MuJoCo, the KDL fork, and PyKDL. See the
[standalone guide](docs/install/standalone.md#python) for editable installs,
Menagerie models, and build options.

The example scripts ship in the wheel. Copy them out, populate the cache, and
run one (the scripts resolve models and bundled assets from the cache, so they
run from anywhere):

```bash
# copy the bundled example scripts into ./mj_kdl_wrapper_examples
mj-kdl-fetch-examples
cd mj_kdl_wrapper_examples

# populate the user cache with Menagerie models and bundled assets
mj-kdl-fetch-menagerie

# check: run the pick-and-place example
python examples/ex_pick.py --gui
```

### ROS 2 (colcon)

Tested on ROS 2 **Jazzy** and **Lyrical**. Build the secorolab KDL as its own
workspace package, then the wrapper against it so the overlay shares one
`liborocos-kdl`:

```bash
# Workspace with the KDL fork and the wrapper as sibling packages
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
git clone -b feature/achd_fixed_joint \
  https://github.com/secorolab/orocos_kinematics_dynamics.git src/orocos_kinematics_dynamics
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git src/mj_kdl_wrapper
# pin a release with --branch v0.1.0 (see Releases for the latest tag)

# Use your distro: jazzy or lyrical
source /opt/ros/jazzy/setup.bash

# 1. Build the shared KDL package
colcon build --packages-select orocos_kdl --cmake-args -DENABLE_TESTS=OFF
source install/setup.bash

# 2. Build the wrapper against it (add -DMJ_KDL_FETCH_MENAGERIE=ON to fetch
# the Menagerie models needed by the examples and tests)
colcon build --packages-select mj_kdl_wrapper --cmake-args -DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON
```

The [ROS 2 guide](docs/install/ros2.md) covers the Python venv, the shared-KDL
rationale, build ordering, and consuming it from your own nodes.

## API

- [C++ API guide](docs/api/cpp.md)
- [Python Bindings API Guide](docs/api/python.md)
- Generated C++ and Python API reference: `build/docs/html/index.html` (build with
  `cmake -B build -DBUILD_DOCS=ON && cmake --build build --target docs`)

## Examples

The example catalog lives in [docs/examples.md](docs/examples.md).
Every C++ `src/examples/ex_*.cpp` example has a same-name Python counterpart in
`python/examples/`.

## Tests

```bash
ctest --test-dir build --output-on-failure
```

Tests need the Menagerie models (`-DMJ_KDL_FETCH_MENAGERIE=ON` at configure) and
self-skip without them. See [test/README.md](test/README.md) for the full list.

## More Documentation

- [C++ tutorial](docs/tutorials/cpp.md)
- [Python tutorial](docs/tutorials/python.md)
- [Torque control notes](docs/howto/torque_control.md)
- [URDF to MJCF notes](docs/howto/urdf_to_mjcf.md)
- [Examples guide](docs/examples.md)

## Assets

| Path | Description |
|------|-------------|
| `assets/robotiq_2f85/2f85.xml` | Local Robotiq 2F-85 gripper asset used by examples/tests |
| `assets/table.xml` | Table asset with authored `table_top` site |
| `assets/mug.xml`, `assets/mug_table.xml` | Pouring example assets |
