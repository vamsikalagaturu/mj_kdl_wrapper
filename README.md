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
cd mj_kdl_wrapper
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
cmake --install build          # optional; self-contained, bundles KDL into the prefix
```

The [standalone guide](docs/install/standalone.md) covers tests, custom
MuJoCo/KDL, sharing one KDL across projects, and all CMake options.

### Python

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"
```

Bundles MuJoCo, the KDL fork, and PyKDL. See the
[standalone guide](docs/install/standalone.md#python) for editable installs,
Menagerie models, and build options.

### ROS 2 (colcon)

Build the secorolab KDL as its own workspace package, then the wrapper against it
so the overlay shares one `liborocos-kdl`:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
git clone -b feature/achd_fixed_joint \
  https://github.com/secorolab/orocos_kinematics_dynamics.git src/orocos_kinematics_dynamics
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git src/mj_kdl_wrapper

source /opt/ros/jazzy/setup.bash
colcon build --packages-select orocos_kdl --cmake-args -DENABLE_TESTS=OFF
source install/setup.bash
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
