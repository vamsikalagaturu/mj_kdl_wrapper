# mj_kdl_wrapper

[![build](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/build.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/build.yml)
[![tests](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/tests.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/tests.yml)
[![docs](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/docs.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/docs.yml)
[![python](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/python.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/python.yml)
[![ros2](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/ros2.yml/badge.svg)](https://github.com/vamsikalagaturu/mj_kdl_wrapper/actions/workflows/ros2.yml)

[MuJoCo](https://github.com/google-deepmind/mujoco) physics with
[Orocos KDL](https://github.com/orocos/orocos_kinematics_dynamics) kinematics and dynamics,
in C++ and Python.

<table>
<tr>
  <td align="center"><img src="docs/screenshots/ex_gravity_comp.png" width="380"/><br/><b>ex_gravity_comp</b> &mdash; Single arm, KDL gravity compensation</td>
</tr>
</table>

## Features

- **Scene builder** -- compose MJCF robots, grippers, objects and cameras into one MuJoCo
  scene via `mjSpec`, with ordered attachment chains and relative placement.
- **KDL from the model** -- the KDL chain is built from the compiled MuJoCo model; several
  robots, each with its own chain, share one simulation.
- **Control** -- POSITION / VELOCITY / TORQUE ports, each mode a group of MuJoCo actuators
  ([torque control](docs/howto/torque_control.md)), plus KDL FK, IK, RNEA and ACHD solvers.
- **One `Env`** -- owns the model, robots, scene slots and overlay geoms; `step` / `update` /
  `reset` drive it, and reset re-seeds everything it holds.
- **Viewer and recording** -- optional libraries: `mj_kdl_wrapper::viewer`, a MuJoCo Studio
  window, and `mj_kdl_wrapper::render`, headless MP4 capture with MuJoCo's Filament renderer
  and ffmpeg.
- **Python bindings** -- the same `Env` and `Robot`, returning PyKDL types and numpy arrays;
  `mj_kdl_wrapper.viewer.Viewer` shows an `Env` through `mujoco.studio`.

## Install

Ubuntu/Debian, CMake >= 3.16, a C++20 compiler and, for Python, Python >= 3.11. MuJoCo 4.0.0
and the secorolab Orocos KDL fork are built from source (`mj_kdl_wrapper.repos`); the system
KDL is never used.

```bash
sudo apt install cmake g++ git python3-dev python3-venv vcstool libeigen3-dev ffmpeg \
  libegl-dev libgl1-mesa-dev libwayland-dev libxkbcommon-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev

git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git && cd mj_kdl_wrapper
vcs import < mj_kdl_wrapper.repos            # the KDL fork and MuJoCo, into third_party/
```

The second line of packages is for the render and viewer libraries (`libegl-dev` for render,
the X11, Wayland and GL headers for MuJoCo's Filament and SDL2 builds); without them CMake
builds the core only.

`main` only advances at releases, so a plain clone is the latest release; `--branch vX.Y.Z`
pins an older one. `vcstool` is also on PyPI; ROS apt sources name it `python3-vcstool`.

**C++**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
cmake --install build               # optional: find_package(mj_kdl_wrapper COMPONENTS render viewer)
./build/src/examples/ex_table_pick_place
```

`pkg-config mj_kdl_wrapper` covers the core library only. The examples link the viewer and
are built only with it.

**Python**

```bash
uv pip install .
python python/mj_kdl_wrapper/examples/ex_table_pick_place.py --gui
```

The wheel builds the core only and bundles the KDL fork, PyKDL and the `assets/` models;
MuJoCo, its plugins, viewer and renderers come from the pinned `mujoco` pip package.

**ROS 2 (colcon)**, Jazzy or Lyrical, with KDL as its own package so the overlay shares one
`liborocos-kdl`:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git src/mj_kdl_wrapper
vcs import src < src/mj_kdl_wrapper/mj_kdl_wrapper.repos
source /opt/ros/jazzy/setup.bash
colcon build --packages-select orocos_kdl --cmake-args -DENABLE_TESTS=OFF
source install/setup.bash
colcon build --packages-select mj_kdl_wrapper --cmake-args -DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON \
  -DMJ_KDL_MUJOCO_DIR="${PWD}/src/third_party/mujoco"
```

The [standalone](docs/install/standalone.md) and [ROS 2](docs/install/ros2.md) guides cover
tests, editable Python installs, a custom MuJoCo or KDL, sharing one KDL across projects and
every CMake option.

## Assets

The bundled models ship in the repo and the wheel. C++ examples and tests load them from the
source `assets/` (`mj_kdl_examples::asset(...)`), Python from the installed package
(`mj_kdl_wrapper.ASSETS_DIR / "..."`), and `cmake --install` copies them into `~/.cache/mj_kdl_wrapper/assets`
for programs that know neither.

| Path | Description |
|------|-------------|
| `assets/kinova_gen3/gen3.xml` | Kinova Gen3; MuJoCo Menagerie's model plus a base_link/shoulder_link contact exclusion and Kinova's joint armature |
| `assets/robotiq_2f85/2f85.xml` | Robotiq 2F-85; ctrl is the driver joint angle, 0 (open) to 0.82 rad (closed). Use `tool_body = "g_base_mount"` (with prefix `g_`): the mount carries mass too |
| `assets/ft_sensor.xml` | 6-axis force-torque sensor |
| `assets/table.xml` | Table with an authored `table_top` site |
| `assets/mug.xml`, `assets/mug_table.xml` | Pouring example |
| `assets/cabinet/cabinet.xml` | Three-drawer cabinet |
| `assets/cube.xml` | Free cube |
| `assets/door_latch/door_latch.xml` | Latched cupboard door (not used by the examples or tests) |

Any other MJCF, e.g. from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
loads through `RobotSpec.path`; a model that brings its own floor needs `use_stage = false`.

## Examples and tests

Each example exists in C++ (`src/examples/`) and Python (`python/mj_kdl_wrapper/examples/`) and ends by
itself; the C++ ones open the viewer unless given `--headless`, the Python ones run headless
unless given `--gui`. The catalog is in [docs/examples.md](docs/examples.md).

```bash
ctest --test-dir build --output-on-failure      # see test/README.md
```

## Documentation

- [C++ tutorial](docs/tutorials/cpp.md) and [C++ API guide](docs/api/cpp.md)
- [Python tutorial](docs/tutorials/python.md) and [Python API guide](docs/api/python.md)
- [Conventions](docs/conventions.md) -- units, frames, quaternion order, F/T sign, what persists
- [Torque control](docs/howto/torque_control.md), [loop pacing](docs/howto/loop_pacing.md)
- Migration notes between versions: [docs/index.md](docs/index.md)
- Generated API reference: `cmake -B build -DBUILD_DOCS=ON && cmake --build build --target docs`,
  then `build/docs/html/index.html`
