# ROS 2 Installation Guide

Full reference for building `mj_kdl_wrapper` in a ROS 2 (Jazzy) colcon workspace,
for both C++ and Python. For non-ROS builds, see the
[Standalone Installation Guide](standalone.md).

## Contents

- [How colcon builds this package](#how-colcon-builds-this-package)
- [KDL must be shared](#kdl-must-be-shared)
- [System packages](#system-packages)
- [ROS 2 C++](#ros-2-c)
  - [Consuming it from your own package](#consuming-it-from-your-own-package)
  - [Build ordering](#build-ordering)
- [ROS 2 Python](#ros-2-python)
- [Sharing one KDL: options](#sharing-one-kdl-options)

## How colcon builds this package

This is a plain CMake project, not `ament_cmake`, and it ships no `package.xml`.
colcon still builds it: the `colcon-cmake` extension (part of
`python3-colcon-common-extensions`) discovers any directory with a
`CMakeLists.txt` and builds it as a `cmake`-type package named after the
`project()` call (`mj_kdl_wrapper`).

It is not registered in the ament index, so `ros2 pkg list` / `ros2 pkg prefix`
will not show it - expected, and it does not affect linking: dependent packages
consume it with `find_package(mj_kdl_wrapper)`, and a downstream
`<depend>mj_kdl_wrapper</depend>` still orders the build correctly (colcon matches
the discovered package name). Without a `package.xml`, `rosdep` cannot resolve this
package's system dependencies, so install the [system packages](#system-packages)
yourself.

## KDL must be shared

This project requires the secorolab Orocos KDL fork (not the system KDL). In a
standalone build the wrapper builds and bundles its own private copy. In ROS 2
that is unsafe: ROS ships the distro Orocos KDL and `python3-pykdl` with the
**same SONAME (`liborocos-kdl.so.1.5`) and `PyKDL` module name**, so a single
process can hold only one copy - the loader keeps the first and silently drops the
other's symbols.

The fix is to build the secorolab fork as its own workspace package and have every
package (including `mj_kdl_wrapper`) consume that one shared `liborocos-kdl`. The
fork ships `package.xml` for `orocos_kdl` (C++, build type `cmake`) and
`python_orocos_kdl`, so colcon can build it directly.

## System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev ffmpeg
```

ROS 2 Jazzy and `colcon` (via `python3-colcon-common-extensions`) are assumed
present and sourced.

## ROS 2 C++

Create the workspace, clone the KDL fork and the wrapper, then build KDL first and
the wrapper against it:

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws
git clone -b feature/achd_fixed_joint \
  https://github.com/secorolab/orocos_kinematics_dynamics.git src/orocos_kinematics_dynamics
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git src/mj_kdl_wrapper

source /opt/ros/jazzy/setup.bash

# 1. Build the fork's C++ KDL as the shared package
colcon build --packages-select orocos_kdl --cmake-args -DENABLE_TESTS=OFF
source install/setup.bash

# 2. Build the wrapper against it (no clone, no rebuild, no bundling)
colcon build --packages-select mj_kdl_wrapper \
  --cmake-args -DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
source install/setup.bash
```

Result: `mj_kdl_wrapper` links `install/orocos_kdl/lib/liborocos-kdl.so` and ships
no KDL of its own; there is exactly one KDL in the overlay.

The wrapper checkout carries a `third_party/COLCON_IGNORE`, so the fork it may
clone into `third_party/` for standalone builds is never picked up as a duplicate
workspace package.

### Consuming it from your own package

```cmake
find_package(mj_kdl_wrapper REQUIRED)
target_link_libraries(my_node mj_kdl_wrapper::mj_kdl_wrapper)
# KDL is available too:
find_package(orocos_kdl REQUIRED)
target_link_libraries(my_node orocos-kdl)
```

### Build ordering

Step 1 must precede step 2 so the `orocos_kdl` package is on `CMAKE_PREFIX_PATH`
when the wrapper configures. Run the two `colcon build` steps in order as shown.
For automatic single-command ordering, declare the dependencies in a consuming
package's own `package.xml`:

```xml
<depend>orocos_kdl</depend>
<depend>mj_kdl_wrapper</depend>
```

(`mj_kdl_wrapper` itself ships no `package.xml`; colcon discovers it from
`CMakeLists.txt` and matches the name for ordering.)

## ROS 2 Python

`rclpy` is only importable from the interpreter ROS 2 was built against (the
system Python). Create the venv with `--system-site-packages` so it can import the
system `rclpy` while keeping the wheel and its pinned `mujoco==3.9.0` inside the
venv:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages ~/ros2_ws/.venv-ros
source ~/ros2_ws/.venv-ros/bin/activate
pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"
python -c "import rclpy, PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mjk.mujoco_version())"
```

The wheel bundles its own `PyKDL`, which takes precedence over the system
`python3-pykdl` on the venv `sys.path`. To confirm which is loaded:

```bash
python -c "import PyKDL; print(PyKDL.__file__)"   # -> venv site-packages, not /usr/lib/python3/dist-packages
```

The fork's `python_orocos_kdl` is a catkin package and is not part of the C++
overlay above; in ROS 2 the Python side uses the bundled wheel PyKDL (or the
system `python3-pykdl`).

### Using C++ and Python together

Source ROS 2 and the workspace first, then activate the venv so it inherits the
ROS 2 environment:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/ros2_ws/.venv-ros/bin/activate
```

## Sharing one KDL: options

| Approach | How | When |
|----------|-----|------|
| `find_package` (recommended) | `-DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON` | KDL is a colcon package on `CMAKE_PREFIX_PATH` (the overlay) |
| By prefix | `-DMJ_KDL_OROCOS_KDL_INSTALL_DIR=<prefix>` | KDL is installed at a known prefix |
| Process isolation | (no build change) | Keep `mj_kdl_wrapper` and system-KDL packages (`kdl_parser`, `tf2_kdl`, `robot_state_publisher`) in separate nodes; ROS 2 is multi-process, so each loads one KDL |
| Bundled (default) | (no build change) | Single-package use; not safe to mix with system-KDL packages in one process |

For the Python wheel, pass the shared-KDL define through scikit-build-core
(`--config-settings=cmake.define.MJ_KDL_OROCOS_KDL_INSTALL_DIR=<prefix>`); note the
prefix must also ship `PyKDL`.
