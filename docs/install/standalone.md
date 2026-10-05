# Standalone Installation Guide

Full reference for building and installing `mjkdl` without ROS, as a
plain CMake C++ library and/or a Python package. For ROS 2 / colcon, see the
[ROS 2 Installation Guide](ros2.md).

Instructions target Ubuntu/Debian. CMake checks `mjVERSION_HEADER` and stops if
`MJKDL_MUJOCO_DIR` points at an unsupported MuJoCo release.

## Contents

- [Dependency versions](#dependency-versions)
- [System packages](#system-packages)
- [C++ (CMake)](#c-cmake)
  - [Default build](#default-build)
  - [Install](#install)
  - [Where the dependencies come from](#where-the-dependencies-come-from)
  - [One shared KDL across several projects](#one-shared-kdl-across-several-projects)
- [Python](#python)
- [Generate documentation](#generate-documentation)
- [All CMake options](#all-cmake-options)

## Dependency versions

| Dependency | Version / source | Notes |
|------------|------------------|-------|
| MuJoCo | `3.14.0` from `cmake/Versions.cmake` | Native library and pinned `mujoco` Python package must match |
| Orocos KDL | secorolab fork, branch `vereshchagin-driver-weighting`, from `mjkdl.repos` | Built from source; system `liborocos-kdl` is not used |
| vcstool | any | Checks out the KDL fork (`vcs import < mjkdl.repos`) |
| CMake | `>=3.16` | Required to configure the C++ build |
| C++ compiler | C++20-capable | `CMAKE_CXX_STANDARD` is set to 20 |
| Python | `>=3.10` | Required for the Python package |
| scikit-build-core | `>=0.11.2` | Python build backend |
| pybind11 | `>=2.13` | Python binding build dependency |

## System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv vcstool \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev \
  ffmpeg doxygen
```

`doxygen` is only needed for the docs target; `python3-dev`/`python3-venv` only
for the Python package. `vcstool` is also on PyPI (`pip install vcstool`); ROS apt sources
name it `python3-vcstool`.

## C++ (CMake)

A standard CMake project. The default build is self-contained: it downloads
MuJoCo and builds the Orocos KDL fork that `vcs import` checked out; the robot models
ship in `assets/`. No system MuJoCo or KDL is ever used.

### Default build

```bash
git clone https://github.com/vamsikalagaturu/mjkdl.git
cd mjkdl

# check out the KDL fork into third_party/orocos_kinematics_dynamics
vcs import < mjkdl.repos

# configure (downloads MuJoCo, builds the KDL fork)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# compile
cmake --build build --parallel $(nproc)

# run the test suite
ctest --test-dir build --output-on-failure
```

What happens during configure/build:

- **MuJoCo** is downloaded into the user cache (`MJKDL_FETCH_MUJOCO=ON`) unless
  `MJKDL_MUJOCO_DIR` already holds a matching install. No system paths are
  searched; set `MJKDL_MUJOCO_DIR` to use an install elsewhere.
- **Orocos KDL** (the secorolab fork) is built from `MJKDL_OROCOS_KDL_DIR` (default
  `third_party/orocos_kinematics_dynamics`, where `vcs import` puts it) via
  `ExternalProject` into `build/orocos_kdl_install`. CMake never clones it; configure stops
  when the checkout is missing. `vcs pull third_party` updates it.
- **Robot models** (Gen3, Robotiq gripper, table, mug, ...) ship in `assets/`. The C++
  examples and tests load them from there via `example_paths.hpp`; tests self-skip when one
  is missing.

### Install

```bash
# choose the prefix at configure time
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/ws"
cmake --build build --parallel $(nproc)
cmake --install build
```

The install is self-contained. Alongside `libmjkdl.so`, its headers, and
the `mjkdl` CMake package config, the built Orocos KDL fork is installed
into the same prefix:

```
<prefix>/lib/libmjkdl.so
<prefix>/lib/liborocos-kdl.so*
<prefix>/include/mjkdl/...
<prefix>/include/kdl/...
<prefix>/lib/cmake/mjkdl/...
<prefix>/lib/pkgconfig/mjkdl.pc
<prefix>/lib/pkgconfig/orocos-kdl.pc
<prefix>/share/orocos_kdl/cmake/...
<prefix>/share/doc/mjkdl/...        # LICENSE, NOTICE and third-party licenses
```

The wrapper is rpath'd to `$ORIGIN` as a `DT_RPATH`, not a `DT_RUNPATH`, so it loads the
co-installed KDL without a build tree, and an `LD_LIBRARY_PATH` holding a system
`liborocos-kdl` cannot swap it out. Consumers:

```cmake
find_package(mjkdl REQUIRED)              # pulls KDL in transitively
target_link_libraries(my_app mjkdl::mjkdl)
# or use KDL directly:
find_package(orocos_kdl REQUIRED)
target_link_libraries(my_app orocos-kdl)
```

A requested version matches only within its minor version: `find_package(mjkdl 0.3)`
does not accept `0.4.x`. The exported target carries `-Wl,--disable-new-dtags`, so a consumer
executable gets a `DT_RPATH` too.

Without CMake, use pkg-config. `mjkdl.pc` requires `orocos-kdl`, which the bundled
fork installs next to it, and carries MuJoCo's include and library paths and the rpaths
(`DT_RPATH`) to the wrapper's and MuJoCo's library directories:

```bash
export PKG_CONFIG_PATH="$HOME/ws/lib/pkgconfig"
g++ -std=c++20 my_app.cpp -o my_app $(pkg-config --cflags --libs mjkdl)
```

With a KDL consumed from elsewhere (`MJKDL_OROCOS_KDL_INSTALL_DIR` or
`MJKDL_OROCOS_KDL_FROM_PACKAGE`), that KDL's `lib/pkgconfig` must be on `PKG_CONFIG_PATH`
as well (a colcon overlay puts it there); the rpath to its library directory is in
`mjkdl.pc`.

A consumer wanting MuJoCo without KDL, glfw and OpenGL links `mujoco::mujoco`, which the config
recreates for the exact copy this wrapper was built against:

```cmake
find_package(mjkdl REQUIRED)
target_link_libraries(my_app mujoco::mujoco)       # MuJoCo alone
```

`MJKDL_MUJOCO_DIR` and `MJKDL_MUJOCO_VERSION` are set alongside it for anyone who needs the
path rather than the target.

MuJoCo is intentionally not bundled into the prefix; consumers resolve it from
`MJKDL_MUJOCO_DIR` (or the `mujoco` pip package). Set
`-DMJKDL_INSTALL_BUNDLED_KDL=OFF` to keep KDL out of a shared prefix such as
`/usr/local`, where it could shadow a distro KDL - but then the install is no
longer self-contained: the library has no rpath to any KDL, so the loader takes the first
`liborocos-kdl` it finds, possibly a system one without the fork's solvers. Configure warns.
To share one KDL, prefer [One shared KDL across several projects](#one-shared-kdl-across-several-projects).

### Where the dependencies come from

Each dependency has a default source, or can point at something you already have:

| To... | Set |
|-------|-----|
| Use an existing MuJoCo install | `-DMJKDL_MUJOCO_DIR=/opt/mujoco-3.14.0` |
| Override the MuJoCo download URL | `-DMJKDL_MUJOCO_URL=<url>` |
| Skip the MuJoCo download | `-DMJKDL_FETCH_MUJOCO=OFF` (then set `MJKDL_MUJOCO_DIR`) |
| Build a KDL fork checkout from elsewhere | `-DMJKDL_OROCOS_KDL_DIR=~/src/orocos_kinematics_dynamics` |
| Build a different KDL branch/tag | check it out in `third_party/orocos_kinematics_dynamics` |
| Reuse a prebuilt KDL by prefix | `-DMJKDL_OROCOS_KDL_INSTALL_DIR=$HOME/ws` |
| Consume KDL via its CMake package | `-DMJKDL_OROCOS_KDL_FROM_PACKAGE=ON` |
| Keep bundled KDL out of the install prefix | `-DMJKDL_INSTALL_BUNDLED_KDL=OFF` |
| Choose build / install locations | `cmake -B <build-dir> -DCMAKE_INSTALL_PREFIX=<prefix>` |

KDL precedence when more than one is set: `MJKDL_OROCOS_KDL_FROM_PACKAGE` >
`MJKDL_OROCOS_KDL_INSTALL_DIR` > the in-tree build (`MJKDL_OROCOS_KDL_DIR`).

### One shared KDL across several projects

The default build bundles a private KDL, which is correct for a single project.
When several projects need KDL, build the fork once into a shared prefix and point
all of them at it, so exactly one `liborocos-kdl` exists (no duplicate copies or
rebuilds):

```bash
# 1. Build the fork once into the shared prefix
cmake -S <kdl-src>/orocos_kdl -B build/orocos_kdl \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX=$HOME/ws
cmake --build build/orocos_kdl --parallel $(nproc)
cmake --install build/orocos_kdl

# 2. Build the wrapper (and any sibling) against that shared KDL
cmake -B build -DCMAKE_INSTALL_PREFIX=$HOME/ws \
  -DMJKDL_OROCOS_KDL_INSTALL_DIR=$HOME/ws
cmake --build build --parallel $(nproc)
cmake --install build
```

`MJKDL_OROCOS_KDL_FROM_PACKAGE=ON` is the equivalent when the shared KDL is on
`CMAKE_PREFIX_PATH` as a CMake package rather than a known prefix - this is how
the [ROS 2 workflow](ros2.md) shares one KDL across a colcon overlay.

## Python

`pip install` builds the extension and bundles the secorolab Orocos KDL fork, PyKDL and the
MuJoCo plugins. MuJoCo itself is not bundled: the wheel pins `mujoco==3.14.0` and loads that
package's shared library. The build is isolated - it does not reuse any C++ build tree.

PyKDL is bundled inside the wheel as a top-level extension module. It imports as
`PyKDL` but does not appear as a separate package in `pip list` / `uv pip list`.

```bash
git clone https://github.com/vamsikalagaturu/mjkdl.git
cd mjkdl
vcs import < mjkdl.repos
uv pip install .
```

Verify:

```bash
python -c "import PyKDL, mujoco, mjkdl as mjk; print(mujoco.mj_versionString(), mjkdl.__mujoco_version__)"
```

### Python build options

All optional, passed via scikit-build-core:

| To... | Add |
|-------|-----|
| Editable install (dev), auto-rebuild on import | `uv pip install -e . --config-settings=editable.rebuild=true` |
| Put the build dir outside the source tree | `--config-settings=build-dir=/path/build_py/{wheel_tag}` |
| Build KDL from an existing checkout | `--config-settings=cmake.define.MJKDL_OROCOS_KDL_DIR=/path` |
| Reuse a prebuilt KDL+PyKDL prefix (skip bundling) | `--config-settings=cmake.define.MJKDL_OROCOS_KDL_INSTALL_DIR=/prefix` |

The shared-KDL option needs a prefix that also ships `PyKDL`; a C++-only install
prefix has KDL but no `PyKDL`, so the default (bundle) is right for standalone use.

### Models and examples

The wheel ships the `assets/` models (`mjkdl.ASSETS_DIR`) and the examples
(`mjkdl.examples`), which load the models from there:

```bash
python -m mjkdl.examples.ex_gravity_comp
```

Model paths and other model sources are documented in the
[Python Bindings API Guide](../api/python.md).

## Generate documentation

```bash
cmake -B build -DBUILD_DOCS=ON
cmake --build build --target docs
```

Open `build/docs/html/index.html`. The docs include the C++ headers, C++
examples, Markdown guides, Python stubs, and Python examples, and link KDL types
locally via a `kdl.tag` generated from the KDL the wrapper uses (the docs target builds
the fork first when it is the bundled one). Requires `doxygen` 1.9.8 or newer.

## All CMake options

Paths / sources:

| Flag | Default | Description |
|------|---------|-------------|
| `MJKDL_MUJOCO_DIR` | `~/.cache/mjkdl/mujoco-${MJKDL_MUJOCO_VERSION}` | MuJoCo location: download destination, or an existing install to use |
| `MJKDL_FETCH_MUJOCO` | `ON` | Download MuJoCo into the cache when `MJKDL_MUJOCO_DIR` is not present |
| `MJKDL_MUJOCO_URL` | (release) | MuJoCo archive URL to download |
| `MJKDL_OROCOS_KDL_DIR` | `third_party/orocos_kinematics_dynamics` | Fork source checkout (`vcs import < mjkdl.repos`), built in place |
| `MJKDL_OROCOS_KDL_INSTALL_DIR` | (empty) | Pre-installed Orocos KDL prefix to consume (skips building and bundling the fork) |
| `MJKDL_OROCOS_KDL_FROM_PACKAGE` | `OFF` | Consume Orocos KDL via `find_package(orocos_kdl)` on `CMAKE_PREFIX_PATH`; skips building and bundling the fork |

Build toggles:

| Flag | Default | Description |
|------|---------|-------------|
| `BUILD_RECORDER` | `ON` | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `BUILD_EXAMPLES` | `ON` | Build the `src/examples/ex_*` programs |
| `BUILD_TESTS` | `ON` | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS` | `OFF` | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the pybind11 extension (driven by the Python build) |
| `MJKDL_INSTALL_CPP_PACKAGE` | `ON` (`OFF` under scikit-build) | Install the C++ library, headers, CMake package config and `mjkdl.pc` |
| `MJKDL_INSTALL_BUNDLED_KDL` | `ON` | Install the built Orocos KDL fork into the prefix so the install is self-contained; `OFF` leaves the library without a path to KDL (configure warns). No effect with `MJKDL_OROCOS_KDL_INSTALL_DIR` / `MJKDL_OROCOS_KDL_FROM_PACKAGE` |
| `MJKDL_WITH_ROS` | `AUTO` | Build `mjkdl::camera_ros` (publishes a rendered camera as `sensor_msgs/Image` + `CameraInfo`): `AUTO` when `rclcpp` and `sensor_msgs` are found, `ON` requires them, `OFF` never |
| `SHOW_EQUALITY_PANEL` | `OFF` | Show the Simulate UI `Equality` section |
| `SHOW_GROUP_PANEL` | `OFF` | Show the Simulate UI `Group enable` section |
