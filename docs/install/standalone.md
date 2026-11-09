# Standalone Installation Guide

Full reference for building and installing `mj_kdl_wrapper` without ROS, as a
plain CMake C++ library and/or a Python package. For ROS 2 / colcon, see the
[ROS 2 Installation Guide](ros2.md).

Instructions target Ubuntu/Debian. CMake checks `mjVERSION_HEADER` and stops if
`MJ_KDL_MUJOCO_DIR` points at an unsupported MuJoCo release.

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
| MuJoCo | `4.0.0`, tag in `mj_kdl_wrapper.repos`, `MJ_KDL_MUJOCO_VERSION` in `cmake/Versions.cmake` | Built from source for C++; the wheel compiles against pip `mujoco-core==4.0.0` and depends on `mujoco==4.0.0` |
| Orocos KDL | secorolab fork, branch `vereshchagin-driver-weighting`, from `mj_kdl_wrapper.repos` | Built from source; system `liborocos-kdl` is not used |
| vcstool | any | Checks out the KDL fork and MuJoCo (`vcs import < mj_kdl_wrapper.repos`) |
| CMake | `>=3.16` | Required to configure the C++ build |
| C++ compiler | C++20-capable | `CMAKE_CXX_STANDARD` is set to 20 |
| Python | `>=3.11` | Required for the Python package (MuJoCo 4.0 dropped 3.10) |
| scikit-build-core | `>=0.11.2` | Python build backend |
| pybind11 | `>=2.13` | Python binding build dependency |

## System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv vcstool \
  libeigen3-dev ffmpeg doxygen \
  libegl-dev libgl1-mesa-dev libwayland-dev libxkbcommon-dev \
  libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
```

`doxygen` is only needed for the docs target; `python3-dev`/`python3-venv` only
for the Python package; `ffmpeg` only for recording. `libegl-dev` is for the render library,
and the X11, Wayland and GL headers are what MuJoCo's Filament and SDL2 builds use for the
viewer; without them `AUTO` leaves those libraries out. `vcstool` is also on PyPI (`pip install vcstool`); ROS apt sources
name it `python3-vcstool`.

## C++ (CMake)

A standard CMake project. The default build is self-contained: it builds MuJoCo and the
Orocos KDL fork that `vcs import` checked out; the robot models ship in `assets/`. No system
MuJoCo or KDL is ever used.

It makes up to three libraries:

| Target | Header | What |
|--------|--------|------|
| `mj_kdl_wrapper::mj_kdl_wrapper` | `mj_kdl_wrapper.hpp` | the core: MuJoCo + KDL, no GL or window |
| `mj_kdl_wrapper::render` | `render.hpp` | `VideoRecorder`: offscreen rendering with MuJoCo's Filament renderer, MP4 through ffmpeg (`MJ_KDL_WITH_RENDER`) |
| `mj_kdl_wrapper::viewer` | `viewer.hpp` | `Viewer`: a MuJoCo Studio window (`MJ_KDL_WITH_VIEWER`, needs render) |

### Default build

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper

# check out the KDL fork and MuJoCo into third_party/
vcs import < mj_kdl_wrapper.repos

# configure
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# compile
cmake --build build --parallel $(nproc)

# run the test suite
ctest --test-dir build --output-on-failure
```

What happens during configure/build:

- **MuJoCo** is built from `MJ_KDL_MUJOCO_DIR` (default `third_party/mujoco`, tag `4.0.0`,
  where `vcs import` puts it) via `ExternalProject` into `build/mujoco`, with Filament for
  render and Studio for the viewer when those are enabled. Configure stops when the checkout
  is missing or its `mjVERSION_HEADER` is not 4.0.0. No system paths are searched.
- **render / viewer** are built when `MJ_KDL_WITH_RENDER` / `MJ_KDL_WITH_VIEWER` allow it
  (`AUTO`: when EGL, respectively X11, is found); configure prints `render : ON|OFF` and
  `viewer : ON|OFF`. Filament's materials and Studio's fonts are staged in
  `build/mujoco_assets`.
- **Orocos KDL** (the secorolab fork) is built from `MJ_KDL_OROCOS_KDL_DIR` (default
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

The install is self-contained. Alongside `libmj_kdl_wrapper.so`, the render and viewer
libraries when built, their headers, and the `mj_kdl_wrapper` CMake package config, the built
MuJoCo and Orocos KDL fork are installed into the same prefix:

```
<prefix>/lib/libmj_kdl_wrapper.so
<prefix>/lib/libmj_kdl_render.so             # with render
<prefix>/lib/libmj_kdl_viewer.so             # with the viewer
<prefix>/lib/libmujoco.so*
<prefix>/lib/mujoco_plugin/...
<prefix>/lib/mujoco_assets/...               # Filament materials, Studio fonts (render / viewer)
<prefix>/lib/liborocos-kdl.so*
<prefix>/include/mj_kdl_wrapper/...
<prefix>/include/mujoco/...
<prefix>/include/kdl/...
<prefix>/lib/cmake/mj_kdl_wrapper/...
<prefix>/lib/pkgconfig/mj_kdl_wrapper.pc
<prefix>/lib/pkgconfig/orocos-kdl.pc
<prefix>/share/orocos_kdl/cmake/...
<prefix>/share/doc/mj_kdl_wrapper/...        # LICENSE, NOTICE and third-party licenses
```

The wrapper is rpath'd to `$ORIGIN` as a `DT_RPATH`, not a `DT_RUNPATH`, so it loads the
co-installed MuJoCo and KDL without a build tree, and an `LD_LIBRARY_PATH` holding a system
`liborocos-kdl` cannot swap it out. Consumers:

```cmake
find_package(mj_kdl_wrapper REQUIRED)              # pulls KDL in transitively
target_link_libraries(my_app mj_kdl_wrapper::mj_kdl_wrapper)
# the optional libraries are components:
find_package(mj_kdl_wrapper REQUIRED COMPONENTS render viewer)
target_link_libraries(my_app mj_kdl_wrapper::viewer)   # brings render and the core
# or use KDL directly:
find_package(orocos_kdl REQUIRED)
target_link_libraries(my_app orocos-kdl)
```

A requested version matches only within its minor version: `find_package(mj_kdl_wrapper 0.3)`
does not accept `0.4.x`. The exported target carries `-Wl,--disable-new-dtags`, so a consumer
executable gets a `DT_RPATH` too.

Without CMake, use pkg-config; it covers the core library only. `mj_kdl_wrapper.pc` requires
`orocos-kdl`, which the bundled fork installs next to it, links `-lmujoco` from the same
library directory and carries the rpath (`DT_RPATH`) to it:

```bash
export PKG_CONFIG_PATH="$HOME/ws/lib/pkgconfig"
g++ -std=c++20 my_app.cpp -o my_app $(pkg-config --cflags --libs mj_kdl_wrapper)
```

With a KDL consumed from elsewhere (`MJ_KDL_OROCOS_KDL_INSTALL_DIR` or
`MJ_KDL_OROCOS_KDL_FROM_PACKAGE`), that KDL's `lib/pkgconfig` must be on `PKG_CONFIG_PATH`
as well (a colcon overlay puts it there); the rpath to its library directory is in
`mj_kdl_wrapper.pc`.

A consumer wanting MuJoCo without KDL links `mujoco::mujoco`, which the config recreates for
the copy installed in the prefix:

```cmake
find_package(mj_kdl_wrapper REQUIRED)
target_link_libraries(my_app mujoco::mujoco)       # MuJoCo alone
```

`MJ_KDL_MUJOCO_VERSION` is set alongside it.

Set `-DMJ_KDL_INSTALL_BUNDLED_KDL=OFF` to keep KDL out of a shared prefix such as
`/usr/local`, where it could shadow a distro KDL - but then the install is no
longer self-contained: the library has no rpath to any KDL, so the loader takes the first
`liborocos-kdl` it finds, possibly a system one without the fork's solvers. Configure warns.
To share one KDL, prefer [One shared KDL across several projects](#one-shared-kdl-across-several-projects).

### Where the dependencies come from

Each dependency has a default source, or can point at something you already have:

| To... | Set |
|-------|-----|
| Build a MuJoCo 4.0.0 checkout from elsewhere | `-DMJ_KDL_MUJOCO_DIR=~/src/mujoco` |
| Build the core only | `-DMJ_KDL_WITH_RENDER=OFF -DMJ_KDL_WITH_VIEWER=OFF` |
| Build a KDL fork checkout from elsewhere | `-DMJ_KDL_OROCOS_KDL_DIR=~/src/orocos_kinematics_dynamics` |
| Build a different KDL branch/tag | check it out in `third_party/orocos_kinematics_dynamics` |
| Reuse a prebuilt KDL by prefix | `-DMJ_KDL_OROCOS_KDL_INSTALL_DIR=$HOME/ws` |
| Consume KDL via its CMake package | `-DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON` |
| Keep bundled KDL out of the install prefix | `-DMJ_KDL_INSTALL_BUNDLED_KDL=OFF` |
| Choose build / install locations | `cmake -B <build-dir> -DCMAKE_INSTALL_PREFIX=<prefix>` |

KDL precedence when more than one is set: `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` >
`MJ_KDL_OROCOS_KDL_INSTALL_DIR` > the in-tree build (`MJ_KDL_OROCOS_KDL_DIR`).

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
  -DMJ_KDL_OROCOS_KDL_INSTALL_DIR=$HOME/ws
cmake --build build --parallel $(nproc)
cmake --install build
```

`MJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON` is the equivalent when the shared KDL is on
`CMAKE_PREFIX_PATH` as a CMake package rather than a known prefix - this is how
the [ROS 2 workflow](ros2.md) shares one KDL across a colcon overlay.

## Python

`pip install` builds the core library's extension and bundles the secorolab Orocos KDL fork
and PyKDL. It compiles against the headers and `libmujoco` of the pip `mujoco-core==4.0.0`
build requirement, with no MuJoCo source build, and the wheel depends on `mujoco==4.0.0`,
whose shared library and plugins it loads at runtime. Viewing and rendering use that package
too: `mj_kdl_wrapper.viewer.Viewer` (`mujoco.studio`) and `mujoco.rendering.filament`. The
build is isolated - it does not reuse any C++ build tree. Python >= 3.11.

PyKDL is bundled inside the wheel as a top-level extension module. It imports as
`PyKDL` but does not appear as a separate package in `pip list` / `uv pip list`.

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper
vcs import < mj_kdl_wrapper.repos
uv venv .venv --python 3.12
uv pip install .
```

Verify:

```bash
python -c "import PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mujoco.mj_versionString(), mjk.__mujoco_version__)"
```

### Python build options

All optional, passed via scikit-build-core:

| To... | Add |
|-------|-----|
| Editable install (dev), auto-rebuild on import | `uv pip install -e . --config-settings=editable.rebuild=true` |
| Put the build dir outside the source tree | `--config-settings=build-dir=/path/build_py/{wheel_tag}` |
| Build KDL from an existing checkout | `--config-settings=cmake.define.MJ_KDL_OROCOS_KDL_DIR=/path` |
| Reuse a prebuilt KDL+PyKDL prefix (skip bundling) | `--config-settings=cmake.define.MJ_KDL_OROCOS_KDL_INSTALL_DIR=/prefix` |

The shared-KDL option needs a prefix that also ships `PyKDL`; a C++-only install
prefix has KDL but no `PyKDL`, so the default (bundle) is right for standalone use.

### Models and examples

The wheel ships the `assets/` models (`mj_kdl_wrapper.ASSETS_DIR`) and the examples
(`mj_kdl_wrapper.examples`), which load the models from there:

```bash
python -m mj_kdl_wrapper.examples.ex_gravity_comp
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
| `MJ_KDL_MUJOCO_DIR` | `third_party/mujoco` | MuJoCo source checkout (`vcs import < mj_kdl_wrapper.repos`), built into `build/mujoco`; ignored by a wheel build, which uses pip `mujoco-core` |
| `MJ_KDL_OROCOS_KDL_DIR` | `third_party/orocos_kinematics_dynamics` | Fork source checkout (`vcs import < mj_kdl_wrapper.repos`), built in place |
| `MJ_KDL_OROCOS_KDL_INSTALL_DIR` | (empty) | Pre-installed Orocos KDL prefix to consume (skips building and bundling the fork) |
| `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` | `OFF` | Consume Orocos KDL via `find_package(orocos_kdl)` on `CMAKE_PREFIX_PATH`; skips building and bundling the fork |

Build toggles:

| Flag | Default | Description |
|------|---------|-------------|
| `MJ_KDL_WITH_RENDER` | `AUTO` | Build `mj_kdl_wrapper::render` (`VideoRecorder`, Filament offscreen rendering): `AUTO` when EGL is found, `ON` requires it, `OFF` never; forced `OFF` in a wheel build |
| `MJ_KDL_WITH_VIEWER` | `AUTO` | Build `mj_kdl_wrapper::viewer` (MuJoCo Studio window): `AUTO` when X11 is found and render is built, `ON` requires both, `OFF` never; forced `OFF` in a wheel build |
| `BUILD_EXAMPLES` | `ON` | Build the `src/examples/ex_*` programs; only with the viewer |
| `BUILD_TESTS` | `ON` | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS` | `OFF` | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the pybind11 extension (driven by the Python build) |
| `MJ_KDL_INSTALL_CPP_PACKAGE` | `ON` (`OFF` under scikit-build) | Install the C++ library, headers, CMake package config and `mj_kdl_wrapper.pc` |
| `MJ_KDL_INSTALL_BUNDLED_KDL` | `ON` | Install the built Orocos KDL fork into the prefix so the install is self-contained; `OFF` leaves the library without a path to KDL (configure warns). No effect with `MJ_KDL_OROCOS_KDL_INSTALL_DIR` / `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` |
| `MJ_KDL_WITH_ROS` | `AUTO` | Build `mj_kdl_wrapper::camera_ros` (publishes a rendered camera as `sensor_msgs/Image` + `CameraInfo`): `AUTO` when `rclcpp` and `sensor_msgs` are found, `ON` requires them, `OFF` never |
