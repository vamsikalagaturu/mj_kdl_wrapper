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
| MuJoCo | `3.9.0` from `cmake/Versions.cmake` | Native library and pinned `mujoco` Python package must match |
| Orocos KDL | secorolab fork, `feature/achd_fixed_joint` | Built from source; system `liborocos-kdl` is not used |
| CMake | `>=3.16` | Required to configure the C++ build |
| C++ compiler | C++20-capable | `CMAKE_CXX_STANDARD` is set to 20 |
| Python | `>=3.10` | Required for the Python package |
| scikit-build-core | `>=0.11.2` | Python build backend |
| pybind11 | `>=2.13` | Python binding build dependency |

## System packages

```bash
sudo apt update
sudo apt install \
  cmake g++ git python3-dev python3-pip python3-venv \
  libeigen3-dev libglfw3-dev libgl-dev libegl-dev \
  ffmpeg doxygen
```

`doxygen` is only needed for the docs target; `python3-dev`/`python3-venv` only
for the Python package.

## C++ (CMake)

A standard CMake project. The default build is self-contained: it downloads
MuJoCo, clones and builds the Orocos KDL fork, and (with the menagerie flag)
fetches the robot models. No system MuJoCo or KDL is ever used.

### Default build

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper

# configure (downloads MuJoCo, builds the KDL fork, fetches Menagerie models)
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMJ_KDL_FETCH_MENAGERIE=ON

# compile
cmake --build build --parallel $(nproc)

# run the test suite
ctest --test-dir build --output-on-failure
```

What happens during configure/build:

- **MuJoCo** is downloaded (`MJ_KDL_FETCH_MUJOCO=ON`) unless `MJ_KDL_MUJOCO_DIR`
  already holds a matching install.
- **Orocos KDL** (the secorolab fork) is cloned into
  `MJ_KDL_OROCOS_KDL_DIR` (default `third_party/orocos_kinematics_dynamics`) and
  built via `ExternalProject` into `build/orocos_kdl_install`. The checkout
  persists across builds and is reused (no re-clone) on later configures.
- **Menagerie** robot models and the bundled `assets/` (Robotiq gripper, table,
  mug) are fetched/copied into the user cache `~/.cache/mj_kdl_wrapper`
  (`$XDG_CACHE_HOME` honored) only with `-DMJ_KDL_FETCH_MENAGERIE=ON`. The C++
  examples and tests resolve both from that cache via `example_paths.hpp`
  (`$MJ_KDL_MENAGERIE` overrides the model location); they self-skip when it is
  empty.

### Install

```bash
# choose the prefix at configure time
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/ws"
cmake --build build --parallel $(nproc)
cmake --install build
```

The install is self-contained. Alongside `libmj_kdl_wrapper.so`, its headers, and
the `mj_kdl_wrapper` CMake package config, the built Orocos KDL fork is installed
into the same prefix:

```
<prefix>/lib/libmj_kdl_wrapper.so
<prefix>/lib/liborocos-kdl.so*
<prefix>/include/mj_kdl_wrapper/...
<prefix>/include/kdl/...
<prefix>/lib/cmake/mj_kdl_wrapper/...
<prefix>/share/orocos_kdl/cmake/...
```

The wrapper is rpath'd to `$ORIGIN`, so it loads the co-installed KDL without a
build tree or `LD_LIBRARY_PATH`. Consumers:

```cmake
find_package(mj_kdl_wrapper REQUIRED)              # pulls KDL in transitively
target_link_libraries(my_app mj_kdl_wrapper::mj_kdl_wrapper)
# or use KDL directly:
find_package(orocos_kdl REQUIRED)
target_link_libraries(my_app orocos-kdl)
```

MuJoCo is intentionally not bundled into the prefix; consumers resolve it from
`MJ_KDL_MUJOCO_DIR` (or the `mujoco` pip package). Set
`-DMJ_KDL_INSTALL_BUNDLED_KDL=OFF` to keep KDL out of a shared prefix such as
`/usr/local`, where it could shadow a distro KDL - but then the install is no
longer self-contained and KDL must be provided some other way.

### Where the dependencies come from

Each dependency is fetched by default, or can point at something you already have:

| To... | Set |
|-------|-----|
| Use an existing MuJoCo install | `-DMJ_KDL_MUJOCO_DIR=/opt/mujoco-3.9.0` |
| Override the MuJoCo download URL | `-DMJ_KDL_MUJOCO_URL=<url>` |
| Skip the MuJoCo download | `-DMJ_KDL_FETCH_MUJOCO=OFF` (then set `MJ_KDL_MUJOCO_DIR`) |
| Clone the KDL fork somewhere specific | `-DMJ_KDL_OROCOS_KDL_DIR=~/src/orocos_kinematics_dynamics` |
| Build a different KDL branch/tag | `-DMJ_KDL_OROCOS_KDL_GIT_TAG=<ref>` |
| Reuse a prebuilt KDL by prefix | `-DMJ_KDL_OROCOS_KDL_INSTALL_DIR=$HOME/ws` |
| Consume KDL via its CMake package | `-DMJ_KDL_OROCOS_KDL_FROM_PACKAGE=ON` |
| Keep bundled KDL out of the install prefix | `-DMJ_KDL_INSTALL_BUNDLED_KDL=OFF` |
| Fetch Menagerie models | `-DMJ_KDL_FETCH_MENAGERIE=ON` |
| Choose build / install locations | `cmake -B <build-dir> -DCMAKE_INSTALL_PREFIX=<prefix>` |

KDL precedence when more than one is set: `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` >
`MJ_KDL_OROCOS_KDL_INSTALL_DIR` > the in-tree build (`MJ_KDL_OROCOS_KDL_DIR` /
`MJ_KDL_FETCH_OROCOS_KDL`).

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

`pip install` builds the extension, bundles MuJoCo's matching shared library
dependency, the secorolab Orocos KDL fork, and PyKDL, and pins `mujoco==3.9.0`.
The build is isolated and self-contained - it does not reuse any C++ build tree.

PyKDL is bundled inside the wheel as a top-level extension module. It imports as
`PyKDL` but does not appear as a separate package in `pip list` / `uv pip list`.

```bash
uv pip install "git+https://github.com/vamsikalagaturu/mj_kdl_wrapper.git"  # from GitHub
uv pip install .                                                            # from a checkout
```

Verify:

```bash
python -c "import PyKDL, mujoco, mj_kdl_wrapper as mjk; print(mujoco.mj_versionString(), mjk.mujoco_version())"
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

### Menagerie models and examples

The examples ship in the wheel; `mj-kdl-fetch-examples` copies them out as an
`examples/` directory. The scripts resolve MuJoCo Menagerie models and bundled
assets from the user cache, populated once by `mj-kdl-fetch-menagerie`, so the
copied scripts run from any directory:

```bash
mj-kdl-fetch-examples                         # copies into ./mj_kdl_wrapper_examples
mj-kdl-fetch-menagerie                        # populates the user cache (models + assets)
cd mj_kdl_wrapper_examples
python examples/ex_gravity_comp.py

# or point at a custom Menagerie checkout
mj-kdl-fetch-menagerie --dest /path/to/menagerie
export MJ_KDL_MENAGERIE=/path/to/menagerie
```

Model resolution, environment variables, and using other model sources are
documented in the [Python Bindings API Guide](../api/python.md).

## Generate documentation

```bash
cmake -B build -DBUILD_DOCS=ON
cmake --build build --target docs
```

Open `build/docs/html/index.html`. The docs include the C++ headers, C++
examples, Markdown guides, Python stubs, and Python examples, and link KDL types
locally via a generated `kdl.tag`. Requires `doxygen`.

## All CMake options

Paths / sources:

| Flag | Default | Description |
|------|---------|-------------|
| `MJ_KDL_MUJOCO_DIR` | `/opt/mujoco-${MJ_KDL_MUJOCO_VERSION}` | Existing MuJoCo install to use |
| `MJ_KDL_FETCH_MUJOCO` | `ON` | Download MuJoCo when `MJ_KDL_MUJOCO_DIR` is not present |
| `MJ_KDL_MUJOCO_URL` | (release) | MuJoCo archive URL to download |
| `MJ_KDL_FETCH_OROCOS_KDL` | `ON` | Clone and build the secorolab Orocos KDL fork (the only KDL used) |
| `MJ_KDL_OROCOS_KDL_GIT_REPOSITORY` | secorolab fork | Orocos KDL git source to build |
| `MJ_KDL_OROCOS_KDL_GIT_TAG` | `feature/achd_fixed_joint` | Orocos KDL branch/tag to build |
| `MJ_KDL_OROCOS_KDL_DIR` | `third_party/orocos_kinematics_dynamics` | Fork source/clone destination; built in place if present, else cloned here when fetch is ON |
| `MJ_KDL_OROCOS_KDL_INSTALL_DIR` | (empty) | Pre-installed Orocos KDL prefix to consume (skips building and bundling the fork) |
| `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` | `OFF` | Consume Orocos KDL via `find_package(orocos_kdl)` on `CMAKE_PREFIX_PATH`; skips building and bundling the fork |
| `MJ_KDL_FETCH_MENAGERIE` | `OFF` | Download MuJoCo Menagerie models |
| `MJ_KDL_MENAGERIE_DIR` | `$XDG_CACHE_HOME/mj_kdl_wrapper/menagerie` (or `~/.cache/mj_kdl_wrapper/menagerie`) | Menagerie location / `MJ_KDL_FETCH_MENAGERIE` destination |

Build toggles:

| Flag | Default | Description |
|------|---------|-------------|
| `BUILD_RECORDER` | `ON` | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `BUILD_EXAMPLES` | `ON` | Build the `src/examples/ex_*` programs |
| `BUILD_TESTS` | `ON` | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS` | `OFF` | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the pybind11 extension (driven by the Python build) |
| `MJ_KDL_INSTALL_CPP_PACKAGE` | `ON` (`OFF` under scikit-build) | Install the C++ library, headers, and CMake package config |
| `MJ_KDL_INSTALL_BUNDLED_KDL` | `ON` | Install the built Orocos KDL fork into the prefix so the install is self-contained. No effect with `MJ_KDL_OROCOS_KDL_INSTALL_DIR` / `MJ_KDL_OROCOS_KDL_FROM_PACKAGE` |
| `SHOW_EQUALITY_PANEL` | `OFF` | Show the Simulate UI `Equality` section |
| `SHOW_GROUP_PANEL` | `OFF` | Show the Simulate UI `Group enable` section |
