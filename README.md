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

### C++ library, examples, and tests

```bash
git clone https://github.com/secorolab/mj_kdl_wrapper.git
cd mj_kdl_wrapper
cmake -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DMJ_KDL_FETCH_MUJOCO=ON \
  -DMJ_KDL_FETCH_OROCOS_KDL=ON \
  -DFETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

The three fetch flags are all ON by default; they are shown above to make the
sources explicit:

- `MJ_KDL_FETCH_MUJOCO` downloads MuJoCo 3.9.0 unless `MUJOCO_ROOT` points at an
  install.
- `MJ_KDL_FETCH_OROCOS_KDL` clones and builds the secorolab Orocos KDL fork - the
  only KDL used; no system `liborocos-kdl` is consulted.
- `FETCH_MENAGERIE` downloads the Kinova GEN3 / Robotiq models the examples and
  tests use.

Point any of these at custom locations with the [CMake options](#cmake-options)
below. Install for use from another CMake project via
`find_package(mj_kdl_wrapper)` (add `-DCMAKE_INSTALL_PREFIX="$HOME/.local"` for a
user-local prefix):

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

Paths / sources (override to use your own):

| Flag | Default | Description |
|------|---------|-------------|
| `MUJOCO_ROOT` | `/opt/mujoco-3.9.0` | Existing MuJoCo install to use |
| `MJ_KDL_FETCH_MUJOCO` | `ON` | Download MuJoCo when `MUJOCO_ROOT` is not present |
| `MJ_KDL_MUJOCO_URL` | (release) | MuJoCo archive URL to download |
| `MJ_KDL_FETCH_OROCOS_KDL` | `ON` | Clone and build the secorolab Orocos KDL fork (the only KDL used) |
| `MJ_KDL_OROCOS_KDL_GIT_REPOSITORY` | secorolab fork | Orocos KDL git source to build |
| `MJ_KDL_OROCOS_KDL_GIT_TAG` | `feature/achd_fixed_joint` | Orocos KDL branch/tag to build |
| `MJ_KDL_OROCOS_KDL_SOURCE_DIR` | (empty) | Local Orocos KDL fork checkout to build instead of cloning |
| `FETCH_MENAGERIE` | `OFF` | Download MuJoCo Menagerie models |
| `MJ_KDL_MENAGERIE_DIR` | `third_party/menagerie` | Menagerie location / `FETCH_MENAGERIE` destination |

Build toggles:

| Flag | Default | Description |
|------|---------|-------------|
| `BUILD_RECORDER` | `ON` | Enable `VideoRecorder` (EGL + ffmpeg headless recording) |
| `BUILD_EXAMPLES` | `ON` | Build the `src/examples/ex_*` programs |
| `BUILD_TESTS` | `ON` | Build and register GoogleTest tests with CTest |
| `BUILD_DOCS` | `OFF` | Generate Doxygen HTML docs (`cmake --build build --target docs`) |
| `SHOW_EQUALITY_PANEL` | `OFF` | Show the Simulate UI `Equality` section |
| `SHOW_GROUP_PANEL` | `OFF` | Show the Simulate UI `Group enable` section |

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
