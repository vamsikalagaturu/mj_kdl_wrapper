# Python Bindings

The Python package exposes the same scene, robot, reset, viewer, and recorder
concepts as the C++ wrapper while keeping ownership explicit. It owns MuJoCo
`mjModel`/`mjData` through `Scene` or `Env` and returns KDL values through the
upstream `PyKDL` module instead of defining duplicate Python KDL classes.

## Minimal Scene

```python
import mj_kdl_wrapper as mjk

spec = mjk.SceneSpec()
spec.timestep = 0.002
spec.add_floor = True
spec.add_skybox = True

robot_spec = mjk.RobotSpec()
robot_spec.path = "third_party/menagerie/kinova_gen3/gen3.xml"
spec.robots = [robot_spec]

scene = mjk.Scene.build(spec)
robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")

robot.ctrl_mode = mjk.CtrlMode.POSITION
robot.jnt_pos_cmd = [0.0] * robot.n_joints
robot.update()
robot.step()

scene.save_xml("scene.xml")
scene.close()
```

## PyKDL Interop

The binding layer constructs Python objects from PyKDL for KDL return types.
That means downstream Python code should use the regular PyKDL solvers:

```python
import PyKDL as kdl
import mj_kdl_wrapper as mjk

scene = mjk.Scene.build(spec)
robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")

chain = robot.kdl_chain()
fk = kdl.ChainFkSolverPos_recursive(chain)
q = kdl.JntArray(robot.n_joints)
for i, value in enumerate(robot.jnt_pos_msr):
    q[i] = value

tcp = kdl.Frame()
fk.JntToCart(q, tcp)
```

`Robot.set_joint_pos()` and `Robot.fk_frame(q)` accept both Python sequences and
`PyKDL.JntArray`, so examples can stay close to equivalent C++ KDL code.

`Scene.body_frame()` and `Scene.site_frame()` return `PyKDL.Frame`.
`Robot.kdl_chain()` returns `PyKDL.Chain`, so callers can construct the normal
PyKDL FK, IK, RNEA, and ACHD solvers.

## Runtime Mutation And Ownership

`Scene` and `Env` own the compiled MuJoCo model/data. `Robot` handles borrow
those pointers and are automatically rebound after public object mutations:

```python
cube = mjk.SceneObject()
cube.name = "cube"
cube.shape = mjk.Shape.BOX
cube.size = [0.02, 0.02, 0.02]
cube.pos = [0.4, 0.0, 0.02]
cube.rgba = [1.0, 0.5, 0.0, 1.0]
cube.mass = 0.1
cube.friction = [0.8, 0.02, 0.001]

scene.add_object(cube)     # existing Robot handles remain valid
robot.update()
scene.remove_object("cube")
```

Calling `Scene.close()` or `Env.close()` invalidates dependent robot handles.
Using an invalidated robot raises `RuntimeError("robot is closed")` instead of
leaving Python with dangling MuJoCo pointers.

`Scene.set_body_pose(name, pos, quat=None)` accepts quaternions in Python
`xyzw` order. The binding converts them to MuJoCo's `wxyz` order before calling
the C++ helper.

## Simulate UI And Viewer Bridge

Run the custom Simulate UI with wrapper panels for `Frames`, `Trace`,
`Perturb`, `Recorder`, and `RTF`:

```bash
python3 python/examples/custom_ui_scene.py
```

Run the official MuJoCo viewer bridge:

```bash
python3 python/examples/viewer_scene.py
```

`viewer_scene.py` exports a temporary `.mjb` and opens it in a separate Python
process, so the installed `mujoco` Python package must match the wrapper-linked
MuJoCo version reported by `mj_kdl_wrapper.mujoco_version()`.

## Examples

Every C++ `src/examples/ex_*.cpp` example has a Python counterpart with the
same base name in `python/examples/`.

Run headless:

```bash
python3 python/examples/ex_pos_ctrl.py
python3 python/examples/ex_vel_ctrl.py
python3 python/examples/ex_pick.py
python3 python/examples/ex_table_pick_place.py
python3 python/examples/ex_table_pour.py
python3 python/examples/ex_record.py
```

Add `--gui` to open the wrapper Simulate UI where the example supports it:

```bash
python3 python/examples/ex_table_scene.py --gui
```

The Python examples intentionally use the public Python wrapper and PyKDL APIs.
They do not reach into raw MuJoCo `qpos`/`qvel` arrays; free-body resets use
`Scene.set_body_pose()` and pose queries use `Scene.body_frame()` or
`Scene.site_frame()`.

## API Documentation Generation

Doxygen includes:

- C++ headers and examples.
- Markdown guides in `docs/`.
- Python stubs from `python/mj_kdl_wrapper/*.pyi`.
- Python examples from `python/examples/*.py`.

Generate HTML docs with:

```bash
cmake -B build -DBUILD_DOCS=ON
cmake --build build --target docs
```

The output is written to `build/docs/html/index.html`.
