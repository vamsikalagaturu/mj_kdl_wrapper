# Examples {#page_examples}

Most examples use the Kinova GEN3 7-DOF arm from MuJoCo Menagerie and the
bundled Robotiq 2F-85 gripper under `assets/robotiq_2f85`. Configure with
`-DMJ_KDL_FETCH_MENAGERIE=ON` to fetch the Kinova model automatically.

Build from the repo root:

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper

cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMJ_KDL_FETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)
```

Every example ends by itself. A C++ example (`src/examples/`) opens the Simulate UI unless
given `--headless`; a Python one (`python/mj_kdl_wrapper/examples/`) runs headless unless given
`--gui`. Either way it runs the same sequence or duration, prints its result, and exits;
closing the window ends it early. The headless runs self-check and are registered with CTest
(`ex_*_headless`). Exceptions: the Python `custom_ui_scene.py` and `viewer_scene.py` always
open a window (for a fixed 1 s and 10 s of simulated time). Recording: `ex_table_pour --record
[out.mp4]` (C++ and Python) and `ex_achd_pick_place --record` (C++) also write an MP4.

Most examples exist in both languages. Python only: `ex_cabinet`, `basic_scene`,
`custom_ui_scene`, `viewer_scene`.

The Python counterparts use the public Python wrapper and upstream `PyKDL`
bindings for FK, IK, RNEA, and ACHD instead of re-binding KDL classes locally.

| Example | Scene | Main behavior |
|---------|-------|---------------|
| `ex_gravity_comp` | arm only | hold home pose with KDL gravity compensation |
| `ex_pos_ctrl` | arm only | position trajectory tracking |
| `ex_vel_ctrl` | arm only | velocity-style convergence control |
| `ex_impedance` | arm + gripper | joint impedance with gripper inertia |
| `ex_table_scene` | table + free objects | table/object scene construction |
| `ex_cabinet` | 3-drawer cabinet, no robot | object-only scene with grasp sites and Simulate UI viewer |
| `ex_table_pick_place` | table + blue cube | scripted tabletop pick, transfer, release, and retreat |
| `ex_table_pour` | table + transparent receiver | scripted pour from small gripper-held bottle into a tabletop vessel |
| `ex_rnea_pick_place` | table + blue cube | Cartesian target interpolation with IK + RNEA inverse dynamics |
| `ex_achd_table_slide` | table contact | ACHD slide along +X while pressing 10 N down through ACHD's external-force input; C++ measures the table reaction |
| `ex_achd_pick_place` | table + blue cube | ACHD Cartesian pick/place with 6D TCP regulation and half-arm support wrench |
| `ex_admittance_ft` | table + wrist FT + gripper | Admittance control driven by a named force-torque sensor, RNEA task-space computed-torque inner loop |
| `ex_dual_arm` | two arms + grippers | multi-robot scene with independent KDL chains |
| `basic_scene` (Python) | arm only | minimal build, step and read |
| `custom_ui_scene` (Python) | arm only | the wrapper's Simulate UI for 1 s |
| `viewer_scene` (Python) | arm only | the scene in MuJoCo's own Python viewer for 10 s |

---

## ex_gravity_comp

**Scene:** Kinova GEN3 arm (arm only, no gripper).

**What it does:**
- Holds the arm at the home pose using KDL gravity compensation for 500 steps.
- `env.on_reset` re-homes the arm, so the UI's reset button restarts it.
- Headless, it measures the EE drift and prints it.

**Control law:** `CtrlMode::TORQUE` — KDL gravity compensation.

```
tau[i] = JntToGravity(q)[i]
```

**Headless output:** `EE drift after 500 steps: X.XXX mm`

---

## ex_pos_ctrl

**Scene:** Kinova GEN3 arm (arm only, no gripper).

**What it does:**
- Moves the arm from the home pose to a target pose using a linearly
  interpolated position setpoint over `kMotionDuration = 2.0 s`.
- Holds the final position for 1 s, then exits.

**Control law:** `CtrlMode::POSITION` — MuJoCo's built-in position actuator.

```
alpha  = clamp((t - t_start) / kMotionDuration, 0, 1)
cmd[i] = home[i] + alpha * (target[i] - home[i])
```

The wrapper's `update()` writes `jnt_pos_cmd` to MuJoCo `ctrl[]` each step;
the actuator handles the PD tracking internally.

**Headless output:** `max joint error at end: X.XXXX rad` (after motion + 1 s settle).

---

## ex_impedance

**Scene:** Kinova GEN3 arm + Robotiq 2F-85 gripper attached at the arm's `pinch_site`.

**What it does:**
- Holds the arm at the home pose using a joint-space impedance controller for 200 steps.
- Gripper cycles fully closed and open every 3 s (`fmod(t, 6) < 3 ? 0.82 : 0`; the gripper's
  ctrl is its driver angle, 0.82 rad closed).
- KDL chain is built with `tool_body = "g_base_mount"` so gripper inertia is lumped
  into the last segment — gravity compensation is correct for the full arm+gripper mass.

**Control law:** `CtrlMode::TORQUE` — PD + KDL gravity.

```
tau[i] = Kp[i] * (home[i] - q[i]) - Kd[i] * dq[i] + JntToGravity(q)[i]
Kp = [100, 200, 100, 200, 100, 200, 100]  Nm/rad
Kd = [10,  20,  10,  20,  10,  20,  10 ]  Nm*s/rad
```

**Headless output:** `EE drift after 200 steps: X.XXX mm`

---

## ex_table_scene

**Scene:** Kinova GEN3 + 2F-85 gripper, mounted on a table (`z = 0.7 m` surface)
with five free objects: 3 boxes (red, green, blue) and 2 spheres (orange, purple).

**What it does:**
- Builds the full scene via `SceneSpec` with an MJCF table asset and five
  primitive `SceneObject` entries using `Shape::BOX` and `Shape::SPHERE`.
- Arm holds home pose via KDL gravity compensation; gripper cycles open/closed.
- Objects sit on the table surface and respond to physics (can be knocked over).

**Control law:** `CtrlMode::TORQUE` — KDL gravity compensation.

```
tau[i] = JntToGravity(q)[i]          // gravity from scene's gravity_z
gripper_ctrl = fmod(t, 6) < 3 ? 0.82 : 0.0
```

Pure gravity compensation cannot hold the arm against the fingers' reaction while they move, so
the headless drift is tens of mm; with the gripper still it is about 0.1 mm.

**Headless output:** `EE drift after 500 steps: X.XXX mm`

---

## ex_table_pick_place

**Scene:** Kinova GEN3 + 2F-85 gripper mounted on a table, with a blue cube
on the tabletop.

**What it does:**
- Adds the table as an MJCF-backed `SceneObject` and places the robot base on the tabletop surface.
- Solves IK waypoints for a table pick, transfer, placement, release, and retreat.
- Runs a scripted sequence:
  `HOME -> PICK_ABOVE -> PICK -> CLOSE -> LIFT -> PLACE_ABOVE -> PLACE -> OPEN -> RETREAT -> HOLD`

**Control law:** `CtrlMode::TORQUE` — joint impedance (PD + KDL gravity).

```
tau[i] = g[i] + Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
```

**Headless output:** `cube final position: [x, y, z] target=[x, y, z] xy_error=X.XXX gripper=open`

---

## ex_table_pour

**Scene:** Kinova GEN3 + 2F-85 gripper mounted on a table, with a transparent
receiver vessel and small free spheres representing rice or pellets.

**What it does:**
- Attaches a small gripper-sized bottle to the gripper as a real MJCF tool
  attachment, so its mass is included through `ToolFrameSpec::tool_body`.
- Initializes the free spheres inside the attached bottle using the `pour_center` site frame.
- Solves IK waypoints for a pre-pour pose, pour pose, tilted pour pose, shake,
  retreat, and hold.
- Uses transparent fixed collision walls for the receiving vessel, so particles
  can visibly collect inside it.

**Control law:** `CtrlMode::TORQUE` — joint impedance (PD + KDL gravity), matching
the pick examples.

```
tau[i] = g[i] + Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
```

**Headless output:** `balls in transparent receiver: N/36` and the grain centroid; it fails
below 4 balls. The GUI run uses more balls and a longer pour hold.

---

## ACHD examples

The ACHD examples use `ChainHdSolver_Vereshchagin` to convert
Cartesian task accelerations into constrained joint accelerations.  For MuJoCo
torque control, the examples command the full inverse-dynamics torque computed
from that ACHD `qddot`:

```
pose error -> beta -> ACHD qddot -> RNEA(q, qdot, qddot) -> jnt_trq_cmd
```

- `ex_achd_table_slide` slides the TCP 0.2 m along +X with linear Z left free, pressing
  10 N down at the TCP through ACHD's external-force input with driver weights 1 (the pinned
  KDL fork's `setDriverWeights()`), gravity as feed-forward. The C++ example prints the mean
  table reaction over the second half of the slide, the command and their ratio (not judged),
  and exits 1 if contact was held for less than half of that window; the contact chatters while
  sliding. It also prints a one-shot nc=6 vs nc=5 comparison.
- `ex_achd_pick_place` runs a pick/place sequence.  During the place-side
  phases it feeds an ACHD-only upward support wrench on `half_arm_2_link` to
  keep the elbow/half-arm from dropping while preserving the TCP task.

Run headless:

```bash
./build/src/examples/ex_achd_table_slide --headless
./build/src/examples/ex_achd_pick_place --headless
```

---

## ex_admittance_ft

**Scene:** Kinova GEN3 + wrist FT sensor + Robotiq 2F-85 gripper mounted on a
table.

**What it does:** Admittance control for the whole run. Admittance is an outer
force->position loop wrapped around an inner motion controller; here the inner
loop is **RNEA task-space computed torque** (`CtrlMode::TORQUE`,
`ChainIkSolverVel_wdls` + `ChainIdSolver_RNE`): a Cartesian PD on the TCP pose
error becomes desired TCP acceleration, WDLS maps it to `qddot`, and RNEA maps
that to torque. It:
- Attach the bundled `ft_sensor.xml` between the wrist pinch site and the gripper
  and register `wrist_ft` as a named `ForceTorqueSensorSpec` (read as a
  KDL/PyKDL wrench).
- Close the gripper, let the wrist load settle, then tare the FT sensor (the
  gripper's ~10 N static load only appears once closed).
- Run a K=0 mass-damper admittance: an intro helical force (`spiral_force`)
  drives the TCP through a helix, then the FT-measured force takes over so a GUI
  ctrl + right-drag is sensed and yielded to; with K=0 the pose holds on release.
- Draw the commanded (yellow) and measured TCP (green) paths.

**Outer admittance law:**

```
f = bias - ft.force             # gravity-tared, deadbanded external force
a = (f - D*v - K*x) / M         # K = 0: no spring, holds on release
v += a*dt;  x += v*dt           # x is the TCP offset from the home pose
target_tcp = nominal_tcp translated by x
```

Run the C++ headless self-check or the viewer (the run ends by itself):

```bash
./build/src/examples/ex_admittance_ft --headless
./build/src/examples/ex_admittance_ft
```

Run the same Python example headless or with the viewer:

```bash
python examples/ex_admittance_ft.py            # headless self-check
python examples/ex_admittance_ft.py --gui
```

---

## ex_dual_arm

**Scene:** Two Kinova GEN3 arms with 2F-85 grippers in a shared simulation.
`arm1` at `x = -1.0 m`, `arm2` at `x = +1.0 m` with prefix `r2_`.

**What it does:**
- Places both robots in one `SceneSpec` with different positions and prefixes.
- Initialises two independent `Robot` handles and KDL chains from the same
  compiled `mjModel` by passing each arm's full body names (`base_link` vs
  `r2_base_link`) and tool names (`g_base_mount` vs `r2_g_base_mount`).
- Both arms hold home pose with PD + KDL gravity; grippers cycle open/closed.
- `jnt_trq_cmd` is primed before the loop so the very first physics step
  already gets correct gravity compensation.

**Control law:** `CtrlMode::TORQUE` — PD + KDL gravity for both arms independently.

```
tau1[i] = Kp[i]*(home[i]-q1[i]) - Kd[i]*dq1[i] + g1[i]
tau2[i] = Kp[i]*(home[i]-q2[i]) - Kd[i]*dq2[i] + g2[i]
```

One `update(&env)` reads and commands both arms; each arm has its own `ChainDynParam`.

**Headless output:** EE Cartesian positions for both arms after 600 steps.

---

## Recording

`--record [out.mp4]` runs an example headless and writes an H.264 MP4 through `VideoRecorder`
(EGL offscreen rendering + ffmpeg pipe); no window is needed.

```bash
./build/src/examples/ex_table_pour --record pour.mp4        # 1080p, default table_pour.mp4
./build/src/examples/ex_achd_pick_place --record pick.mp4   # 720p, default achd_pick_place.mp4
python -m mj_kdl_wrapper.examples.ex_table_pour --record pour.mp4
```

**Requires:** `BUILD_RECORDER=ON` (default) and `ffmpeg` in `PATH`.
