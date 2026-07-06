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

Every C++ example accepts `--headless` to skip the GLFW window and run a fixed
number of physics steps, printing a brief result to stdout. Every
`src/examples/ex_*.cpp` file also has a same-name Python counterpart in
`python/examples/`; Python examples run headless by default and accept `--gui`
where a Simulate UI view is useful.

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
| `ex_pick` | floor cube | scripted pick and lift |
| `ex_table_pick_place` | table + blue cube | scripted tabletop pick, transfer, release, and retreat |
| `ex_table_pour` | table + transparent receiver | scripted pour from small gripper-held bottle into a tabletop vessel |
| `ex_rnea_pick_place` | table + blue cube | Cartesian target interpolation with IK + RNEA inverse dynamics |
| `ex_achd_table_slide` | table contact | ACHD partial constraint comparison with wrist/table support |
| `ex_achd_pick_place` | table + blue cube | ACHD Cartesian pick/place with 6D TCP regulation and half-arm support wrench |
| `ex_admittance_ft` | table + wrist FT + gripper | Admittance control (POSITION inner loop) driven by a named force-torque sensor |
| `ex_admittance_ft_rnea` | table + wrist FT + gripper | Same admittance, RNEA task-space computed-torque inner loop |
| `ex_admittance_ft_achd` | table + wrist FT + gripper | Same admittance, ACHD (Vereshchagin) task-space inner loop |
| `ex_dual_arm` | two arms + grippers | multi-robot scene with independent KDL chains |
| `ex_record` | arm only | headless MP4 recording |

---

## ex_gravity_comp

**Scene:** Kinova GEN3 arm (arm only, no gripper).

**What it does:**
- Holds the arm at the home pose using KDL gravity compensation.
- In GUI mode: detects sim resets (time going backward) and re-seeds the state.
- In headless mode: runs 500 steps, measures EE drift, and prints it.

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
- After the trajectory finishes, holds the final position indefinitely.

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

**Scene:** Kinova GEN3 arm + Robotiq 2F-85 gripper attached at `bracelet_link`.

**What it does:**
- Holds the arm at the home pose using a joint-space impedance controller.
- Gripper cycles fully closed and open every 3 s (`fmod(t, 6) < 3 ? 0.8 : 0`).
- KDL chain is built with `tool_body = "g_base"` so gripper inertia is lumped
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
gripper_ctrl = fmod(t, 6) < 3 ? 0.8 : 0.0
```

**Headless output:** `EE drift after 500 steps: X.XXX mm`

---

## ex_pick

**Scene:** Kinova GEN3 + 2F-85 gripper + orange cube (4 cm, mass 0.1 kg) on the floor.

**What it does:**
- Runs a scripted pick-and-place sequence via a 6-state machine:
  `HOME -> PREGRASP -> GRASP -> CLOSE -> LIFT -> HOLD`
- IK waypoints are solved offline using `KDL::ChainIkSolverPos_NR_JL` with
  joint limits, seeded from each prior solution.
- Each state linearly interpolates from the measured pose at state-entry
  (`q_enter`) to the IK target over a nominal duration, transitioning when
  both elapsed time and joint error criteria are met (or on timeout).

**Control law:** `CtrlMode::TORQUE` — joint impedance (PD + KDL gravity) throughout all states.

```
tau[i] = g[i] + Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
q_des linearly interpolated: q_enter + alpha * (q_target - q_enter)
alpha = clamp((t - t_enter) / duration, 0, 1)
```

State table (durations in seconds):

| State    | Duration | Timeout | Settle tol | Gripper |
|----------|----------|---------|------------|---------|
| HOME     | 1.0      | 2.5     | 0.08 rad   | open    |
| PREGRASP | 5.0      | 7.0     | 0.08 rad   | open    |
| GRASP    | 5.0      | 8.0     | 0.03 rad   | open    |
| CLOSE    | 1.5      | 2.5     | (none)     | closed  |
| LIFT     | 3.0      | 5.0     | 0.08 rad   | closed  |
| HOLD     | 10.0 (GUI) / 1.0 (headless) | same | (none)     | closed  |

**Headless output:** `cube Z after pick: X.XXX m`

---

## ex_table_pick_place

**Scene:** Kinova GEN3 + 2F-85 gripper mounted on a table, with a blue cube
on the tabletop.

**What it does:**
- Adds the table as an MJCF-backed `SceneObject` and places the robot base on the tabletop surface.
- Solves IK waypoints for a table pick, transfer, placement, release, and retreat.
- Runs a scripted sequence:
  `HOME -> PICK_ABOVE -> PICK -> CLOSE -> LIFT -> PLACE_ABOVE -> PLACE -> OPEN -> RETREAT -> HOLD`

**Control law:** `CtrlMode::TORQUE` — joint impedance (PD + KDL gravity), matching `ex_pick`.

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

**Headless output:** `balls in transparent receiver: N/36`

---

## ACHD examples

The ACHD examples use `ChainHdSolver_Vereshchagin_Fixed_Joint` to convert
Cartesian task accelerations into constrained joint accelerations.  For MuJoCo
torque control, the examples command the full inverse-dynamics torque computed
from that ACHD `qddot`:

```
pose error -> beta -> ACHD qddot -> RNEA(q, qdot, qddot) -> jnt_trq_cmd
```

- `ex_achd_table_slide` compares table-supported motion with linear-Z
  constrained vs unconstrained.
- `ex_achd_pick_place` runs a pick/place sequence.  During the place-side
  phases it feeds an ACHD-only upward support wrench on `half_arm_2_link` to
  keep the elbow/half-arm from dropping while preserving the TCP task.

Run headless:

```bash
./build/src/examples/ex_achd_table_slide --headless
./build/src/examples/ex_achd_pick_place --headless
```

---

## ex_admittance_ft / ex_admittance_ft_rnea / ex_admittance_ft_achd

**Scene:** Kinova GEN3 + wrist FT sensor + Robotiq 2F-85 gripper mounted on a
table.

**What they do:** Admittance control for the whole run. Admittance is an outer
force->position loop wrapped around an inner motion controller; the three
examples share the outer loop and differ only in the inner loop:

- `ex_admittance_ft` -- ideal **POSITION** inner loop (`CtrlMode::POSITION`,
  `set_joint_pos`): the TCP follows the commanded offset exactly.
- `ex_admittance_ft_rnea` -- **RNEA task-space computed-torque** inner loop
  (`CtrlMode::TORQUE`, `ChainIkSolverVel_wdls` + `ChainIdSolver_RNE`): a
  Cartesian PD on TCP pose error becomes desired TCP acceleration, WDLS maps it
  to `qddot`, and RNEA maps that to torque.
- `ex_admittance_ft_achd` -- **ACHD task-space** inner loop (`CtrlMode::TORQUE`,
  `ChainHdSolver_Vereshchagin_Fixed_Joint` + `ChainIdSolver_RNE`): a Cartesian PD on the TCP
  pose error is the desired acceleration (`beta`), ACHD resolves it into joint
  accelerations through the constrained dynamics, and RNEA maps those to torque.
  No IK step -- the Cartesian target feeds the solver directly.

All three:
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

Run C++ headless self-checks or interactive windows:

```bash
./build/src/examples/ex_admittance_ft --headless
./build/src/examples/ex_admittance_ft_rnea --headless
./build/src/examples/ex_admittance_ft_achd --headless
./build/src/examples/ex_admittance_ft
```

Run the same Python examples headless or interactive:

```bash
python examples/ex_admittance_ft.py            # headless self-check
python examples/ex_admittance_ft.py --gui
python examples/ex_admittance_ft_rnea.py
python examples/ex_admittance_ft_rnea.py --gui
python examples/ex_admittance_ft_achd.py
python examples/ex_admittance_ft_achd.py --gui
```

---

## ex_dual_arm

**Scene:** Two Kinova GEN3 arms with 2F-85 grippers in a shared simulation.
`arm1` at `x = -1.0 m`, `arm2` at `x = +1.0 m` with prefix `r2_`.

**What it does:**
- Places both robots in one `SceneSpec` with different positions and prefixes.
- Initialises two independent `Robot` handles and KDL chains from the same
  compiled `mjModel` by passing matching `base_link`/`bracelet_link` names
  with the correct prefix (`""` vs `"r2_"`).
- Both arms hold home pose with PD + KDL gravity; grippers cycle open/closed.
- `jnt_trq_cmd` is primed before the loop so the very first physics step
  already gets correct gravity compensation.

**Control law:** `CtrlMode::TORQUE` — PD + KDL gravity for both arms independently.

```
tau1[i] = Kp[i]*(home[i]-q1[i]) - Kd[i]*dq1[i] + g1[i]
tau2[i] = Kp[i]*(home[i]-q2[i]) - Kd[i]*dq2[i] + g2[i]
```

Each arm calls `update()` and `JntToGravity` separately every step.

**Headless output:** EE Cartesian positions for both arms after 600 steps.

---

## ex_record

**Scene:** Kinova GEN3 arm (arm only).

**What it does:**
- Records a 5 s gravity-compensated simulation to an H.264 MP4 using
  `VideoRecorder` (EGL offscreen rendering + ffmpeg pipe).
- No GLFW window is needed.
- Camera orbits 360 degrees around the arm over the recording duration.
- Frames are captured at exactly `kFps = 60` Hz regardless of the physics
  timestep (`steps_per_frame = floor(1 / (fps * dt))`).

**Control law:** `CtrlMode::TORQUE` — KDL gravity compensation.

```
tau[i] = JntToGravity(q)[i]
```

**Usage:**

```bash
./build/ex_record [output.mp4] [360p|480p|720p|1080p|2k|4k]
# e.g.
./build/ex_record fly.mp4 720p
```

Resolution presets map to 16:9 dimensions (`R1080p` = 1920x1080, etc.).
Default is `1080p`; output file defaults to `sim.mp4`.

**Requires:** `BUILD_RECORDER=ON` (default) and `ffmpeg` in `PATH`.
