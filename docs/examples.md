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
closing the window ends it early. Headless, each example checks its own goal (the limit is
printed next to the measured value), prints `PASS` or `FAIL`, and exits 1 on a failure. The C++
headless runs are registered with CTest (`ex_*_headless`); the Python ones run under pytest
(`python/tests/test_examples.py`). Recording: `ex_table_pour --record [out.mp4]` (C++ and
Python) and `ex_achd_pick_place --record` (C++) also write an MP4.

Every example exists in both languages, with the same scene, gains, durations and limits.

The Python counterparts use the public Python wrapper and upstream `PyKDL`
bindings for FK, IK, RNEA, and ACHD instead of re-binding KDL classes locally.

**Control cycle.** Every control loop does one `update()` per physics step:

```
update(env)      # read the state; write the command computed in the previous cycle
compute command  # into the robot's ports
step(env)
```

`update()` reads the measurements and then writes the active mode's command in one call, like
one exchange on a fieldbus, so a command reaches the actuators one control period after the
state it was computed from. `env.on_reset` primes the first torque command (gravity at the reset
pose), so the first step after a reset already holds the arm.

| Example | Scene | Main behavior | Headless check |
|---------|-------|---------------|----------------|
| `ex_gravity_comp` | arm only | hold home pose with KDL gravity compensation | EE drift |
| `ex_joint_ctrl` | arm only | POSITION motion to a target, then VELOCITY motion back home | joint error at the target, VELOCITY converges before the timeout |
| `ex_table_pick_place` | table + blue cube | joint-impedance pick-place, pushed while carrying the cube | cube placed, push deflection vs stiffness, spring-back, elbow height |
| `ex_table_pour` | table + transparent receiver | scripted pour from a gripper-held bottle into a tabletop vessel | balls in the receiver |
| `ex_rnea_pick_place` | two arms + grippers, table, two cubes, free objects, cameras | dual-arm pick-place with RNEA computed torque | both cubes placed, elbow heights, objects undisturbed, no arm contact |
| `ex_achd_table_slide` | arm without gripper, table | guarded ACHD approach of the wrist flange onto the table, then a slide along +X while pressing 10 N down through ACHD's external-force input | touchdown speed, table contact held, elbow height |
| `ex_achd_pick_place` | table + blue cube | ACHD Cartesian pick/place with 6D TCP regulation and half-arm support wrench | cube placed, elbow drop |
| `ex_admittance_ft` | table + wrist FT + gripper | admittance driven by a named force-torque sensor, RNEA task-space inner loop with a null-space posture | helix and push metrics, elbow height |

---

## ex_gravity_comp

**Scene:** Kinova GEN3 arm (arm only, no gripper).

**What it does:**
- Holds the arm at the home pose using KDL gravity compensation for 15 s (7500 steps).
- `env.on_reset` re-homes the arm, so the UI's reset button puts it back.

**Control law:** `CtrlMode::TORQUE` - KDL gravity compensation.

```
tau[i] = JntToGravity(q)[i]
```

**Headless check:** `EE drift after 7500 steps: X.XXXX mm (limit 0.1000 mm)`; measured 0.0000 mm.

---

## ex_joint_ctrl

**Scene:** Kinova GEN3 arm (arm only, no gripper), with `<joint>_velocity` actuators
(`kv = 500`) added through `RobotSpec::modes`.

**What it does:**
- Motion 1, `CtrlMode::POSITION` (MuJoCo's built-in position servos): moves the arm from home to
  a target pose with a linearly interpolated setpoint over 2 s, then holds it for 1 s.
- Motion 2, `CtrlMode::VELOCITY`: `set_control_mode()` switches without a jump, then a clamped
  proportional velocity command brings the arm back home and stops once every joint is within
  0.01 rad.
- `env.on_reset` re-homes the arm, so the UI's reset button starts motion 1 again.

**Control laws:**

```
POSITION:  jnt_pos_cmd[i] = home[i] + clamp(t / 2.0, 0, 1) * (target[i] - home[i])
VELOCITY:  jnt_vel_cmd[i] = clamp(2.0 * (home[i] - q[i]), -0.6, 0.6)
```

**Headless check:** the POSITION motion ends within 0.01 rad of the target (measured 0.0061) and
the VELOCITY motion converges within 0.01 rad of home before its 5 s timeout (measured 2.66 s).

---

## ex_table_pick_place

**Scene:** Kinova GEN3 + 2F-85 gripper mounted on a table, with a blue cube
on the tabletop.

**What it does:**
- Adds the table as an MJCF-backed `SceneObject` and places the robot base on the tabletop surface.
- Solves IK waypoints for a table pick, transfer, placement, release, and retreat.
- Runs a scripted sequence:
  `HOME -> PICK_ABOVE -> PICK -> CLOSE -> LIFT -> PLACE_ABOVE -> PLACE -> OPEN -> RETREAT -> HOLD`
- While the arm carries the cube to `PLACE_ABOVE`, a scene wrench slot (`bind_scene_wrench`)
  pushes `bracelet_link` with 20 N along +x, across the carry, from 0.8 s to 2.0 s into the
  phase (0.2 s ramps). The arm gives way by about 2 cm, springs back once released, and goes
  on to place the cube. The KDL chain lumps the gripper's mass into its tool
  (`tool_body = "g_base_mount"`), so gravity compensation covers arm and gripper.

**Control law:** `CtrlMode::TORQUE` - joint impedance (PD + KDL gravity).

```
tau[i] = g[i] + Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
Kp = [100, 200, 100, 200, 100, 200, 100]  Nm/rad
Kd = [10,  20,  10,  20,  10,  20,  10 ]  Nm*s/rad
```

With gravity compensated exactly, the quasi-static response to a force `F` at the pushed body's
centre of mass is `dq = K^-1 J_push^T F`, which moves the TCP by `J K^-1 J_push^T F` (`J` the
TCP Jacobian from `ChainJntToJacSolver`, `J_push` the same with its reference point moved to
the centre of mass). The example compares the peak deflection along the push, relative to the
tracking lag just before it and taken at full force, with that prediction.

**Headless check (limit, measured C++ / Python):** cube on the table within 5 mm of the place
spot (1.56 / 1.61 mm); push deflection at least 10 mm (20.57 / 20.57 mm) and within 0.85..1.15
of the prediction (20.30 mm, ratio 1.013); deflection 0.5 s after the push at most 4 mm (1.72
mm); cube centre within 10 mm of the TCP while pushed (6.28 mm, the grasp offset); lowest elbow
(`forearm_link` origin) at least 450 mm above the table (597 mm).

---

## ex_rnea_pick_place

**Scene:** two Kinova GEN3 + 2F-85 arms on the bundled table (1.6 m x 1.2 m top at
`z = 0.7 m`), facing each other: arm 1 at `(-0.70, -0.12)` facing +x, arm 2 at
`(0.70, 0.12)` facing -x with every name prefixed `r2_`. Each arm's cube (`cube`, `r2_cube`)
sits at the same spot in its own base frame, so the arms work in opposite halves of the table.
Five free objects the arms must leave alone (three boxes, two spheres) stand on the table, and
two scene cameras (`overview`, `side`) join each arm's own `wrist` / `r2_wrist` camera.

**What it does:**
- Registers each arm as its own `Robot` (`init_robot_from_mjcf(..., "r2_", &tool)` for the
  second) with its own KDL chain and RNEA solver.
- Solves the IK waypoints once: the targets are in the base frame, the same for both arms.
- Runs the `ex_table_pick_place` sequence on both arms in lockstep, one `update(&env)` per step.

**Control law:** `CtrlMode::TORQUE` - computed torque through `ChainIdSolver_RNE`, per arm.

```
qddot_des[i] = Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
tau          = RNEA(q, dq, qddot_des)
Kp = [100, 200, 100, 200, 100, 200, 100]  1/s^2
Kd = [20,  28,  20,  28,  20,  28,  20 ]  1/s
```

**Headless check (limit, measured C++ / Python):** each cube on the table within 5 mm of its
place spot (1.75 / 1.74 mm, both arms); each lowest elbow at least 450 mm above the table
(598 mm); no free object moved more than 1 mm (0.37 mm); no contact
between the two arms (0); RNEA never failed.

---

## ex_table_pour

**Scene:** Kinova GEN3 + 2F-85 gripper mounted on a table, with a transparent
receiver vessel and 36 small free spheres representing rice or pellets.

**What it does:**
- Attaches a small gripper-sized bottle to the gripper as a real MJCF tool
  attachment, so its mass is included through `ToolFrameSpec::tool_body`.
- Initializes the free spheres inside the attached bottle using the `pour_center` site frame.
- Solves IK waypoints for a pre-pour pose, a pour pose shifted until the tilted outlet sits over
  the receiver, the tilted pour pose and a retreat, then runs
  `HOME -> PRE_POUR -> POUR -> TILT -> POUR_HOLD -> RETREAT -> HOLD`.
- Uses transparent fixed collision walls for the receiving vessel, so particles
  can visibly collect inside it.

**Control law:** `CtrlMode::TORQUE` - joint impedance (PD + KDL gravity), matching
the pick examples.

```
tau[i] = g[i] + Kp[i] * (q_des[i] - q[i]) - Kd[i] * dq[i]
```

**Headless check:** `balls in transparent receiver: N/36`, at least 24; measured 34 (C++) and
35 (Python).

---

## ACHD examples

The ACHD examples use `ChainHdSolver_Vereshchagin` to convert
Cartesian task accelerations into constrained joint accelerations.  For MuJoCo
torque control, the examples command the full inverse-dynamics torque computed
from that ACHD `qddot`:

```
pose error -> beta -> ACHD qddot -> RNEA(q, qdot, qddot) -> jnt_trq_cmd
```

- `ex_achd_table_slide` uses the Gen3 without a gripper: the TCP is the wrist flange (the
  bracelet's `pinch_site`), so the wrist itself presses and slides on the table. It starts with
  the flange pointing down 12 cm above the table (an IK solve from home, held by the position
  servos), then lowers it in a guarded approach:
  all six TCP directions are tracked (nc=6), the reference descends at 0.5 m/s per metre of height
  (at most 0.08 m/s, at least 0.02 m/s), and the first table contact ends it. Then the press
  ramps up over 0.5 s and the TCP slides 0.2 m along +X with linear Z left free (nc=5),
  pressing 10 N down at the TCP through ACHD's external-force input with driver weights 1 (the
  pinned KDL fork's `setDriverWeights()`), gravity as feed-forward. It prints the mean table
  reaction over the second half of the slide, the command and their ratio (not judged,
  measured 0.92). The headless run exits 1 if contact was held for less than 60 % of that
  window (measured 80 %; the contact chatters while sliding), the TCP touched down faster than
  0.05 m/s (measured 0.020; without the approach it fell 0.39 m and hit at 0.87 m/s), or the
  elbow (`forearm_link` origin) went below 0.30 m above the table (measured 0.398). The C++
  example also prints a one-shot nc=6 vs nc=5 comparison.
- `ex_achd_pick_place` runs the pick/place sequence. In every phase it feeds an ACHD-only upward
  support wrench on `half_arm_2_link`: the 6-D TCP task leaves the 7-DOF arm's elbow free, and
  without it the elbow sags to the table. The headless run exits 1 unless the cube rests on the
  table within 5 mm of the place spot (measured 1.0 mm), the elbow stays within 100 mm of its
  reference (measured 53 mm; unsupported it fell about 550 mm), and no solver fails.

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
that to torque. The 6-D TCP task leaves the 7-DOF arm's elbow free, so a joint-space PD
toward the home posture is added in the task's null space with the same WDLS pseudo-inverse,
`qddot = J#(beta - J z) + z` (Siciliano et al., *Robotics: Modelling, Planning and Control*,
2009, Sec. 3.5.1); without it the elbow sags about 445 mm during the run. It:
- Attaches the bundled `ft_sensor.xml` between the wrist pinch site and the gripper
  and registers `wrist_ft` as a named `ForceTorqueSensorSpec` (read as a
  KDL/PyKDL wrench).
- Closes the gripper, lets the wrist load settle, then tares the FT sensor (the
  gripper's ~10 N static load only appears once closed).
- Runs a K=0 mass-damper admittance: an intro helical force (`spiral_force`)
  drives the TCP through a helix, then the FT-measured force takes over so a GUI
  ctrl + right-drag (headless: a scripted push on the tool) is sensed and yielded to; with K=0
  the pose holds on release.
- Draws the commanded (yellow) and measured TCP (green) paths in the viewer.

**Outer admittance law:**

```
f = bias - ft.force             # gravity-tared, deadbanded external force
a = (f - D*v - K*x) / M         # K = 0: no spring, holds on release
v += a*dt;  x += v*dt           # x is the TCP offset from the home pose
target_tcp = nominal_tcp translated by x
```

**Headless check (limit, measured):** helix response > 0.10 m (0.120), helix tracking error
< 6 mm (3.1), settle error < 6 mm (0.0), no false force at the handoff (0), push response
> 0.12 m (0.176), recovery error < 2 mm (0.0), hold drift after the push < 1 mm (0), lowest
elbow (`forearm_link` origin) >= 0.64 m above the table (0.691, its start height; without the
null-space term 0.246).

```bash
./build/src/examples/ex_admittance_ft --headless
python examples/ex_admittance_ft.py            # headless self-check
python examples/ex_admittance_ft.py --gui
```

---

## Recording

`--record [out.mp4]` runs an example headless and writes an H.264 MP4 through `VideoRecorder`
(EGL offscreen rendering + ffmpeg pipe); no window is needed. A recorder that cannot start or
write fails the run.

```bash
./build/src/examples/ex_table_pour --record pour.mp4        # 1080p, default table_pour.mp4
./build/src/examples/ex_achd_pick_place --record pick.mp4   # 720p, default achd_pick_place.mp4
python -m mj_kdl_wrapper.examples.ex_table_pour --record pour.mp4
```

**Requires:** `BUILD_RECORDER=ON` (default) and `ffmpeg` in `PATH`.
