# Torque Control {#page_howto_torque_control}

This document explains how torque-mode control works in mj-kdl-wrapper, why KDL is
used for all dynamics computations, how MuJoCo's equation of motion relates to the
torques you send, and how ACHD and RNEA fit together.

This document covers torque mode only. Position and velocity modes hand their
setpoint to a MuJoCo servo and do not involve KDL dynamics.

---

## Control Modes and Actuator Groups

Each mode is a set of real MuJoCo actuators, and a mode switch enables one
[actuator group](https://mujoco.readthedocs.io/en/stable/XMLreference.html#option-disableactuator)
and disables the others. This mirrors a drive whose firmware switches between
position and torque loops; nothing is written to `qfrc_applied`.

`build_scene()` keeps the actuators the MJCF declares (a `<position>` servo is
POSITION, a `<motor>` is TORQUE) and adds one actuator per extra mode listed in
`RobotSpec::modes`:

| Added mode | Actuator | Limits |
|---|---|---|
| TORQUE | `<joint>_torque` motor, same gear | ctrlrange = the servo's forcerange |
| VELOCITY | `<joint>_velocity` velocity servo with `kv` | forcerange = the motor's ctrlrange |

`RobotSpec::modes` defaults to `{ { CtrlMode::TORQUE } }`, so every arm gets a
torque group. `{}` keeps only what the MJCF declares, and `CtrlModeSpec::joints`
limits a mode to some joints (e.g. only the wheels of a mobile base).

Robot `r` (its index in `SceneSpec::robots`) owns group `1 + 3*r + mode`
(POSITION = 0, TORQUE = 1, VELOCITY = 2), so groups 1-30 are reserved and a
scene holds at most 10 robots with modes. Group 0 is left to actuators the
wrapper does not manage (a gripper, the pivot of a drive).

```cpp
mj_kdl::RobotSpec arm{ .path = "gen3.xml" };               // POSITION (native) + TORQUE
mj_kdl::RobotSpec drive{                                    // test/fixtures/motor_wheel.xml
  .path  = "motor_wheel.xml",
  .modes = { { .mode = mj_kdl::CtrlMode::VELOCITY, .joints = { "wheel" }, .kv = 2.0 } },
};

mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE);  // or robot.ctrl_mode = TORQUE
mj_kdl::set_control_mode(&env, 1, mj_kdl::CtrlMode::VELOCITY);  // env.scene slot users
```

`set_control_mode()` seeds the new group before enabling it (POSITION from the
current joint position, VELOCITY from the current velocity, TORQUE at zero), so
the switch does not jump. Assigning `ctrl_mode` directly switches on the next
`update(&env)` without the seeding, keeping commands you primed before the switch
(e.g. a gravity torque). In TORQUE, `jnt_trq_cmd / gear` goes to the motor's
ctrl and is clamped to its ctrlrange, so a real drive's torque limit applies.

The viewer's Physics panel shows the group checkboxes and follows programmatic
switches each frame; it lists groups 0-5 only (MuJoCo's `mjNGROUP`).

### Limits and measurements

- `jnt_saturated[i]` is set by `update()` when joint `i`'s command was clamped to its
  actuator's `ctrlrange`, in any mode.
- `joint_force_limits(&robot)` returns each joint's torque limit in the active mode:
  `max(|lo|, |hi|)` of the actuator's `forcerange` times `|gear|` (in TORQUE the smaller of
  that and its `ctrlrange` times `|gear|`), or the `fallback` argument where unlimited. Clamp
  or scale against these, not a fixed number: GEN3's large joints allow 105 Nm, its small
  ones 52 Nm.
- `jnt_trq_msr` is `qfrc_actuator` on the robot's joints. Only the active mode's actuators
  produce force, so it is the drive torque in every mode.

---

## Why KDL for Torque Computations?

In `CtrlMode::TORQUE` only the motors act, so the caller is fully responsible
for computing the correct torques at every step.

A natural candidate for feedforward is `data->qfrc_bias` (MuJoCo's gravity/Coriolis
bias), but this has a critical limitation: it is computed from the *full* model, so its
values change whenever the gripper or any distal mass moves. Reading `qfrc_bias` back
as a feedforward therefore couples your control law to the simulation internals and
makes it impossible to reason about dynamics independently.

The correct approach -- and the only one supported by this wrapper -- is to build a
KDL chain that contains all links from base to tool tip and use
`KDL::ChainDynParam::JntToGravity` (or `JntToMass`, `JntToCoriolis`) for every
torque-mode computation.

**Rule:** if `ctrl_mode == TORQUE`, all feedforward torques come from KDL.

---

## MuJoCo Equation of Motion

MuJoCo's continuous-time equations of motion are:

```
M(q) * v_dot + c(q, v) = tau + J^T * f_constraint
```

which rearranges to the forward dynamics solve:

```
v_dot = M^-1 * (tau - c + J^T * f_constraint)
```

The key quantities stored in `mjData` are:

| Field | Meaning |
|---|---|
| `qfrc_bias` | Bias force **c** = Coriolis + centrifugal + gravity |
| `qfrc_actuator` | Forces from the enabled actuators (the torque motors in TORQUE) |
| `qfrc_passive` | Spring/damper passive forces |
| `qfrc_applied` | User-supplied generalized forces (unused by the wrapper) |

The total applied force is `tau = qfrc_actuator + qfrc_passive + qfrc_applied`.

**Critical sign convention:** `c` is *subtracted* in the forward dynamics.
To hold a static pose (v_dot = 0, f_constraint = 0):

```
tau = c   =>   qfrc_actuator = qfrc_bias
```

This means the controller must *explicitly supply* the gravity and Coriolis
compensation -- MuJoCo does not provide it passively regardless of the actuator
type used.

---

## Computed-Torque Control (Full RNEA)

### Why a disabled servo, not a nulled one

An earlier version kept the position servos enabled in torque mode and nulled
them with `ctrl = qpos + (kv/kp) * qvel`. With the `implicitfast` integrator the
servo's `kv` still enters the implicit velocity derivative, so the arm carried
hidden damping of `kv` (100 on GEN3 joints 1-4) that no controller had asked
for. A disabled group contributes neither force nor derivative, so torque mode
now matches a real torque interface.

### RNEA control loop

With only the torque motors enabled, the MuJoCo plant (from the equation of motion
above) reduces to:

```
M(q) * v_dot = tau - c(q, v)   =>   v_dot = M^-1 * (tau - c)
```

Applying `tau = M(q)*qddot_des + C(q,qdot)*qdot + g(q)` -- which equals `c` when
`qddot_des = 0`, and generally equals `c + M*qddot_des` -- yields:

```
v_dot = M^-1 * (M*qddot_des + c - c) = qddot_des
```

This is exact computed-torque cancellation. `KDL::ChainIdSolver_RNE` computes
`M*qddot_des + C*qdot + g` in O(n) with the Recursive Newton-Euler algorithm:

```cpp
KDL::ChainIdSolver_RNE rnea(robot.chain, KDL::Vector(0, 0, -9.81));
KDL::JntArray q(n), qdot(n), qddot_des(n), torques(n);
KDL::Wrenches f_ext(robot.chain.getNrOfSegments(), KDL::Wrench::Zero());

// in control loop:
for (unsigned i = 0; i < n; ++i) {
    q(i)         = robot.jnt_pos_msr[i];
    qdot(i)      = robot.jnt_vel_msr[i];
    qddot_des(i) = Kp[i] * (q_des(i) - q(i)) - Kd[i] * qdot(i);
}
rnea.CartToJnt(q, qdot, qddot_des, f_ext, torques);
for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = torques(i);
mj_kdl::update(&env);
```

With Kp[i] acting as a squared natural frequency (rad/s^2 per rad) and
Kd[i] ~ 2*sqrt(Kp[i]) for critical damping, the closed loop per joint is a
decoupled second-order system: `e_ddot + Kd*e_dot + Kp*e = 0`.

### Inertia model accuracy

The KDL chain built by `init_robot_from_mjcf` reads `body_inertia` (principal
moments) and `body_iquat` (principal-axis orientation) from the compiled MuJoCo
model and correctly rotates them into the body frame (`I = R * diag(lambda) * R^T`).

Each KDL joint also carries the MJCF joint `armature` as its rotor inertia, which
RNEA and ACHD both use, so the two models agree on reflected motor inertia. The
bundled `assets/kinova_gen3/gen3.xml` sets the GEN3's values (0.5580 kg*m^2 on
joints 1-4, 0.1389 on 5-7). The Menagerie model has none; without the servo's
hidden damping its light wrist went unstable under the torque examples at 2 ms.

---

## ACHD Constraint Torques and the RNEA Bridge

### The Vereshchagin (ACHD) solver

The `KDL::ChainHdSolver_Vereshchagin` solver (ACHD) takes Cartesian
acceleration constraints at the end-effector and computes the constraint force
magnitudes `nu` and the resulting joint accelerations `qdd`.

Its key inputs and outputs:

```
Inputs:
  q, qd            -- current joint state
  alpha (6 x nc)   -- unit constraint forces at the EE (expressed in base frame)
  beta  (nc x 1)   -- desired acceleration energy setpoints per constraint
  ff_tau           -- feedforward joint torques (e.g. null-space damping)
  f_ext            -- external Cartesian wrenches on each segment

Outputs:
  qdd              -- joint accelerations satisfying the constraints (FD solution)
  constraint_tau   -- the constraint's share of each joint's acceleration balance
```

The solver models gravity via a root acceleration `acc_root = (0, 0, -g)` -- the
standard ABA pseudo-force trick that makes gravity appear as an inertial effect.
`getTotalTorque()` returns what each joint feels in that recursion (constraint +
nature + external), not a command; it is near zero for a static constrained hold.

### Which torque realises ACHD's qdd

The robot (or MuJoCo) obeys `M * qdd + c = tau`. The constraint force ACHD solves
for is `alpha * nu` at the end-effector, and the joint torque that produces it is
`J^T * alpha * nu`, so the torque that realises ACHD's `qdd` is

```
tau = ff_tau - J^T * alpha * nu  =  RNEA(q, qd, qdd)
```

`constraint_tau` is not that torque. It is `-S_i^T * A_i * nu`, where `A_i` is the
end-effector constraint carried inward through the articulated-body projections
of the joints distal to `i` (Shakhimardanov 2015, eqs. 3.19-3.20 and 3.34): what
joint `i` feels while the joints beyond it are free. It equals the `J^T` term only
at the last joint. On GEN3 with random states:

| Command | Max error vs. RNEA(qdd) |
|---|---|
| `ff_tau - J^T * alpha * nu` | 4e-14 Nm |
| `ff_tau + constraint_tau` | 4.8 Nm |

Sent as the command, `ff_tau + constraint_tau` still tracks because the Cartesian
PD in `beta` absorbs the error, but less well (helix tracking 1.7 mm against
0.7 mm through RNEA in an FT admittance test), on a real robot as in MuJoCo.

With six constraints on a seven-joint arm, the elbow's null space is left
unconstrained and falls under gravity unless `ff_tau` damps it (the examples use
`ff_tau = -kd * qd`); with zero driver weights the constraint cancels `ff_tau` in
the task directions.

### The two-step pipeline: ACHD -> RNEA

1. **ACHD:** Given Cartesian constraints (alpha, beta), compute `qdd` -- the joint
   accelerations that satisfy the task while minimising acceleration energy (Gauss'
   principle).

2. **RNEA:** Given `qdd` from step 1, compute `tau = M*qdd + C*qd + G`, the torque
   that realises it. `ff_tau - J^T * alpha * nu` (with `getContraintForceMagnitude()`
   and a Jacobian) gives the same torque without RNEA.

```cpp
KDL::Twist root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
KDL::ChainHdSolver_Vereshchagin achd(robot.chain, root_acc, nc);
KDL::ChainIdSolver_RNE rnea(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));

KDL::JntArray qdd(n), ff_tau(n), constraint_tau(n), tau_cmd(n);
KDL::Wrenches f_ext_achd(ns, KDL::Wrench::Zero());
KDL::Wrenches f_ext_rnea_zero(ns, KDL::Wrench::Zero());

// in control loop:
for (unsigned i = 0; i < n; ++i) ff_tau(i) = -kd_null * qd(i);
achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext_achd, ff_tau, constraint_tau);
rnea.CartToJnt(q, qd, qdd, f_ext_rnea_zero, tau_cmd);  // qdd is from ACHD

// update() clamps each torque to its joint's limit and flags it in jnt_saturated.
for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = tau_cmd(i);
mj_kdl::update(&env);
```

For the combined ACHD -> RNEA controller, do not pass ACHD task/support wrenches
into RNEA.  The wrench is part of the ACHD constrained dynamics solve; RNEA is
only the inverse-dynamics bridge that converts the resulting `qdd` into the full
joint torque needed by MuJoCo or the robot.  Keep the RNEA external-wrench vector
zero in this path.

`tau_cmd` from RNEA is what gets sent to the actuators.

### Partial constraints: table slide

ACHD does not require all six TCP axes to be constrained.  The table-slide
example uses this to let the table/contact system handle vertical support while
the controller regulates only the useful task axes:

```
alpha columns:
  0: TCP linear X
  1: TCP linear Y
  2: TCP angular X
  3: TCP angular Y
  4: TCP angular Z

omitted:
  TCP linear Z
```

In `src/examples/ex_achd_table_slide.cpp`, `set_alpha_no_linear_z()` builds this
5-column `alpha`, and `beta` contains only the X/Y position and orientation
errors.  There is no linear-Z beta term, so ACHD is free to choose the vertical
joint acceleration that best satisfies the remaining constraints and the system
dynamics.  This is the right model for a supported slide: wrist/table contact can
carry the vertical reaction, while the controller commands forward motion and
orientation.

The RNEA bridge is unchanged for partial constraints:

```
ACHD(alpha_5d, beta_5d, f_ext_achd) -> qdd
RNEA(q, qd, qdd, zero_f_ext)        -> full joint torque command
```

RNEA does not need to know which Cartesian axes were constrained.  It only sees
the resolved joint acceleration `qdd` from ACHD and computes the full torque
needed to realise that acceleration in the robot dynamics.  The one-shot printout
in `ex_achd_table_slide` compares the `nc=6` and `nc=5` cases so the torque
difference from disabling linear Z is visible.

**Note on real robots:** if the drive firmware adds its own gravity compensation,
subtract `G(q)` from the RNEA torque; the rest of the pipeline is unchanged.

---

## Reference

- `init_robot_from_mjcf()` -- API doc in `mj_kdl_wrapper.hpp`
- `KDL::ChainDynParam` -- orocos_kdl documentation
- `KDL::ChainIdSolver_RNE` -- orocos_kdl documentation
- `KDL::ChainHdSolver_Vereshchagin` -- orocos_kdl documentation
- A. Shakhimardanov, *Composable Robot Motion Stack*, PhD thesis, KU Leuven, 2015, ch. 3
- `test_control_modes.cpp` -- actuator groups, mode switches, torque limits, wheel modes
- `test_mjcf_trq_ctrl.cpp` -- gravity accuracy and impedance drift tests
- `src/examples/ex_impedance.cpp` -- single arm + gripper torque control (PD + gravity)
- `src/examples/ex_pick.cpp` -- scripted floor pick and lift
- `src/examples/ex_table_pick_place.cpp` -- tabletop pick and place (gravity-comp)
- `src/examples/ex_rnea_pick_place.cpp` -- tabletop pick and place (full RNEA)
- `src/examples/ex_achd_table_slide.cpp` -- ACHD-based Cartesian sliding task
- `src/examples/ex_achd_pick_place.cpp` -- ACHD -> RNEA pick and place
- `src/examples/ex_achd_press.cpp` -- ACHD press against the table with a commanded wrench
- `src/examples/ex_admittance_ft.cpp` -- F/T admittance around an RNEA task-space inner loop
- `src/examples/ex_dual_arm.cpp` -- two arms, each with gripper
