# Torque Control {#page_howto_torque_control}

This document explains how torque-mode control works in mj-kdl-wrapper, why KDL is
used for all dynamics computations, how MuJoCo's equation of motion relates to the
torques you send, and how ACHD and RNEA fit together.

This document covers torque mode only. Position and velocity modes write directly
to `data->ctrl` and do not involve KDL dynamics.

---

## Why KDL for Torque Computations?

When `ctrl_mode = CtrlMode::TORQUE`, joint commands are written directly to
`data->qfrc_applied` -- bypassing MuJoCo actuators entirely. This means the
caller is fully responsible for computing the correct torques at every step.

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
| `qfrc_applied` | User-supplied generalized forces (what `jnt_trq_cmd` writes to) |
| `qfrc_actuator` | Forces from model actuators through their moment arms |
| `qfrc_passive` | Spring/damper passive forces |

The total applied force is `tau = qfrc_applied + qfrc_actuator + qfrc_passive`.

**Critical sign convention:** `c` is *subtracted* in the forward dynamics.
To hold a static pose (v_dot = 0, f_constraint = 0):

```
tau = c   =>   qfrc_applied = qfrc_bias
```

This means the controller must *explicitly supply* the gravity and Coriolis
compensation -- MuJoCo does not provide it passively regardless of the actuator
type used.

---

## Computed-Torque Control (Full RNEA)

### Position actuator nulling

MuJoCo position actuators (such as those in the Kinova GEN3 Menagerie model) compute:

```
force = kp * (ctrl - pos) - kv * vel
```

For the GEN3:

| Actuator class | kp   | kv  |
|---------------|------|-----|
| large (j0-j3) | 2000 | 100 |
| small (j4-j6) |  500 |  50 |

**Case 1 -- partial nulling (`ctrl = qpos`):**
Setting `ctrl = qpos` zeroes the stiffness term `kp*(ctrl-pos)` but leaves the
velocity term `-kv*vel` active. With full RNEA computed-torque the desired closed-loop
is `qddot = qddot_des`, but the residual damping breaks this:

```
qddot = qddot_des + (kv / M) * qvel
```

For the GEN3's lightweight wrist links (M ~ 0.01 kg*m^2) and kv = 50, the extra
term is `kv/M ~ 5000 rad/s` -- orders of magnitude larger than any Kd gain.
The arm barely moves.

**Case 2 -- full nulling (`ctrl = qpos + (kv/kp) * qvel`):**
Reading `kp` and `kv` from the compiled model and setting:

```
ctrl = qpos + (kv / kp) * qvel
```

drives the complete actuator force to zero:

```
kp * (ctrl - pos) - kv * vel
= kp * ((pos + (kv/kp)*vel) - pos) - kv * vel
= kv * vel - kv * vel
= 0
```

`qfrc_applied` is then the sole torque source, exactly matching a real robot's
torque interface. The relevant code is in `src/mj_kdl_wrapper.cpp`:

```cpp
case CtrlMode::TORQUE:
    if (ctrl_id >= 0) {
        const double kp =  m->actuator_gainprm[ctrl_id * mjNGAIN + 0];
        const double kv = -m->actuator_biasprm[ctrl_id * mjNBIAS + 2];
        d->ctrl[ctrl_id] = d->qpos[qpos_id] + (kp > 0.0 ? kv/kp : 0.0) * d->qvel[dof_id];
    }
    d->qfrc_applied[dof_id] = r->jnt_trq_cmd[i];
    break;
```

This works for any actuator type that stores its bias as `biasprm[2] = -kv` (MuJoCo's
`mjBIAS_AFFINE` / biastype=1), which covers all standard `<position>` actuators.

### RNEA control loop

With the actuator fully nulled, the MuJoCo plant (from the equation of motion above)
reduces to:

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
mj_kdl::update(&robot);
```

With Kp[i] acting as a squared natural frequency (rad/s^2 per rad) and
Kd[i] ~ 2*sqrt(Kp[i]) for critical damping, the closed loop per joint is a
decoupled second-order system: `e_ddot + Kd*e_dot + Kp*e = 0`.

### Inertia model accuracy

The KDL chain built by `init_robot_from_mjcf` reads `body_inertia` (principal
moments) and `body_iquat` (principal-axis orientation) from the compiled MuJoCo
model and correctly rotates them into the body frame (`I = R * diag(lambda) * R^T`).

Neither model includes reflected motor/gear inertia (armature). For the real
GEN3 this is the dominant inertia term; for simulation it is irrelevant since
MuJoCo also omits it. If you add `armature` to the MJCF joints, update the KDL
chain inertias accordingly (or the computed-torque feedforward will be inaccurate).

---

## ACHD Constraint Torques and the RNEA Bridge

### The Vereshchagin (ACHD) solver

The `KDL::ChainHdSolver_Vereshchagin_Fixed_Joint` solver (ACHD) takes Cartesian
acceleration constraints at the end-effector and computes both a forward-dynamics
result (joint accelerations `qdd`) and the joint torques that enforce those
constraints (`constraint_tau`).

Its key inputs and outputs:

```
Inputs:
  q, qd            -- current joint state
  alpha (6 x nc)   -- unit constraint forces at the EE (expressed in base frame)
  beta  (nc x 1)   -- desired acceleration energy setpoints per constraint
  ff_tau           -- feedforward joint torques (typically zero)
  f_ext            -- external Cartesian wrenches on each segment

Outputs:
  qdd              -- joint accelerations satisfying the constraints (FD solution)
  constraint_tau   -- joint torques arising from the Lagrange multiplier forces
```

The solver models gravity via a root acceleration `acc_root = (0, 0, -g)` -- the
standard ABA pseudo-force trick that makes gravity appear as an inertial effect.
This means `constraint_tau` is computed in an ABA frame where gravity is already
absorbed, and `total_tau` (obtainable via `getTotalTorque()`) is near zero for a
static constrained hold:

```
total_tau = ff_tau + bias(gravity+Coriolis) + parent_inertial + constraint_tau ~= 0
```

The TCP's spatial acceleration from `getTransformedLinkAcceleration()` correctly
shows `xdd_lin = (0, 0, g)` at the end-effector -- confirming the solver holds
the TCP stationary in the inertial frame (`xdd_inertial = xdd_ABA + acc_root = 0`).

### Why constraint_tau alone is insufficient for MuJoCo

From MuJoCo's equation of motion:

```
v_dot = M^-1 * (tau - c)
```

To produce the desired `qdd` from ACHD, we need:

```
tau = M * qdd + c
```

ACHD's `constraint_tau` is defined as:

```
constraint_tau = total_tau - bias - parent_inertial
               = (M * qdd) - bias_ACHD
```

Since `bias_ACHD ~= c = qfrc_bias`, this gives:

```
constraint_tau ~= M * qdd - c
```

But MuJoCo requires `tau = M * qdd + c`. Sending `constraint_tau` as `qfrc_applied`
therefore produces:

```
v_dot = M^-1 * (constraint_tau - c)
      = M^-1 * (M*qdd - c - c)
      = qdd - 2 * M^-1 * c
```

The `2 * M^-1 * c` residual is an uncompensated double-gravity term that causes
the arm to drift. Measured on the Kinova GEN3 home-hold task:

| Command sent to `qfrc_applied` | TCP position error (2500 steps) |
|---|---|
| `constraint_tau` only | ~31 mm |
| RNEA (`M*qdd + C*qd + G`) | ~7 mm |

Adding an integral term (Ki) to the Cartesian PD reduces the steady-state
offset but does not fix the underlying mismatch -- it just accumulates enough
integral action to fight the double-gravity residual.

### The correct two-step pipeline: ACHD -> RNEA

The proper way to use ACHD output in MuJoCo torque control is:

1. **ACHD:** Given Cartesian constraints (alpha, beta), compute `qdd` -- the joint
   accelerations that satisfy the task while minimising acceleration energy (Gauss'
   principle). The `xdd_tcp = (0, 0, g)` output confirms the TCP is correctly
   held against gravity in the ABA frame.

2. **RNEA:** Given `qdd` from step 1, compute `tau = M*qdd + C*qd + G`. This is
   exactly `c + M*qdd` -- what MuJoCo requires as `qfrc_applied` to realise the
   desired joint accelerations.

```cpp
KDL::Twist root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd(robot.chain, root_acc, nc);
KDL::ChainIdSolver_RNE rnea(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));

KDL::JntArray qdd(n), ff_tau(n), constraint_tau(n), tau_cmd(n);
KDL::Wrenches f_ext_achd(ns, KDL::Wrench::Zero());
KDL::Wrenches f_ext_rnea_zero(ns, KDL::Wrench::Zero());

// in control loop:
KDL::SetToZero(ff_tau);
achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext_achd, ff_tau, constraint_tau);
rnea.CartToJnt(q, qd, qdd, f_ext_rnea_zero, tau_cmd);  // qdd is from ACHD

for (unsigned i = 0; i < n; ++i)
    robot.jnt_trq_cmd[i] = clamp(tau_cmd(i), -tau_max, tau_max);
mj_kdl::update(&robot);
```

For the combined ACHD -> RNEA controller, do not pass ACHD task/support wrenches
into RNEA.  The wrench is part of the ACHD constrained dynamics solve; RNEA is
only the inverse-dynamics bridge that converts the resulting `qdd` into the full
joint torque needed by MuJoCo or the robot.  Keep the RNEA external-wrench vector
zero in this path.

The `constraint_tau` output is still useful for diagnostics -- it isolates the
task-space contribution -- but `tau_cmd` from RNEA is what gets sent to the
actuators.

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

**Note on real robots:** On a real robot with an inner gravity-compensation loop
(common on torque-controlled manipulators), `constraint_tau` *would* be the correct
outer-loop command because the inner loop already provides `c`. MuJoCo does not
provide such a layer -- it simulates raw physics -- so the full RNEA is necessary.

---

## Reference

- `init_robot_from_mjcf()` -- API doc in `mj_kdl_wrapper.hpp`
- `KDL::ChainDynParam` -- orocos_kdl documentation
- `KDL::ChainIdSolver_RNE` -- orocos_kdl documentation
- `KDL::ChainHdSolver_Vereshchagin_Fixed_Joint` -- orocos_kdl documentation
- `test_mjcf_trq_ctrl.cpp` -- gravity accuracy and impedance drift tests
- `src/examples/ex_impedance.cpp` -- single arm + gripper torque control (PD + gravity)
- `src/examples/ex_pick.cpp` -- scripted floor pick and lift
- `src/examples/ex_table_pick_place.cpp` -- tabletop pick and place (gravity-comp)
- `src/examples/ex_rnea_pick_place.cpp` -- tabletop pick and place (full RNEA)
- `src/examples/ex_achd_table_slide.cpp` -- ACHD-based Cartesian sliding task
- `src/examples/ex_dual_arm.cpp` -- two arms, each with gripper
