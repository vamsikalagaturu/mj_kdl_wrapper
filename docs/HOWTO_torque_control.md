# Torque Control and Tool Inertia {#page_howto_torque_control}

This document explains how torque-mode control works in mj-kdl-wrapper, why KDL is
used for all dynamics computations, how gripper (tool) inertia is incorporated into
the KDL chain, and how the multi-robot prefix system works.

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

## Adding Tool Inertia to the KDL Chain

### The problem

`init_robot_from_mjcf()` walks the MuJoCo body tree from `base_body` to `tip_body`
and builds a KDL chain from those bodies' masses and inertias. For a Kinova GEN3 the
tip is `bracelet_link`. When a Robotiq 2F-85 gripper is attached there, its ~0.9 kg
mass is *not* in the chain, so `JntToGravity` underestimates the required joint
torques. The arm will slowly sag under gravity.

### The fix -- `tool_body`

Pass the root body of the attached tool as the `tool_body` argument:

```cpp
mj_kdl::init_robot_from_mjcf(
    &robot, model, data,
    "base_link", "bracelet_link",
    "",          // prefix (empty for single-arm, see Multi-robot below)
    "g_base"     // root body of the gripper subtree
);
```

Internally this:

1. Calls `mj_forward(model, data)` to ensure `xpos`/`xmat` are valid.
2. Collects every body in the subtree rooted at `tool_body` (via a single forward
   pass over the body array -- children always appear after their parent in MuJoCo's
   topological ordering).
3. Computes the *lumped* `KDL::RigidBodyInertia` for the entire subtree in three steps:

   **Step 1 -- total mass and world-frame COM:**
   ```
   m_total = sum(m_i)
   p_com_w = (1/m_total) * sum(m_i * p_i_w)
   ```
   where `p_i_w = data->xpos[b] + R_i * body_ipos[b]` is the body's COM in world
   frame and `R_i` is the body's world-frame rotation matrix from `data->xmat[9*b]`.

   **Step 2 -- combined inertia via parallel-axis theorem (world frame):**
   ```
   I_w = sum( R_i * I_body_i * R_i^T + m_i * [|r_i|^2 I_3 - r_i r_i^T] )
   ```
   where `r_i = p_i_w - p_com_w` is the offset from the lumped COM and
   `I_body_i` is the body's inertia tensor from `model->body_inertia[3*b]` (diagonal
   in body frame).

   **Step 3 -- rotate into tip_body frame:**
   ```
   I_tip = R_tip^T * I_w * R_tip
   p_com_tip = R_tip^T * (p_com_w - p_tip_w)
   ```
   where `R_tip` is the tip body's world-frame rotation matrix.

4. Appends a `KDL::Segment` with `KDL::Joint(KDL::Joint::None)` and the computed
   `KDL::RigidBodyInertia` to the end of the chain.

Because the joint type is `None`, this segment does **not** contribute to
`chain.getNrOfJoints()` -- `n_joints` stays at 7 for the GEN3. The inertia *is*
included in all `KDL::ChainDynParam` computations because those traverse every
segment, moveable or not.

### Result

With `tool_body = "g_base"` the KDL gravity torques match MuJoCo's `qfrc_bias`
to within ~0.05 Nm across the full workspace (verified by `test_mjcf_trq_ctrl::GravityAccuracy`).

---

## Example: Single Arm with Gripper

```cpp
mj_kdl::AttachmentSpec gs;
gs.mjcf_path = "third_party/menagerie/robotiq_2f85/2f85.xml";
gs.attach_to = "bracelet_link";
gs.prefix    = "g_";
gs.pos[2]    = -0.061525;
gs.euler[0]  = 180.0;

mj_kdl::RobotSpec rs;
rs.path = "third_party/menagerie/kinova_gen3/gen3.xml";
rs.attachments.push_back(gs);

mj_kdl::SceneSpec sc;
sc.robots.push_back(rs);
mj_kdl::build_scene(&model, &data, &sc);

mj_kdl::Robot robot;
mj_kdl::init_robot_from_mjcf(&robot, model, data,
    "base_link", "bracelet_link", "", "g_base");

// KDL chain now includes gripper inertia.
KDL::ChainDynParam dyn(robot.chain, KDL::Vector(0, 0, -9.81));

robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

KDL::JntArray q(robot.n_joints), g(robot.n_joints);
while (mj_kdl::tick(&viewer, model, data)) {
    mj_kdl::update(&robot);
    for (unsigned i = 0; i < robot.n_joints; ++i) q(i) = robot.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < robot.n_joints; ++i) robot.jnt_trq_cmd[i] = g(i);
}
```

---

## Multi-robot Scenes and Prefixes

### How prefixes work

`init_robot_from_mjcf()` does two things internally:

1. `build_kdl_from_model()` -- walks from `base_body` to `tip_body`, reading joint
   names from the MuJoCo model (these are the *raw* names in the compiled model,
   e.g. `joint_1`, `r2_joint_1`).

2. `build_index_map()` -- for each joint name stored in step 1, looks up the
   actuator/DOF index in the MuJoCo model. If `prefix` is non-empty, it is
   *prepended* to the joint name before the lookup.

This creates two valid usage patterns:

### Pattern A: Shorthand (arm1 body names, prefix for arm2)

```cpp
// arm1 -- body names have no prefix, joints are joint_1 ... joint_7
mj_kdl::init_robot_from_mjcf(&arm1, model, data,
    "base_link", "bracelet_link", "", "g_base");

// arm2 -- use arm1's body names but tell the index map to look for r2_joint_*
mj_kdl::init_robot_from_mjcf(&arm2, model, data,
    "base_link", "bracelet_link", "r2_", "r2_g_base");
```

`build_kdl_from_model` walks `base_link -> bracelet_link` (the *arm1* bodies, which
also describe arm2's geometry identically). `build_index_map` looks up `r2_joint_1`,
`r2_joint_2`, ... in the compiled model.

### Pattern B: Fully-qualified names (empty prefix)

```cpp
// arm1 -- no prefix either way
mj_kdl::init_robot_from_mjcf(&arm1, model, data,
    "base_link", "bracelet_link", "", "g_base");

// arm2 -- walk arm2's own bodies (r2_base_link -> r2_bracelet_link)
// Joints are already named r2_joint_1 ... r2_joint_7 in the walk result.
// Passing prefix="" tells the index map to look them up as-is.
mj_kdl::init_robot_from_mjcf(&arm2, model, data,
    "r2_base_link", "r2_bracelet_link", "", "r2_g_base");
```

### Common mistake: double prefix

```cpp
// WRONG -- walk produces r2_joint_*, then prefix prepends r2_ again.
// build_index_map looks for r2_r2_joint_1, which does not exist.
mj_kdl::init_robot_from_mjcf(&arm2, model, data,
    "r2_base_link", "r2_bracelet_link", "r2_", "r2_g_base");
// Error: joint 'r2_r2_joint_1' not found in MuJoCo model
```

**Rule:** use `prefix` OR fully-qualified body names, never both at the same time.

---

## Computed-Torque Control (Full RNEA)

### The problem with naive TORQUE mode and position actuators

MuJoCo position actuators (such as those in the Kinova GEN3 Menagerie model) compute:

```
force = kp * (ctrl - pos) - kv * vel
```

The original TORQUE mode implementation set `ctrl = qpos` each step. This zeroed the
stiffness term `kp*(ctrl-pos)` but left the velocity term `-kv*vel` active --
effectively adding an unmodelled damping of `kv` Nm*s/rad per joint. For the GEN3:

| Actuator class | kp   | kv  |
|---------------|------|-----|
| large (j0-j3) | 2000 | 100 |
| small (j4-j6) |  500 |  50 |

With full RNEA computed-torque, the closed-loop dynamics are nominally
`qddot = qddot_des`. The unmodelled `-kv*vel` term breaks this:

```
qddot = qddot_des + (kv / M) * qvel
```

For the GEN3's lightweight wrist links (M ~ 0.01 kg*m^2) and kv = 50, the extra
damping term is `kv/M ~ 5000 rad/s` -- orders of magnitude larger than any Kd gain.
The arm barely moves.

### The fix -- full actuator nulling

`update()` in TORQUE mode now reads `kp` and `kv` from the compiled model and sets:

```
ctrl = qpos + (kv / kp) * qvel
```

This drives the full actuator force to zero:

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

### Using RNEA for full computed-torque control

With the actuator fully nulled, the simulation plant is:

```
M(q) * qddot + C(q, qdot) * qdot + g(q) = tau_applied
```

which is the standard rigid-body dynamics equation. Applying
`tau = M(q)*qddot_des + C(q,qdot)*qdot + g(q)` via `KDL::ChainIdSolver_RNE`
yields exact closed-loop decoupling `qddot = qddot_des` (to within model accuracy).

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
A direct comparison of RNEA gravity torques between the URDF model (parsed via
`kdl_parser`) and the MuJoCo model shows differences below 3 mNm at the home
configuration -- the two models are consistent.

Neither model includes reflected motor/gear inertia (armature). For the real
GEN3 this is the dominant inertia term; for simulation it is irrelevant since
MuJoCo also omits it. If you add `armature` to the MJCF joints, update the KDL
chain inertias accordingly (or the computed-torque feedforward will be inaccurate).

### Comparison with gravity-comp impedance

| Controller | tau formula | Kp units | xy_error (pick-place) |
|---|---|---|---|
| Gravity-comp impedance | `g(q) + Kp*e - Kd*edot` | Nm/rad | ~0.075 m |
| Full RNEA computed-torque | `M*qddot_des + C*qdot + g` | rad/s^2 per rad | ~0.009 m |

The tighter tracking from full RNEA comes from inertia and Coriolis cancellation
during fast transits.

---

## Reference

- `init_robot_from_mjcf()` -- API doc in `mj_kdl_wrapper.hpp`
- `KDL::ChainDynParam` -- orocos_kdl documentation
- `KDL::ChainIdSolver_RNE` -- orocos_kdl documentation
- `test_mjcf_trq_ctrl.cpp` -- gravity accuracy and impedance drift tests
- `src/examples/ex_impedance.cpp` -- single arm + gripper torque control (PD + gravity)
- `src/examples/ex_pick.cpp` -- scripted floor pick and lift
- `src/examples/ex_table_pick_place.cpp` -- tabletop pick and place (gravity-comp)
- `src/examples/ex_rnea_pick_place.cpp` -- tabletop pick and place (full RNEA)
- `src/examples/ex_dual_arm.cpp` -- two arms, each with gripper
- `test/compare_rnea_urdf_mujoco.cpp` -- URDF vs MuJoCo inertia comparison
