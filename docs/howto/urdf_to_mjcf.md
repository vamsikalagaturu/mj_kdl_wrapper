# Importing a URDF Robot {#page_howto_urdf}

Reference document covering the key issues encountered when trying to use URDF robots with MuJoCo,
and why MJCF (MuJoCo native format, e.g. from MuJoCo Menagerie) is strongly preferred.

---

## Why URDF Loads Break Control

MuJoCo's URDF importer creates joints (hinge/slide) but does **not** create actuators.
After loading a URDF, `model->nu == 0` -- there are no entries in `data->ctrl`.

Consequence:
- `CtrlMode::POSITION` and `CtrlMode::VELOCITY` write to `data->ctrl[kdl_to_mj_ctrl[i]]`,
  which requires `kdl_to_mj_ctrl[i] >= 0` (a valid actuator index).
- With a URDF model, all `kdl_to_mj_ctrl[i] == -1`, so every write is silently skipped.
- Only `CtrlMode::TORQUE` works, because it writes to `data->qfrc_applied` which does
  not depend on actuators.

**Bottom line:** a URDF robot in MuJoCo is unactuated by default. You must add actuators
manually (via `mjSpec`) or use a Menagerie MJCF that already has them.

---

## URDF -> MJCF Conversion Pipeline

If you need to start from a URDF (vendor-supplied), the recommended path is:

### 1. Convert with MuJoCo's own tool

```bash
/opt/mujoco-3.9.0/bin/compile robot.urdf robot.xml
```

This produces a flat MJCF. It is a raw mechanical conversion -- no actuators, no sensors,
no scene lighting. You will need to add those.

### 2. Add actuators

For a 7-DOF position-controlled arm, add inside `<actuator>`:

```xml
<actuator>
  <!-- position actuator per joint -->
  <position name="act_joint1" joint="joint1" kp="2000" kv="100"/>
  <position name="act_joint2" joint="joint2" kp="2000" kv="100"/>
  ...
</actuator>
```

`kp` is the proportional gain (stiffness), `kv` is the derivative gain (damping).
The Kinova GEN3 Menagerie model uses `kp=2000, kv=100` as a starting point.

For velocity control, use `<velocity>` actuators instead:

```xml
<velocity name="act_joint1" joint="joint1" kv="50"/>
```

For direct torque control, use `<motor>`:

```xml
<motor name="act_joint1" joint="joint1" gear="1"/>
```

### 3. Fix inertias

URDF inertias are often zero, near-zero, or poorly conditioned. MuJoCo will warn
or reject them. Add to the `<compiler>` block:

```xml
<compiler balanceinertia="true" inertiafromgeom="false"/>
```

`balanceinertia="true"` automatically fixes inertia tensors that violate the
triangle inequality. `inertiafromgeom="false"` preserves the URDF-specified inertias
rather than recomputing from geometry.

### 4. Fix mesh paths

URDF mesh paths are often relative to the URDF directory. After conversion, add:

```xml
<compiler meshdir="path/to/meshes/"/>
```

If meshes are referenced with `package://` URIs (ROS convention), strip the prefix and
point `meshdir` to the actual mesh directory.

### 5. Remove or handle `<site>` and `<sensor>` blocks

MuJoCo supports these natively but the URDF importer may produce empty or broken
versions. Review and clean up after conversion.

---

## Using mjSpec to Add Actuators Programmatically

If you want to inject actuators at load time (without editing the XML manually), use
the `mjSpec` API:

```cpp
mjSpec *spec = mj_parseXML("robot.xml", nullptr, err, sizeof(err));

/* Iterate joints and add a position actuator for each hinge. */
mjsBody *world = mjs_findBody(spec, "world");
for (mjsJoint *jnt = mjs_firstElement(spec, mjOBJ_JOINT);
     jnt != nullptr;
     jnt = mjs_nextElement(spec, jnt))
{
    if (mjs_getInt(jnt, mjATTR_TYPE) != mjJNT_HINGE) continue;

    mjsActuator *act = mjs_addActuator(spec, nullptr);
    mjs_setString(act, mjATTR_NAME,  /* actuator name */);
    mjs_setString(act, mjATTR_JOINT, mjs_getName(jnt));
    /* set kp, kv, etc. via mjs_setDouble */
}

mjModel *model = mj_compile(spec, nullptr);
mj_deleteSpec(spec);
```

See `mujoco/mujoco.h` for the full `mjs_*` API.

---

## KDL Chain from URDF vs MJCF

`init_robot_from_mjcf()` walks the MuJoCo body tree between `base_body` and `tip_body`
to build the KDL chain. It does not need a URDF.

The old `init_robot_from_urdf()` (now removed) used `kdl_parser` to parse the URDF
separately. This had two problems:
- The KDL chain came from the URDF, not from the compiled MuJoCo model, so inertia
  corrections applied by `balanceinertia` were not reflected in KDL dynamics.
- It required bundling `kdl_parser`, `urdfdom`, and `tinyxml2` as dependencies.

With `init_robot_from_mjcf()`, KDL inertias are read directly from `model->body_inertia`
after MuJoCo compilation, so they always match the simulation.

---

## Checklist When Porting a New Robot from URDF

- [ ] Convert URDF to MJCF with `compile`
- [ ] Add `<compiler balanceinertia="true"/>` if inertia warnings appear
- [ ] Set `meshdir` to point to mesh files
- [ ] Add actuators matching the desired control mode (position / velocity / torque)
- [ ] Verify `model->nu == model->nv` (or however many actuated joints you expect)
- [ ] Set a reasonable `<option timestep="0.002"/>` (MuJoCo default 0.002 s is usually fine)
- [ ] Add a `<keyframe name="home" qpos="..."/>` for the home configuration
- [ ] Identify `base_body` and `tip_body` names for `init_robot_from_mjcf()`
- [ ] Check joint names match what KDL expects (use `model->names` or `mj_id2name`)
- [ ] Run `test_init` equivalent: verify `n_joints`, run 100 steps, check no NaN in qpos

---

## Reference

- MuJoCo model specification: https://mujoco.readthedocs.io/en/stable/XMLreference.html
- MuJoCo Menagerie (reference MJCF models): https://github.com/google-deepmind/mujoco_menagerie
- Kinova GEN3 Menagerie model: `kinova_gen3/gen3.xml` in the user cache populated by
  `mj-kdl-fetch-menagerie` (`~/.cache/mj_kdl_wrapper/menagerie/kinova_gen3/gen3.xml`, or
  `$XDG_CACHE_HOME/mj_kdl_wrapper/...`)
