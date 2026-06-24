# Tests

Tests use GoogleTest and are registered with CTest.  Build and run:

```bash
git clone https://github.com/secorolab/mj_kdl_wrapper.git
cd mj_kdl_wrapper

cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DMJ_KDL_FETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single binary directly
./build/test/test_init
```

All tests self-skip if Menagerie is absent. Fetch it into the user cache with:

```bash
cmake -B build -DMJ_KDL_FETCH_MENAGERIE=ON
```

| Test | What it covers |
|------|----------------|
| `test_init` | `build_scene`, `init_robot_from_mjcf`, cleanup |
| `test_dual_arm` | multi-robot scene, independent KDL chains |
| `test_table_scene` | MJCF table asset, `SceneObject`, runtime add/remove |
| `test_mjcf_load` | arm-only model (nv=7) + arm+gripper model (nq>=13) |
| `test_mjcf_pos_ctrl` | position trajectory tracking |
| `test_mjcf_vel_ctrl` | velocity-style convergence control |
| `test_mjcf_trq_ctrl` | gravity accuracy with gripper mass, impedance drift |
| `test_mjcf_pick` | full pick-and-place with gripper: cube lifted > 0.20 m |
| `urdf_solver_probe` | standalone Kinova URDF ACHD probe plus URDF-vs-MuJoCo RNEA torque comparison |

---

### test_init

**Scene:** single Kinova GEN3 arm from Menagerie MJCF.

- DOF count is 7, joint names resolve correctly.
- `set_joint_pos()` and `mj_forward()` complete without error.
- 100 physics steps complete without error.
- **ResetRestoresDefaultPose** -- `reset()` returns joints to the model's default keyframe pose.
- **ResetSyncsCmdPorts** -- `reset()` re-seeds `jnt_pos_cmd` / `jnt_trq_cmd` from measured state.
- **ResetInvokesOnResetCallback** -- `Env::on_reset` is called exactly once per `reset(Env*)` invocation.
- **ResetWithoutOnResetCallbackIsNoOp** -- `reset(Env*)` with no hook set does not crash.
- **EnvResetInvokesHookAndSyncsRobot** -- `reset(Env*)` invokes the environment hook and syncs registered robot ports/forces.

### test_dual_arm

**Scene:** two Kinova GEN3 arms in a shared `SceneSpec`.

- Both arms initialised with independent `Robot` handles and KDL chains.
- Each arm runs gravity compensation for 500 steps; EE drift < 0.1 mm per arm.

### test_table_scene

**Scene:** Kinova GEN3 arm on a table with box and sphere objects.

- Gravity compensation drift < 1 mm after 500 steps.
- Runtime `scene_add_object` / `scene_remove_object`: model rebuilds cleanly.

### test_mjcf_load

Two fixtures:

- **MjcfLoadTest** (arm from `scene.xml`): `nv==7`, `nbody>=9`, KDL chain has 7
  joints, EE within workspace at home.
- **MjcfGripperTest** (arm + 2F-85): `nq>=13`, `nu>=8`, KDL chain 7 joints,
  EE workspace, gripper driver range `[~0, ~0.8]` rad.

### test_mjcf_pos_ctrl

`CtrlMode::POSITION`.  Linearly interpolates from home to a target pose over 5 s,
settles 1 s.  Max joint error < 0.05 rad.

- **ClampCtrlrange** -- position commands are clamped to the actuator `ctrlrange`; out-of-range setpoints are rejected.
- **QfrcAppliedUnchangedInPositionMode** -- `qfrc_applied` is never written in POSITION mode; torque commands from a prior TORQUE phase are not silently zeroed.

### test_mjcf_vel_ctrl

Velocity-style control implemented by integrating a proportional velocity command
into the position command accepted by the Menagerie actuator model.  The arm
converges from home to the target pose within the configured joint tolerance.

### test_mjcf_trq_ctrl

`CtrlMode::TORQUE`, arm + 2F-85 gripper attached.

- **GravityAccuracy** -- KDL gravity vs `qfrc_bias` at q=0: max error < 5e-2 Nm.
- **ImpedanceDrift** -- PD + gravity for 500 steps: EE drift < 5 mm.
- **TrqMsrReadsQfrcActuator** -- `jnt_trq_msr` reflects `qfrc_actuator` (not `qfrc_bias`) after `update()`.

### test_mjcf_pick

**Scene:** GEN3 (MJCF) + Robotiq 2F-85 + 4 cm cube.

- KDL chain has 7 joints.
- IK error < 2 mm for each waypoint.
- Full pick sequence: cube lifted > 0.20 m.
