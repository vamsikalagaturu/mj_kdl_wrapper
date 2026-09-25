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
| `test_init` | `init_env`, `init_robot_from_mjcf`, `reset`, two independent `Env`s |
| `test_dual_arm` | multi-robot scene, independent KDL chains |
| `test_table_scene` | MJCF table asset, `SceneObject`, runtime add/remove |
| `test_mjcf_load` | arm-only model (nv=7) + arm+gripper model (nq>=13), joint edge cases |
| `test_mjcf_pos_ctrl` | position trajectory tracking |
| `test_mjcf_vel_ctrl` | velocity-style convergence control |
| `test_mjcf_trq_ctrl` | gravity accuracy with gripper mass, impedance drift |
| `test_mjcf_pick` | full pick-and-place with gripper: cube lifted > 0.20 m |
| `test_control_modes` | control modes as actuator groups: added actuators, switching, limits |
| `test_scene_state` | `Env` scene slots, what `step()` leaves current, what `reset()` restores |
| `urdf_solver_probe` | standalone Kinova URDF ACHD probe plus URDF-vs-MuJoCo RNEA torque comparison |

---

### test_init

**Scene:** single Kinova GEN3 arm from Menagerie MJCF.

- DOF count is 7; `init_robot_from_mjcf()` registers the robot with the `Env`.
- 100 physics steps advance time.
- **ResetRestoresDefaultPose** -- `reset()` returns joints to the model's default keyframe pose.
- **ResetSyncsCmdPorts** -- `reset()` re-seeds `jnt_pos_cmd` / `jnt_trq_cmd` from measured state.
- **ResetRestoresEveryPort** -- every port is re-seeded: msr from the state, pos cmd = pose,
  vel/trq cmd 0, saturated 0.
- **ResetInvokesOnResetCallback** -- `Env::on_reset` is called exactly once per `reset(Env*)` invocation.
- **ResetWithoutOnResetCallbackIsNoOp** -- `reset(Env*)` with no hook set does not crash.
- **EnvResetInvokesHookAndSyncsRobot** -- `reset(Env*)` invokes the environment hook and syncs registered robot ports/forces.
- **CleanupRobotUnregistersIt** -- `cleanup(Robot*)` removes the robot from its `Env`.
- **TwoEnvs.StepIndependently** -- two `Env`s in one process keep their own frames and time.
- **ResetKeepsARequestedControlMode**, **OnResetPrimesCommandsAndMovesAreReadBack** -- reset keeps
  a pending `ctrl_mode`; `on_reset` runs after the re-seed and what it moves is read back.
- **Recorder.OutputPathReachesFfmpegVerbatim** -- a path with `"` and `$(...)` is written as named
  and runs nothing (skips without EGL or ffmpeg).
- **Recorder.FreeCameraLeavesAFixedCamera** -- `set_free_camera(VideoRecorder*)` switches a fixed
  camera back to a free one.
- **AFailureSaysWhy** -- a failed call returns a `Status` whose `error` names the cause (unknown
  body, object with unset fields, unknown object on remove).
- **SceneSpecRequired.AnUnsetFieldFailsTheBuild** -- unset mass, friction, camera `fovy` or
  timestep fails `build_scene()`.

### test_dual_arm

**Scene:** two Kinova GEN3 arms in a shared `SceneSpec`.

- Both arms initialised with independent `Robot` handles and KDL chains.
- Each arm runs gravity compensation for 500 steps; EE drift < 0.1 mm per arm.

### test_table_scene

**Scene:** Kinova GEN3 arm on a table with box and sphere objects.

- Gravity compensation drift < 1 mm after 500 steps.
- Runtime `scene_add_object` / `scene_remove_object` on the `Env`: robots and scene slots follow
  the rebuilt model; a slot whose body is removed is unbound.

### test_mjcf_load

Two fixtures:

- **MjcfLoadTest** (arm from `scene.xml`): `nv==7`, `nbody>=9`, KDL chain has 7
  joints, EE within workspace at home.
- **MjcfGripperTest** (arm + 2F-85): `nq>=13`, `nu>=8`, KDL chain 7 joints,
  EE workspace, gripper driver range `[~0, ~0.8]` rad.
- **MjcfPathTest** (`fixtures/meshdir/`): a relative model path to an MJCF with a relative
  `meshdir` builds.
- **JointEdgeCaseTest** (`fixtures/joint_edge_cases.xml`): a chain refuses a body with two joints
  and a ball joint on its path; a plain hinge chain builds; the scalar joint getters resolve a
  fixed-tendon actuator to its joint and refuse a spatial-tendon actuator and a ball joint.

### test_mjcf_pos_ctrl

`CtrlMode::POSITION`.  Linearly interpolates from home to a target pose over 5 s,
settles 1 s.  Max joint error < 0.05 rad.

- **ClampCtrlrange** -- position commands are clamped to the actuator `ctrlrange` and flagged in `jnt_saturated`.

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

### test_scene_state

**Scene:** GEN3, a falling cube and 50 fixed filler bodies.

- **FreeBodyPoseAndDerivedFrameAgreeAfterAStep** -- after `step()`, the cube's body frame equals
  its `qpos` pose to 1e-12 (no one-step lag).
- **StepMatchesMjStepBitwise** -- 200 `step()` calls give the same `qpos`/`qvel` as `mj_step`.
- **StepHonoursAQposWrittenBetweenSteps** -- a direct `qpos` write before `step()` is integrated
  as `mj_step` would, and the frame follows it.
- **AFrameFollowsAQposWrittenDirectly** -- `get_body_frame()` sees a direct `qpos` write with no
  step in between.
- **UpdateReadsThenAppliesRobotsAndSlots** -- `update(Env*)` reads robot ports, then writes robot
  torque commands and slot wrenches.
- **ResetRestoresEverySlot** -- `reset()` zeroes wrench slots, sets actuator slot commands to the
  reset `ctrl`, and re-reads joint and free-body slots.
- Slot binding and wrench clearing.
- **DISABLED_ViewerKeepsUserWrenchesWhileAnotherThreadReads** -- opens a Simulate window, so it
  runs only with `--gtest_also_run_disabled_tests`: a user wrench survives 1000 steps while the
  render thread runs and a second thread reads frames.

### test_control_modes

Each mode is an actuator group switched with `opt.disableactuator`.

- **ArmModesTest** (GEN3 `<position>` servos, Menagerie UR5e `<general>` servos): the
  `<joint>_torque` motors are added in a disabled group; POSITION tracks a 0.2 rad ramp within
  0.05 rad; POSITION -> TORQUE -> POSITION holds the pose within 0.01 rad; TORQUE saturates at
  the servo's `forcerange` (GEN3 105 Nm, UR5e 150 Nm); `update()` and mode switches never write
  `qfrc_applied`; two arms run different modes.
- **GripperModesTest** (GEN3 + 2F-85): the gripper gets no torque actuator and stays in group 0;
  it closes and opens while the arm runs in TORQUE.
- **MotorWheelModesTest** (`fixtures/motor_wheel.xml`): only the listed wheel gets a
  `<velocity>` actuator, the pivot is left alone; VELOCITY tracks 5 rad/s through `env.scene`,
  then TORQUE takes over without a jump; a motor-driven robot starts in TORQUE and
  `joint_force_limits()` follows the active mode.
