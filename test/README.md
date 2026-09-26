# Tests

Tests use GoogleTest and are registered with CTest.  Build and run:

```bash
git clone https://github.com/vamsikalagaturu/mj_kdl_wrapper.git
cd mj_kdl_wrapper

cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DMJ_KDL_FETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single binary directly
./build/test/test_init
```

Tests on a Menagerie model or a bundled asset self-skip when the user cache lacks it; the
fixture tests (`test/fixtures/`) always run. Fetch the models into the user cache with:

```bash
cmake -B build -DMJ_KDL_FETCH_MENAGERIE=ON
```

The Python examples run headless under pytest (`python/tests/test_examples.py`), each checked
for exit code 0; the two that always open a window are left out.

| Test | What it covers |
|------|----------------|
| `test_init` | `Env`/`Robot` lifecycle: init, `reset` (hook, ports, options), cleanup, adoption, a chain from outside, offscreen rendering |
| `test_dual_arm` | two prefixed arms, independent KDL chains, gravity per arm |
| `test_table_scene` | MJCF table asset, `SceneObject` quat and prefix, runtime add/remove |
| `test_mjcf_load` | chains vs model frames, joint limits, cameras, sites, contact exclusions, attach targets, joint edge cases |
| `test_mjcf_pos_ctrl` | POSITION trajectory tracking, ctrlrange clamping |
| `test_mjcf_vel_ctrl` | VELOCITY convergence on the arm |
| `test_mjcf_trq_ctrl` | gravity with the gripper's mass, impedance drift, `jnt_trq_msr` |
| `test_mjcf_ft_sensor` | named force-torque sensor: resolve, read, reset, reject |
| `test_mjcf_pick` | pick with the gripper: IK waypoints, cube lifted |
| `test_control_modes` | control modes as actuator groups: added actuators, switching, limits, VELOCITY |
| `test_scene_state` | `Env` scene slots, what `step()` leaves current, what `reset()` restores |
| `test_regressions` | fixes from the 0.4.0 audit: rebuilds, re-init, log levels, recorders, screenshot, viewer failure |
| `test_camera_ros` | ROS camera publisher (built only when configured with ROS) |
| `ex_*_headless` | each C++ example runs `--headless` and passes its own self-check |

The opt-in viewer test (`DISABLED_` prefix, see test_scene_state) opens a Simulate window:

```bash
./build/test/test_scene_state --gtest_also_run_disabled_tests --gtest_filter='*Viewer*'
```

---

### test_init

**Scene:** single Kinova GEN3 arm from the bundled `kinova_gen3/gen3.xml`.

- **BasicDOF**, **SimulationAdvance** -- 7 joints, the robot is registered; 100 steps advance
  time by 100 timesteps.
- **ResetRestoresTheKeyframePose** -- `reset()` puts the joints on keyframe 0 and time at 0.
- **ResetOptionsPickTheKeyframeOrTheModelDefault** -- `ResetOptions::keyframe` = 1 resets to the
  second keyframe; `use_keyframe = false` resets to `qpos0`; `ResetInfo` says which.
- **ResetSyncsCmdPorts**, **ResetRestoresEveryPort** -- every port is re-seeded: msr from the
  state, pos cmd = pose, vel/trq cmd 0, saturated 0.
- **ResetInvokesOnResetCallback**, **EnvResetInvokesHookAndSyncsRobot** -- `Env::on_reset` runs
  once per `reset()` with the Env's context; `qfrc_applied` is cleared.
- **ResetWithoutAHookStillReseedsTheRobot** -- no hook set: the robot is still re-seeded.
- **ResetKeepsARequestedControlMode**, **OnResetPrimesCommandsAndMovesAreReadBack** -- reset keeps
  a pending `ctrl_mode`; `on_reset` runs after the re-seed and what it moves is read back.
- **CleanupRobotUnregistersIt** -- after `cleanup(Robot *)` `update()` no longer commands it.
- **ChainFromOutsideDrivesTheSameJoints** -- `init_robot_from_chain()` reads the same joints as the
  derived chain; a joint list of the wrong length is refused.
- **TwoEnvs.StepIndependently** -- two `Env`s in one process keep their own frames and time.
- **EnvSpec.OwnsItsStringsAcrossARebuild** -- `scene_add_object()` rebuilds after the caller's
  path and prefix strings are gone.
- **EnvAdopt.RunsOnTheCallersPairAndNeverFreesIt** -- `Env::adopt` runs the Env on the caller's
  model/data, rebuilds included, and never frees them.
- **SceneFloor.PlacedAtFloorZ** -- the ground plane sits at `SceneSpec::floor_z`.
- **Recorder.OutputPathReachesFfmpegVerbatim** -- a path with `"` and `$(...)` is written as named
  and runs nothing (skips without EGL or ffmpeg).
- **Offscreen.RendersTheSceneIntoABuffer** -- `init_offscreen()` + `render_rgb()` fill the buffer
  (skips without EGL).
- **AFailureSaysWhy** -- a failed call returns a `Status` whose `error` names the cause.
- **SceneSpecRequired.AnUnsetFieldFailsTheBuild** -- unset mass, friction, camera `fovy` or
  timestep fails `build_scene()`.
- **ExamplePaths.StaleMenagerieEnvFallsBackToCache** -- a Menagerie env override pointing
  nowhere falls back to the user cache.

### test_dual_arm

**Scene:** two Kinova GEN3 arms facing each other, the second prefixed `r2_`.

- **PrefixNamesTheWholeChain** -- a prefix names every joint of the chain.
- **KdlGravityMatchesMujocoForBothArms** -- KDL gravity equals `qfrc_bias` at rest to 1e-9 Nm.
- **DualArmDrift** -- 500 steps of gravity compensation; EE drift <= 1 um per arm.

### test_table_scene

**Scene:** Kinova GEN3 arm on a table with box and sphere objects.

- **GravityCompDrift** -- gravity compensation drift <= 1 um after 500 steps.
- **EnvAddRemoveReinitsRobot** -- runtime `scene_add_object` / `scene_remove_object` on the
  `Env`: robots and scene slots follow the rebuilt model; a slot whose body is removed is unbound.
- **SceneObjectTransform.PathBackedObjectAppliesQuat** -- an MJCF-backed object's `[x, y, z, w]`
  `quat` turns its authored site as expected.
- **SceneObjectPrefix.NamesStayAsAuthoredUnlessAPrefixIsSet** -- the same asset twice needs
  distinct prefixes.

### test_mjcf_load

- **MjcfLoadTest** (arm from Menagerie's `scene.xml`): KDL FK equals the MuJoCo frame of
  `bracelet_link`; `joint_limits` follow the model (+-inf for a continuous joint);
  `save_model_xml()` output loads back with the same `nq`/`nbody` and a runtime mass change.
- **MjcfGripperTest** (arm + 2F-85): the TCP chain equals the `g_pinch` site; the driver joint
  range is 0..0.8 rad and the actuator's ctrlrange tops out at 0.82; `bind_scene_joint()` by
  name; `AttachmentSpec::contact_exclusions` adds the pair; attaching to a body with an offset;
  `CameraSpec` on the world and on a body; `SiteSpec` adds a site and leaves an authored one.
- **AttachToFrame** (`fixtures/joint_edge_cases.xml` as an object): a robot stands on an object's
  named `<frame>`.
- **MjcfPathTest** (`fixtures/meshdir/`): a relative model path to an MJCF with a relative
  `meshdir` builds.
- **JointEdgeCaseTest** (`fixtures/joint_edge_cases.xml`): a chain refuses a body with two joints
  and a ball joint on its path; a plain hinge chain builds; a joint slot refuses a ball joint.

### test_mjcf_pos_ctrl

`CtrlMode::POSITION`.  Linearly interpolates from home to a target pose over 1.5 s, settles
0.5 s.  Max joint error < 0.01 rad.

- **ClampCtrlrange** -- position commands are clamped to the actuator `ctrlrange` and flagged in
  `jnt_saturated` (Gen3 joints 2, 4 and 6).

### test_mjcf_vel_ctrl

`CtrlMode::VELOCITY` through the `<joint>_velocity` actuators added by `RobotSpec::modes`: a
clamped proportional velocity command brings the arm to the target within 0.01 rad in under 2.5 s.

### test_mjcf_trq_ctrl

`CtrlMode::TORQUE`, arm + 2F-85 gripper attached.

- **GravityIncludesTheGripperMass** -- at home, KDL gravity with the lumped tool equals
  `qfrc_bias` to 1e-9 Nm, and a chain without the tool is off by more than 1 Nm.
- **ImpedanceDrift** -- PD + gravity for 500 steps: EE drift < 10 um.
- **TrqMsrReadsQfrcActuator** -- nonzero torque commands come back in `jnt_trq_msr` as
  `qfrc_actuator`.

### test_mjcf_ft_sensor

**Scene:** GEN3 + `assets/ft_sensor.xml` + 2F-85, sensor named through `ToolFrameSpec::ft_sensors`.

- **ReadsNamedWrench** -- the logical sensor resolves its `<force>`/`<torque>` sensors and frame
  site; `update()` reads their `sensordata`; at rest it carries the weight below it.
- **ResetReReadsTheWrench** -- after `reset()` the wrench equals the sensors' `sensordata`.
- **RejectsMissingTorqueSensor** -- a sensor naming a missing `<torque>` sensor fails
  `init_robot_from_mjcf()`, and the robot is not registered.

### test_mjcf_pick

**Scene:** GEN3 + Robotiq 2F-85 + 4 cm cube on the floor, driven by the examples' helpers
(`src/examples/common.hpp`).

- **TcpLiesBeyondTheWrist** -- the TCP chain ends at the pinch site, 21.7 cm out of the bracelet.
- **IkStaysOnTheSeedBranch** -- consecutive IK waypoints stay within 1.5 rad of each other.
- **CubeLifted** -- the pick sequence lifts the cube above 0.28 m.

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
- **BindFreeBodyRejectsFixedUnknownAndDuplicate**, **SceneJointTracksQposAndRejectsAFreeJoint**,
  **ASlotAddressSurvivesFurtherBinds** -- slot binding rules; slot pointers stay valid.
- **ApplyClearsAWrenchThatIsNoLongerPushed** -- a wrench set to zero is cleared from
  `xfrc_applied`.
- **DISABLED_ViewerKeepsUserWrenchesWhileAnotherThreadReads** -- opens a Simulate window, so it
  runs only with `--gtest_also_run_disabled_tests`: a user wrench survives 1000 steps while the
  render thread runs and a second thread reads frames.

### test_control_modes

Each mode is an actuator group switched with `opt.disableactuator`.

- **ArmModesTest** (GEN3 `<position>` servos, Menagerie UR5e `<general>` servos): the
  `<joint>_torque` motors are added in a disabled group; POSITION tracks a 0.2 rad ramp within
  0.05 rad; POSITION -> TORQUE -> POSITION holds the pose within 0.01 rad; TORQUE saturates at
  the servo's `forcerange` (GEN3 105 Nm, UR5e 150 Nm); VELOCITY tracks `jnt_vel_cmd` within
  2e-3 rad/s; `update()` and mode switches never write `qfrc_applied`; two arms run different
  modes.
- **GripperModesTest** (GEN3 + 2F-85): the gripper gets no torque actuator and stays in group 0;
  it closes and opens while the arm runs in TORQUE.
- **MotorWheelModesTest** (`fixtures/motor_wheel.xml`): only the listed wheel gets a
  `<velocity>` actuator, the pivot is left alone; VELOCITY tracks 5 rad/s through `env.scene`,
  then TORQUE takes over without a jump; a scene actuator slot flags a clamped command; a
  motor-driven robot starts in TORQUE and `joint_force_limits()` follows the active mode.

### test_regressions

Fixes from the 0.4.0 audit; headless, self-skips without the bundled Gen3.

- **RebuildTest** -- `scene_add_object()` keeps the physics state and the commands; a failed
  `scene_remove_object()` keeps the object order; removing a robot's joint fails and leaves the
  Env; a failed re-init leaves the robot as it was; an `on_reset` set before `init_env()` is
  kept; recorders follow a rebuild and outlive each other; a viewer that cannot open returns an
  error.
- **LogLevel.IsASeverityThreshold** -- each level shows its own and more severe messages.
- **Screenshot.PathReachesFfmpegVerbatimAndAMissingFfmpegIsAFailure** -- the screenshot writer
  takes its path verbatim and reports a missing ffmpeg.

### test_camera_ros

Built only when the wrapper is configured with ROS (`mj_kdl_wrapper::camera_ros` exists).

- **IntrinsicsComeFromTheModelFovy** -- the published `CameraInfo` K matrix follows the camera's
  `fovy` and image size, in the camera's optical frame.
- **NobodyWatchingCostsNothing** -- with no subscriber the publisher never asks for a frame.
