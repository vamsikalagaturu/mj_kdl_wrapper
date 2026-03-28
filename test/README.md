# Tests

Tests use GoogleTest and are registered with CTest.  Build and run:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTS=ON -DFETCH_MENAGERIE=ON
cmake --build build --parallel $(nproc)

# Run all tests
ctest --test-dir build --output-on-failure

# Run a single binary directly
./build/test/test_init
```

All tests self-skip if `third_party/menagerie/` is absent.  Fetch it with:

```bash
cmake -B build -DFETCH_MENAGERIE=ON
```

---

### test_init

**Scene:** single Kinova GEN3 arm from Menagerie MJCF.

- DOF count is 7, joint names resolve correctly.
- `set_joint_pos()` and `mj_forward()` complete without error.
- 100 physics steps complete without error.

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

### test_mjcf_trq_ctrl

`CtrlMode::TORQUE`, arm + 2F-85 gripper attached.

- **GravityAccuracy** -- KDL gravity vs `qfrc_bias` at q=0: max error < 5e-2 Nm.
- **ImpedanceDrift** -- PD + gravity for 500 steps: EE drift < 5 mm.

### test_mjcf_pick

**Scene:** GEN3 (MJCF) + Robotiq 2F-85 + 4 cm cube.

- KDL chain has 7 joints.
- IK error < 2 mm for each waypoint.
- Full pick sequence: cube lifted > 0.20 m.
