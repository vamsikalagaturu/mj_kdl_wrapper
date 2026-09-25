import pytest

import mj_kdl_wrapper as mjk


def _model_path() -> str:
    try:
        return mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _scene_spec() -> mjk.SceneSpec:
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = _model_path()
    spec.robots = [robot_spec]
    return spec


def _cube(name: str = "cube") -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = name
    obj.shape = mjk.Shape.BOX
    obj.size = [0.02, 0.02, 0.02]
    obj.pos = [0.4, 0.0, 0.8]
    obj.rgba = [1.0, 0.0, 0.0, 1.0]
    obj.mass = 0.1
    obj.friction = [1.0, 0.005, 0.0001]
    return obj


def _skip_without_model() -> None:
    _model_path()


def test_build_env_and_step_robot():
    _skip_without_model()

    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        env.update()
        assert robot.n_joints == 7
        assert len(robot.joint_names) == 7
        assert len(robot.jnt_pos_msr) == 7
        robot.jnt_pos_cmd = [0.0] * robot.n_joints
        assert env.step()
    finally:
        env.close()


def test_pykdl_chain_frame_and_joint_array_interop():
    _skip_without_model()
    kdl = pytest.importorskip("PyKDL")

    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        chain = robot.kdl_chain()
        assert isinstance(chain, kdl.Chain)
        assert chain.getNrOfJoints() == robot.n_joints

        q = kdl.JntArray(robot.n_joints)
        frame = kdl.Frame()
        assert kdl.ChainFkSolverPos_recursive(chain).JntToCart(q, frame) >= 0
        robot.set_joint_pos(q)
    finally:
        env.close()


def test_add_remove_object_keeps_robot_handle_valid():
    _skip_without_model()

    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        env.add_object(_cube())
        env.update()
        assert robot.n_joints == 7
        assert robot.jnt_pos_msr != pytest.approx([0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        assert env.step()
        env.remove_object("cube")
        env.update()
        assert robot.n_joints == 7
        assert len(robot.jnt_pos_msr) == 7
        assert env.step()
    finally:
        env.close()


def test_env_close_invalidates_robot_handles():
    _skip_without_model()

    env = mjk.Env.build(_scene_spec())
    robot = env.create_robot("base_link", "bracelet_link")
    env.close()
    with pytest.raises(RuntimeError, match="env is closed"):
        env.update()
    with pytest.raises(RuntimeError, match="env is closed"):
        env.step()
    with pytest.raises(RuntimeError, match="robot is closed"):
        _ = robot.jnt_pos_msr


def test_control_modes_switch_and_opt_out():
    _skip_without_model()

    assert [m.mode for m in mjk.RobotSpec().modes] == [mjk.CtrlMode.TORQUE]
    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        assert robot.ctrl_mode == mjk.CtrlMode.POSITION
        env.update()
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        assert robot.ctrl_mode == mjk.CtrlMode.TORQUE
        assert robot.jnt_trq_cmd.tolist() == [0.0] * robot.n_joints
        assert len(robot.jnt_vel_cmd) == robot.n_joints
        assert not robot.jnt_saturated.any()
        robot.jnt_trq_cmd = [1000.0] * robot.n_joints
        env.update()
        assert robot.jnt_saturated.all()
        with pytest.raises(RuntimeError, match="control mode"):
            robot.set_control_mode(mjk.CtrlMode.VELOCITY)
        env.set_control_mode(0, mjk.CtrlMode.POSITION)
    finally:
        env.close()

    spec = _scene_spec()
    spec.robots[0].modes = []
    env = mjk.Env.build(spec)
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        with pytest.raises(RuntimeError, match="control mode"):
            robot.set_control_mode(mjk.CtrlMode.TORQUE)
    finally:
        env.close()


def test_set_body_pose_accepts_python_xyzw_quaternion():
    _skip_without_model()
    kdl = pytest.importorskip("PyKDL")

    spec = _scene_spec()
    spec.objects = [_cube()]
    env = mjk.Env.build(spec)
    try:
        quat_xyzw = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        env.set_body_pose("cube", [0.1, 0.2, 0.3], quat_xyzw)
        frame = env.body_frame("cube")
        assert isinstance(frame, kdl.Frame)
        x, y, z, w = frame.M.GetQuaternion()
        assert [x, y, z, w] == pytest.approx(quat_xyzw)
        assert list(frame.p) == pytest.approx([0.1, 0.2, 0.3])
    finally:
        env.close()


def test_reset_restores_commands_and_clears_wrenches():
    _skip_without_model()

    spec = _scene_spec()
    spec.objects = [_cube()]
    env = mjk.Env.build(spec)
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        robot.jnt_pos_cmd = [0.3] * robot.n_joints
        env.data.body("cube").xfrc_applied[2] = 50.0
        for _ in range(50):
            env.update()
            env.step()
        env.reset()
        assert robot.jnt_pos_cmd == pytest.approx(robot.jnt_pos_msr)
        env.update()
        z0 = env.body_frame("cube").p.z()
        for _ in range(50):
            env.update()
            env.step()
        assert env.body_frame("cube").p.z() < z0, "the reset cleared the lifting wrench"
    finally:
        env.close()


def test_ports_are_written_whole_not_in_place():
    _skip_without_model()

    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        with pytest.raises(ValueError, match="read-only"):
            robot.jnt_pos_cmd[0] = 0.5
        held = robot.jnt_pos_cmd
        q = held.copy()
        q[0] = 0.5
        robot.jnt_pos_cmd = q
        assert robot.jnt_pos_cmd[0] == 0.5
        env.reset()
        assert held[0] != 0.5, "a held port stays the copy it was"
    finally:
        env.close()


def test_site_spec_quat_takes_four_values():
    site = mjk.SiteSpec()
    with pytest.raises(TypeError):
        site.quat = [0.0, 0.0, 0.0, 1.0, 0.0]


def test_env_is_a_context_manager_and_saves_its_model(tmp_path):
    _skip_without_model()

    with mjk.Env.build(_scene_spec()) as env:
        robot = env.create_robot("base_link", "bracelet_link")
        env.save_model_xml(str(tmp_path / "scene.xml"))
    assert (tmp_path / "scene.xml").stat().st_size > 0
    with pytest.raises(RuntimeError, match="closed"):
        _ = robot.jnt_pos_msr


def test_robot_from_a_given_chain_matches_the_derived_one():
    _skip_without_model()

    with mjk.Env.build(_scene_spec()) as env:
        derived = env.create_robot("base_link", "bracelet_link")
        env.reset()
        chain = derived.kdl_chain()
        given = env.create_robot_from_chain(chain, derived.joint_names)
        assert given.n_joints == derived.n_joints
        assert given.jnt_pos_msr.tolist() == derived.jnt_pos_msr.tolist()
        assert given.kdl_chain().getNrOfSegments() == chain.getNrOfSegments()


def test_joint_force_limits_follow_the_active_mode():
    _skip_without_model()

    with mjk.Env.build(_scene_spec()) as env:
        robot = env.create_robot("base_link", "bracelet_link")
        robot.set_control_mode(mjk.CtrlMode.TORQUE)
        limits = robot.joint_force_limits()
        assert limits.shape == (robot.n_joints,)
        assert (limits > 0).all() and (limits < 1e6).all()


def test_offscreen_render_returns_an_image():
    _skip_without_model()

    with mjk.Env.build(_scene_spec()) as env:
        try:
            rec = mjk.VideoRecorder.open_offscreen(env, 64, 48)
        except RuntimeError as exc:
            pytest.skip(f"no offscreen rendering: {exc}")
        with rec:
            rgb = rec.render_rgb()
        assert rgb.shape == (48, 64, 3) and rgb.dtype.name == "uint8"
        assert rgb.any()


def test_env_runs_on_real_mujoco_objects():
    _skip_without_model()
    mujoco = pytest.importorskip("mujoco")

    with mjk.Env.build(_scene_spec()) as env:
        robot = env.create_robot("base_link", "bracelet_link")
        assert isinstance(env.model, mujoco.MjModel)
        assert isinstance(env.data, mujoco.MjData)
        mujoco.mj_forward(env.model, env.data)

        env.data.qpos[env.model.joint(robot.joint_names[0]).qposadr[0]] = 0.3
        env.update()
        assert robot.jnt_pos_msr[0] == pytest.approx(0.3)

        old_model, old_data = env.model, env.data
        env.add_object(_cube())
        assert env.model.nbody == old_model.nbody + 1
        assert env.model is not old_model and env.data is not old_data
        assert env.step()
    with pytest.raises(RuntimeError, match="closed"):
        _ = env.model
