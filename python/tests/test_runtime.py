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
        frame = robot.fk_frame(q)
        assert isinstance(frame, kdl.Frame)
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
        assert robot.jnt_trq_cmd == [0.0] * robot.n_joints
        assert len(robot.jnt_vel_cmd) == robot.n_joints
        assert robot.jnt_saturated == [False] * robot.n_joints
        robot.jnt_trq_cmd = [1000.0] * robot.n_joints
        env.update()
        assert robot.jnt_saturated == [True] * robot.n_joints
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


def test_reset_restores_commands_and_slots():
    _skip_without_model()

    spec = _scene_spec()
    spec.objects = [_cube()]
    env = mjk.Env.build(spec)
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        robot.jnt_pos_cmd = [0.3] * robot.n_joints
        env.set_body_wrench("cube", [0.0, 0.0, 50.0])
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
