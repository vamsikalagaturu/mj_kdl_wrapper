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


def test_build_scene_and_robot_step():
    _skip_without_model()

    scene = mjk.Scene.build(_scene_spec())
    try:
        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
        robot.update()
        assert robot.n_joints == 7
        assert len(robot.joint_names) == 7
        assert len(robot.jnt_pos_msr) == 7
        robot.jnt_pos_cmd = [0.0] * robot.n_joints
        assert robot.step()
    finally:
        scene.close()


def test_pykdl_chain_frame_and_joint_array_interop():
    _skip_without_model()
    kdl = pytest.importorskip("PyKDL")

    scene = mjk.Scene.build(_scene_spec())
    try:
        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
        chain = robot.kdl_chain()
        assert isinstance(chain, kdl.Chain)
        assert chain.getNrOfJoints() == robot.n_joints

        q = kdl.JntArray(robot.n_joints)
        frame = robot.fk_frame(q)
        assert isinstance(frame, kdl.Frame)
        robot.set_joint_pos(q)
    finally:
        scene.close()


def test_scene_add_object_keeps_existing_robot_handle_valid():
    _skip_without_model()

    scene = mjk.Scene.build(_scene_spec())
    try:
        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
        scene.add_object(_cube())
        robot.update()
        assert robot.n_joints == 7
        assert robot.jnt_pos_msr != pytest.approx([0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        assert robot.step()
    finally:
        scene.close()


def test_scene_remove_object_keeps_existing_robot_handle_valid():
    _skip_without_model()

    spec = _scene_spec()
    spec.objects = [_cube()]
    scene = mjk.Scene.build(spec)
    try:
        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
        scene.remove_object("cube")
        robot.update()
        assert robot.n_joints == 7
        assert len(robot.jnt_pos_msr) == 7
        assert robot.step()
    finally:
        scene.close()


def test_env_add_remove_object_rebuilds_robot_joint_maps():
    _skip_without_model()

    env = mjk.Env.build(_scene_spec())
    try:
        robot = env.create_robot("base_link", "bracelet_link")
        env.add_object(_cube())
        robot.update()
        assert robot.n_joints == 7
        assert robot.jnt_pos_msr != pytest.approx([0.4, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])
        assert robot.step()
        env.remove_object("cube")
        robot.update()
        assert robot.n_joints == 7
        assert len(robot.jnt_pos_msr) == 7
        assert robot.step()
    finally:
        env.close()


def test_scene_and_env_close_invalidate_robot_handles():
    _skip_without_model()

    scene = mjk.Scene.build(_scene_spec())
    scene_robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link")
    scene.close()
    with pytest.raises(RuntimeError, match="robot is closed"):
        scene_robot.update()
    with pytest.raises(RuntimeError, match="robot is closed"):
        scene_robot.step()
    with pytest.raises(RuntimeError, match="robot is closed"):
        _ = scene_robot.jnt_pos_msr

    env = mjk.Env.build(_scene_spec())
    env_robot = env.create_robot("base_link", "bracelet_link")
    env.close()
    with pytest.raises(RuntimeError, match="robot is closed"):
        env_robot.update()
    with pytest.raises(RuntimeError, match="robot is closed"):
        env_robot.step()
    with pytest.raises(RuntimeError, match="robot is closed"):
        _ = env_robot.jnt_pos_msr


def test_set_body_pose_accepts_python_xyzw_quaternion():
    _skip_without_model()
    kdl = pytest.importorskip("PyKDL")

    spec = _scene_spec()
    spec.objects = [_cube()]
    scene = mjk.Scene.build(spec)
    try:
        quat_xyzw = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        scene.set_body_pose("cube", [0.1, 0.2, 0.3], quat_xyzw)
        frame = scene.body_frame("cube")
        assert isinstance(frame, kdl.Frame)
        x, y, z, w = frame.M.GetQuaternion()
        assert [x, y, z, w] == pytest.approx(quat_xyzw)
    finally:
        scene.close()
