import pytest

import mj_kdl_wrapper as mjk


def _model_path() -> str:
    try:
        return mjk.menagerie.model_path("kinova_gen3", env_var="MJ_KDL_MODEL")
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _asset_path(name: str) -> str:
    try:
        return mjk.menagerie.asset_path(name)
    except RuntimeError:
        mjk.menagerie.fetch_assets()
        return mjk.menagerie.asset_path(name)


def test_ft_sensor_returns_pykdl_wrench():
    kdl = pytest.importorskip("PyKDL")

    ft = mjk.AttachmentSpec()
    ft.mjcf_path = _asset_path("ft_sensor.xml")
    ft.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")

    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = _asset_path("robotiq_2f85/2f85.xml")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "wrist_ft_site")
    gripper.prefix = "g_"

    robot_spec = mjk.RobotSpec()
    robot_spec.path = _model_path()
    robot_spec.attachments = [ft, gripper]

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.robots = [robot_spec]

    scene = mjk.Scene.build(spec)
    try:
        ft_spec = mjk.ForceTorqueSensorSpec()
        ft_spec.name = "wrist_ft"
        ft_spec.frame_site = "wrist_ft_site"

        tool = mjk.ToolFrameSpec()
        tool.tool_body = "g_base"
        tool.tcp_site = "g_pinch"
        tool.ft_sensors = [ft_spec]

        robot = mjk.Robot.from_scene(scene, "base_link", "bracelet_link", tool=tool)
        robot.update()

        assert robot.ft_sensor_names == ["wrist_ft"]
        assert isinstance(robot.ft_sensor("wrist_ft"), kdl.Wrench)
        assert isinstance(robot.ft_sensor_frame("wrist_ft"), kdl.Frame)
    finally:
        scene.close()
