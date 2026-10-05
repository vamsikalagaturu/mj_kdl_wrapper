import pytest

import mj_kdl_wrapper as mjk


def test_ft_sensor_returns_pykdl_wrench():
    kdl = pytest.importorskip("PyKDL")

    ft = mjk.AttachmentSpec()
    ft.mjcf_path = str(mjk.ASSETS_DIR / "ft_sensor.xml")
    ft.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "pinch_site")

    gripper = mjk.AttachmentSpec()
    gripper.mjcf_path = str(mjk.ASSETS_DIR / "robotiq_2f85/2f85.xml")
    gripper.attach_to = mjk.AttachTarget(mjk.AttachKind.Site, "wrist_ft_site")
    gripper.prefix = "g_"

    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
    robot_spec.attachments = [ft, gripper]

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.robots = [robot_spec]

    env = mjk.Env.build(spec)
    try:
        ft_spec = mjk.ForceTorqueSensorSpec()
        ft_spec.name = "wrist_ft"
        ft_spec.frame_site = "wrist_ft_site"

        tool = mjk.ToolFrameSpec()
        tool.tool_body = "g_base_mount"
        tool.tcp_site = "g_pinch"
        tool.ft_sensors = [ft_spec]

        robot = env.create_robot("base_link", "bracelet_link", tool=tool)
        env.update()

        assert robot.ft_sensor_names == ["wrist_ft"]
        assert isinstance(robot.ft_sensor("wrist_ft"), kdl.Wrench)
        assert isinstance(env.site_frame("wrist_ft_site"), kdl.Frame)
    finally:
        env.close()
