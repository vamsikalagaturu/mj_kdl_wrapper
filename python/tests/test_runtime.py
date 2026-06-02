from pathlib import Path

import pytest

import mj_kdl_wrapper as mjk


def test_build_scene_and_robot_step():
    model_path = Path("third_party/menagerie/kinova_gen3/gen3.xml")
    if not model_path.exists():
        pytest.skip("MuJoCo Menagerie Kinova model is not available")

    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(model_path)
    spec.robots = [robot_spec]

    scene = mjk.Scene.build(spec)
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
