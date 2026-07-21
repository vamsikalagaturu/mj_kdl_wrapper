"""Robot-less scenes: options are honored and mesh SceneObjects load.

Regression tests for two fixes in build_scene:
  * a scene with no robots builds (was hard-rejected), and its timestep/gravity
    from SceneSpec are applied;
  * a mesh-backed SceneObject compiles (the asset spec's meshes are loaded before
    it is freed, so the scene compile can resolve them).
"""

from pathlib import Path

import mj_kdl_wrapper as mjk

CABINET = Path(mjk.menagerie.asset_path("cabinet/cabinet.xml"))


def test_bundled_asset_path_resolves():
    assert CABINET.exists()
    assert (CABINET.parent / "cabinet_drawer.stl").exists()


def _cube() -> mjk.SceneObject:
    obj = mjk.SceneObject()
    obj.name = "cube"
    obj.shape = mjk.Shape.BOX
    obj.size = [0.03, 0.03, 0.03]
    obj.pos = [0.0, 0.0, 0.5]
    obj.rgba = [1.0, 0.0, 0.0, 1.0]
    obj.mass = 0.2
    obj.friction = [1.0, 0.005, 0.0001]
    return obj


def test_robotless_scene_applies_timestep():
    spec = mjk.SceneSpec()
    spec.timestep = 0.004
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [_cube()]
    scene = mjk.Scene.build(spec)  # no robots
    try:
        assert scene.timestep() == 0.004
        scene.step()
    finally:
        scene.close()


def test_mesh_scene_object_builds_and_moves():
    obj = mjk.SceneObject()
    obj.name = "cabinet"
    obj.mjcf_path = str(CABINET)
    obj.pos = [0.0, 0.0, 0.0]
    obj.fixed = True
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = True
    spec.objects = [obj]
    scene = mjk.Scene.build(spec)
    try:
        # Meshes compiled -> the drawer's grasp site exists; force pulls the
        # drawer through the cabinet rails, then out and onto the floor.
        closed = scene.site_frame("cabinet_grasp1").p
        scene.set_body_wrench("cabinet_drawer1", [40.0, 0.0, 0.0])
        for _ in range(120):
            scene.step()
        guided = scene.site_frame("cabinet_grasp1").p
        assert guided.x() > closed.x() + 0.05
        assert abs(guided.y()) < 0.03
        for _ in range(400):
            scene.step()
        scene.set_body_wrench("cabinet_drawer1", [0.0, 0.0, 0.0])
        for _ in range(400):
            scene.step()
        opened = scene.site_frame("cabinet_grasp1").p
        assert opened.x() > closed.x() + 0.3
        assert opened.z() < closed.z() - 0.01
    finally:
        scene.close()


def test_mesh_scene_object_applies_euler():
    from PyKDL import Vector

    obj = mjk.SceneObject()
    obj.name = "cabinet"
    obj.mjcf_path = str(CABINET)
    obj.euler = [30.0, 40.0, 50.0]
    obj.fixed = True
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = False
    spec.add_skybox = False
    spec.objects = [obj]
    scene = mjk.Scene.build(spec)
    try:
        y = scene.body_frame("cabinet").M * Vector(0.0, 1.0, 0.0)
        assert abs(y.x() + 0.456825992585671) < 1e-9
        assert abs(y.y() - 0.802872337479472) < 1e-9
        assert abs(y.z() - 0.383022221559489) < 1e-9
    finally:
        scene.close()
