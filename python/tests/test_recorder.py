import pytest

import mj_kdl_wrapper as mjk

CAMERA = (4.0, 90.0, -30.0, (0.0, 0.0, 0.3))


def _env() -> mjk.Env:
    cube = mjk.SceneObject()
    cube.name = "cube"
    cube.shape = mjk.Shape.BOX
    cube.size = [0.1, 0.1, 0.1]
    cube.pos = [0.0, 0.0, 0.1]
    cube.rgba = [1.0, 0.0, 0.0, 1.0]
    cube.mass = 0.2
    cube.friction = [1.0, 0.005, 0.0001]
    top = mjk.CameraSpec()
    top.name = "top"
    top.pos = [0.0, 0.0, 3.0]
    top.fovy = 45.0
    spec = mjk.SceneSpec()
    spec.timestep = 0.002
    spec.add_floor = True
    spec.add_skybox = False
    spec.objects = [cube]
    spec.cameras = [top]
    return mjk.Env.build(spec)


def _open(opener, *args) -> mjk.VideoRecorder:
    try:
        rec = opener(*args)
    except RuntimeError as exc:
        pytest.skip(f"no offscreen rendering: {exc}")
    rec.set_free_camera(*CAMERA)
    return rec


def test_recorder_writes_a_video_and_closes(tmp_path):
    out = tmp_path / "clip.mp4"
    with _env() as env:
        with _open(mjk.VideoRecorder.open, env, str(out), 64, 48, 30) as rec:
            for _ in range(5):
                env.step()
                assert rec.record_frame()
        with pytest.raises(RuntimeError, match="recorder is closed"):
            rec.record_frame()
    assert out.stat().st_size > 0


def test_preset_recorder_renders_its_frame_size(tmp_path):
    with _env() as env:
        path = str(tmp_path / "preset.mp4")
        env.step()
        with _open(mjk.VideoRecorder.open_preset, env, path, mjk.VideoResolution.R360p) as rec:
            rgb = rec.render_rgb()
    assert rgb.shape == (360, 640, 3) and rgb.any()


def test_recorder_switches_cameras():
    with _env() as env:
        env.step()
        with _open(mjk.VideoRecorder.open_offscreen, env, 64, 48) as rec:
            free = rec.render_rgb()
            assert free.any()
            assert rec.use_camera("top")
            assert (rec.render_rgb() != free).any()
            assert not rec.use_camera("missing")
            assert rec.use_camera("")


def test_recorder_follows_a_rebuild():
    cabinet = mjk.SceneObject()
    cabinet.name = "cabinet"
    cabinet.mjcf_path = mjk.menagerie.asset_path("cabinet/cabinet.xml")
    cabinet.pos = [0.0, 0.6, 0.0]
    cabinet.fixed = True
    with _env() as env:
        with _open(mjk.VideoRecorder.open_offscreen, env, 160, 120) as before:
            env.add_object(cabinet)
            env.step()
            with _open(mjk.VideoRecorder.open_offscreen, env, 160, 120) as fresh:
                expected = fresh.render_rgb()
                assert expected.any()
                # A remade GL context may rasterize its first frame a few pixels differently.
                assert (before.render_rgb() != expected).any(axis=2).mean() < 0.001
