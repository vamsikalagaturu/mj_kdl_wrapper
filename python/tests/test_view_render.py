"""The C++ viewer and recorder through Python, without a display: the same checks as
test/test_viewer.cpp and test/test_render.cpp."""

import numpy as np
import pytest

import mj_kdl_wrapper as mjk


def _env() -> mjk.Env:
    spec = mjk.SceneSpec()
    spec.timestep = 0.001
    spec.use_stage = True
    robot_spec = mjk.RobotSpec()
    robot_spec.path = str(mjk.ASSETS_DIR / "kinova_gen3/gen3.xml")
    spec.robots = [robot_spec]
    return mjk.Env.build(spec)


def test_a_viewer_that_cannot_open_raises(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":987")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with _env() as env:
        viewer = mjk.Viewer()
        with pytest.raises(RuntimeError):
            viewer.open(env)
        assert not viewer.is_running()
        viewer.pace()  # a closed viewer does not pace
        assert env.step()


def test_offscreen_frames_show_the_scene_and_follow_the_state():
    with _env() as env:
        with mjk.VideoRecorder.open_offscreen(env, 320, 240) as recorder:
            first = recorder.render_rgb()
            assert first.shape == (240, 320, 3)
            assert first.dtype == np.uint8
            assert first.std() > 0.0
            env.data.qpos[1] += 0.6
            assert env.step()
            moved = recorder.render_rgb()
            assert np.mean(np.any(moved != first, axis=2)) > 0.01


def test_a_recorder_writes_an_mp4(tmp_path):
    path = tmp_path / "out.mp4"
    with _env() as env:
        with mjk.VideoRecorder.open(env, str(path), 320, 240, 30) as recorder:
            for _ in range(10):
                assert env.step()
                assert recorder.record_frame()
    assert path.stat().st_size > 0
