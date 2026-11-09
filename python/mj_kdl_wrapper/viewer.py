"""A MuJoCo Studio window on an Env, from the mujoco package's own viewer (mujoco.studio).

The Env keeps stepping on the caller's thread; the window runs on a thread of its own and draws
what each ``sync()`` publishes, as in mujoco's ``sample/studio/custom_sim_loop.py``::

    from mj_kdl_wrapper.viewer import Viewer

    with Viewer(env, "my scene") as viewer:
        while True:
            env.update()
            ...  # control
            if not viewer.step():  # sync, env.step() unless paused, pace
                break
"""

from __future__ import annotations

import dataclasses
import time

from mujoco.studio import app as studio_app
from mujoco.studio import launch_passive
from mujoco.studio import messages
from mujoco.studio import plugin
from mujoco.studio import sim
from mujoco.studio import viewer_protocol

from . import _mj_kdl_wrapper

# Overlays are copied out of the Env at most this often [s]; the window draws at display rate.
_OVERLAY_PERIOD = 1.0 / 60.0


@dataclasses.dataclass(frozen=True)
class OverlaySnapshot(messages.Snapshot):
    """The Env's overlay geoms (trace segments, arrows), sent from the Env to the window."""

    geoms: tuple


class _EnvControls(plugin.SimPlugin):
    """What the window asks of the Env: its pause, speed and reset buttons."""

    def __init__(self) -> None:
        self.paused = False
        self.speed = 100.0  # percent of real time
        self.reset_requested = False
        self.reset_key: int | None = None

    @messages.handler
    def on_step_control(self, snapshot: messages.StepControlSnapshot) -> None:
        self.paused = snapshot.pause_state != sim.PauseState.UNPAUSED
        self.speed = snapshot.speed

    # The handle resets mjData itself after this handler; the Env re-seeds once sync() returns.
    @messages.handler
    def on_reset(self, event: messages.ResetEvent) -> None:
        self.reset_requested = True
        self.reset_key = event.key


class _OverlayView(plugin.ViewerPlugin):
    """Draws the latest OverlaySnapshot on top of the model."""

    def __init__(self) -> None:
        self._geoms: list = []

    @messages.handler
    def on_overlay(self, snapshot: OverlaySnapshot) -> None:
        self._geoms = list(snapshot.geoms)

    @messages.handler
    def on_update(self, _: messages.UpdateEvent) -> None:
        self.viewer.extra_geoms[:] = self._geoms


class Viewer:
    """A Studio window following an Env; see the module docstring for the loop it expects.

    gfx picks the graphics mode as mujoco.studio does: '' chooses for the session, 'web' streams
    the window to a browser instead.
    """

    def __init__(self, env: _mj_kdl_wrapper.Env, title: str = "MuJoCo", gfx: str = "") -> None:
        self._env = env
        self._controls = _EnvControls()
        self._handle = launch_passive.launch(
            viewer_protocol.ViewerConfig(title=title, gfx=gfx),
            plugins=[studio_app.App(), self._controls, _OverlayView()],
        )
        self._tick = 0.0
        self._overlays_sent = 0.0

    @property
    def paused(self) -> bool:
        """True while the window's pause is on; the caller holds physics then."""
        return self._controls.paused

    @property
    def speed(self) -> float:
        """The window's speed, in percent of real time."""
        return self._controls.speed

    def is_running(self) -> bool:
        """True until the window is closed."""
        return self._handle.is_running()

    def sync(self) -> bool:
        """Exchange state with the window; False once it is closed.

        Applies the window's edits to env.data (perturbation, state edits, physics options) and
        its reset, then publishes the Env's state and overlays to it.
        """
        if not self._handle.is_running():
            return False
        model = self._env.model
        # Control modes own disableactuator; the Physics panel may send back an older one.
        modes = model.opt.disableactuator
        self._handle.sync(model, self._env.data)
        model.opt.disableactuator = modes
        if self._controls.reset_requested:
            self._controls.reset_requested = False
            key = self._controls.reset_key
            options = None
            if key is not None and key >= 0:
                options = _mj_kdl_wrapper.ResetOptions()
                options.keyframe = key
            self._env.reset(options)
        now = time.monotonic()
        if now - self._overlays_sent >= _OVERLAY_PERIOD:
            self._overlays_sent = now
            self._handle.send_to_viewer(OverlaySnapshot(tuple(self._env.overlay_geoms())))
        return self._handle.is_running()

    def step(self) -> bool:
        """Sync with the window, step the Env unless the window is paused, then pace; False once
        the window is closed. A loop written for Env.step() runs unchanged on it."""
        if not self.sync():
            return False
        if not self.paused:
            self._env.step()
        self.pace()
        return True

    def pace(self) -> None:
        """Sleep out this step's share of wall time at the window's speed."""
        period = self._env.model.opt.timestep * 100.0 / max(self._controls.speed, 1e-3)
        now = time.monotonic()
        if self._tick and now < self._tick + period:
            time.sleep(self._tick + period - now)
            self._tick += period
        else:
            # Past the deadline: resynchronise, so a loop back from a stall does not burst.
            self._tick = now

    def close(self) -> None:
        """Close the window."""
        self._handle.close()

    def __enter__(self) -> Viewer:
        return self

    def __exit__(self, *_) -> None:
        self.close()
