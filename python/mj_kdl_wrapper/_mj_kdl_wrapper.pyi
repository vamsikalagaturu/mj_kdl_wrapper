from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Sequence, Union

import mujoco
import numpy as np
import numpy.typing as npt
import PyKDL as kdl

_JointValues = Union[Sequence[float], npt.NDArray[np.float64]]


class _Enum:
    """A pybind11 enum: members are class attributes, not an enum.Enum."""
    __members__: ClassVar[dict[str, Any]]
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __index__(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


class LogLevel(_Enum):
    """Log threshold: messages at or above it print; NONE prints nothing."""
    INFO: ClassVar[LogLevel]
    WARN: ClassVar[LogLevel]
    ERROR: ClassVar[LogLevel]
    NONE: ClassVar[LogLevel]


class AttachKind(_Enum):
    World: ClassVar[AttachKind]
    Body: ClassVar[AttachKind]
    Site: ClassVar[AttachKind]
    Frame: ClassVar[AttachKind]


class Shape(_Enum):
    Unspecified: ClassVar[Shape]
    BOX: ClassVar[Shape]
    SPHERE: ClassVar[Shape]
    CYLINDER: ClassVar[Shape]


class Condim(_Enum):
    Tangential: ClassVar[Condim]
    Torsional: ClassVar[Condim]
    Rolling: ClassVar[Condim]


class CtrlMode(_Enum):
    POSITION: ClassVar[CtrlMode]
    TORQUE: ClassVar[CtrlMode]
    VELOCITY: ClassVar[CtrlMode]



class CtrlModeSpec:
    """A control mode build_scene adds to a robot, on its own actuator group."""
    mode: CtrlMode
    joints: list[str]
    kv: float
    def __init__(
        self, mode: CtrlMode = CtrlMode.TORQUE, joints: Sequence[str] = (), kv: float = 0.0
    ) -> None: ...


class AttachTarget:
    """Placement target in an accumulated scene spec."""
    kind: AttachKind
    name: str
    def __init__(self, kind: AttachKind = AttachKind.World, name: str = "") -> None: ...


class AttachmentSpec:
    """MJCF attachment applied to a robot root or prior attachment."""
    mjcf_path: str
    attach_to: AttachTarget
    prefix: str
    pos: list[float]
    quat: list[float]
    modes: list[CtrlModeSpec]
    """Control modes for its own joints; [] = native only."""
    contact_exclusions: list[tuple[str, str]]
    def __init__(self) -> None: ...


class RobotSpec:
    """Robot root MJCF plus placement and ordered attachment specs."""
    path: str
    prefix: str
    attach_to: AttachTarget
    pos: list[float]
    quat: list[float]
    attachments: list[AttachmentSpec]
    modes: list[CtrlModeSpec]
    """Extra control modes; default [TORQUE], [] = native only."""
    def __init__(self) -> None: ...


class SceneObject:
    """MJCF asset or primitive object. A primitive needs size, rgba and friction, and mass
    unless fixed; rgba on an asset recolors its geoms."""
    name: str
    mjcf_path: str
    prefix: str
    attach_to: AttachTarget
    shape: Shape
    size: Optional[list[float]]
    pos: list[float]
    quat: list[float]
    rgba: Optional[list[float]]
    fixed: bool
    mass: Optional[float]
    condim: Condim
    friction: Optional[list[float]]
    contype: int
    """Primitive's collision bitmask; MuJoCo's default 1."""
    conaffinity: int
    """Primitive's collision bitmask; MuJoCo's default 1."""
    def __init__(self) -> None: ...


class SiteSpec:
    """A frame marked on a body of the assembled scene."""
    body: str
    name: str
    pos: list[float]
    quat: list[float]
    size: float
    """Marker sphere radius [m]; 0.005 by default."""
    group: int
    """Visibility group of the marker; 4 by default."""
    def __init__(self) -> None: ...


class CameraSpec:
    """Named camera on a body, or in the world when body is empty. pos and fovy are required."""
    name: str
    body: str
    pos: Optional[list[float]]
    quat: list[float]
    fovy: Optional[float]
    def __init__(self) -> None: ...


class SceneSpec:
    """Full scene description. timestep and use_stage are required."""
    robots: list[RobotSpec]
    timestep: Optional[float]
    gravity_z: float
    use_stage: Optional[bool]
    stage: str
    floor_z: float
    objects: list[SceneObject]
    sites: list[SiteSpec]
    cameras: list[CameraSpec]
    def __init__(self) -> None: ...


class ForceTorqueSensorSpec:
    """Logical FT sensor backed by MuJoCo force and torque sensors."""
    name: str
    force_sensor: str
    torque_sensor: str
    frame_site: str
    def __init__(self) -> None: ...


class ToolFrameSpec:
    """Optional tool mass/TCP description for building the robot KDL chain."""
    tool_body: str
    tcp_site: str
    ft_sensors: list[ForceTorqueSensorSpec]
    def __init__(self) -> None: ...


class ResetOptions:
    keyframe: int
    use_keyframe: bool
    def __init__(self) -> None: ...


class ResetInfo:
    @property
    def used_keyframe(self) -> bool: ...
    @property
    def keyframe(self) -> int: ...


class ResetContext:
    """A copy of the reset's options and result, passed to Env.on_reset."""
    @property
    def options(self) -> ResetOptions: ...
    @property
    def info(self) -> ResetInfo: ...


class Robot:
    """Robot registered with an Env (Env.create_robot) and backed by a wrapper-built KDL chain.
    Env.update() reads and commands it; Env.close() closes it."""
    ctrl_mode: CtrlMode
    paused: bool
    @property
    def n_joints(self) -> int: ...
    @property
    def joint_names(self) -> list[str]: ...
    @property
    def joint_limits(self) -> list[tuple[float, float]]: ...
    # Ports read as read-only copies; assign the whole port to write it.
    @property
    def jnt_pos_msr(self) -> npt.NDArray[np.float64]: ...
    @jnt_pos_msr.setter
    def jnt_pos_msr(self, values: _JointValues) -> None: ...
    @property
    def jnt_vel_msr(self) -> npt.NDArray[np.float64]: ...
    @jnt_vel_msr.setter
    def jnt_vel_msr(self, values: _JointValues) -> None: ...
    @property
    def jnt_trq_msr(self) -> npt.NDArray[np.float64]: ...
    @jnt_trq_msr.setter
    def jnt_trq_msr(self, values: _JointValues) -> None: ...
    @property
    def jnt_pos_cmd(self) -> npt.NDArray[np.float64]: ...
    @jnt_pos_cmd.setter
    def jnt_pos_cmd(self, values: _JointValues) -> None: ...
    @property
    def jnt_vel_cmd(self) -> npt.NDArray[np.float64]: ...
    @jnt_vel_cmd.setter
    def jnt_vel_cmd(self, values: _JointValues) -> None: ...
    @property
    def jnt_trq_cmd(self) -> npt.NDArray[np.float64]: ...
    @jnt_trq_cmd.setter
    def jnt_trq_cmd(self, values: _JointValues) -> None: ...
    @property
    def jnt_saturated(self) -> npt.NDArray[np.bool_]: ...
    def set_control_mode(self, mode: CtrlMode) -> None:
        """Switch mode without a jump: seeds the new mode's commands from the current state."""
    def set_joint_pos(self, q: Union[_JointValues, kdl.JntArray]) -> None:
        """Write MuJoCo joint positions; frames read afterwards follow them."""
    def kdl_chain(self) -> kdl.Chain:
        """The robot's own chain as a PyKDL.Chain, valid while the robot lives."""
    @property
    def ft_sensor_names(self) -> list[str]: ...
    def ft_sensor(self, name: str) -> kdl.Wrench:
        """Return the latest measured force-torque sensor value as a PyKDL.Wrench."""
    name: str
    """Set by the caller, before or after init; the viewer shows it."""
    @property
    def has_tcp_frame(self) -> bool: ...
    @property
    def tcp_site(self) -> str: ...
    @property
    def tip_T_tcp(self) -> kdl.Frame: ...
    def joint_force_limits(self, fallback: float = 1e6) -> npt.NDArray[np.float64]:
        """Per-joint force/torque limit of the active mode's actuators; fallback where
        unlimited."""


class Env:
    """The simulation: compiled scene and its robots. The loop is update() (read every robot,
    apply its commands), then step(); Viewer shows it, VideoRecorder records or renders it."""
    on_reset: Optional[Callable[[ResetContext], None]]
    """Called by reset() after every robot is re-seeded; its exception comes out of reset() or
    step()."""
    @property
    def spec(self) -> SceneSpec:
        """A copy of the SceneSpec the Env runs, with objects added or removed since build()."""
    @property
    def model(self) -> mujoco.MjModel:
        """The live mujoco.MjModel; a new object after add_object()/remove_object()."""
    @property
    def data(self) -> mujoco.MjData:
        """The live mujoco.MjData; a new object after add_object()/remove_object()."""
    @staticmethod
    def build(spec: SceneSpec) -> Env: ...
    def close(self) -> None:
        """Free the model; robots become closed."""
    def __enter__(self) -> Env: ...
    def __exit__(self, *args: object) -> None: ...
    def create_robot(
        self,
        base_body: str,
        tip_body: str,
        prefix: str = "",
        tool: Optional[ToolFrameSpec] = None,
    ) -> Robot: ...
    def create_robot_from_chain(
        self,
        chain: kdl.Chain,
        joint_names: Sequence[str],
        prefix: str = "",
        tool: Optional[ToolFrameSpec] = None,
    ) -> Robot:
        """Register a robot driven by chain; joint_names are the MuJoCo joints in chain order."""
    def step(self) -> bool:
        """Advance one timestep. step() never sleeps."""
    def update(self) -> None:
        """Read every robot, then apply its commands."""
    def clear_trace(self) -> None:
        """Remove every overlay geom: segments and arrows."""
    def add_trace_segment(
        self,
        a: Sequence[float],
        b: Sequence[float],
        rgba: Optional[Sequence[float]] = None,
    ) -> None:
        """Add an overlay line segment in world coordinates [m]."""
    def add_overlay_arrow(
        self,
        from_: Sequence[float],
        dir: Sequence[float],
        length: float,
        rgba: Optional[Sequence[float]] = None,
    ) -> None:
        """Add an overlay arrow from a point along dir, length [m] long."""
    def overlay_geoms(self) -> list[mujoco.MjvGeom]:
        """The overlay geoms as mujoco.MjvGeom; Viewer and VideoRecorder draw them themselves."""
    overlay_max_geoms: int
    """Overlay geoms beyond this many are dropped; 8192 by default."""
    overlay_rgba: list[float]
    """Overlay colour [r, g, b, a] when a call gives none; orange by default."""
    def reset(self, options: Optional[ResetOptions] = None) -> ResetInfo:
        """Reset MuJoCo state, re-seed every robot, then call on_reset; what it moves is read
        back."""
    def add_object(self, object: SceneObject) -> None:
        """Rebuild with an added object; robots and scene slots follow the new model."""
    def set_control_mode(self, robot: int, mode: CtrlMode) -> None:
        """Switch robot (its SceneSpec.robots index) to mode, for robots driven without a
        Robot."""
    def remove_object(self, name: str) -> None:
        """Rebuild without the named object; robots and scene slots follow the new model."""
    def body_frame(self, name: str) -> kdl.Frame: ...
    def site_frame(self, name: str) -> kdl.Frame: ...
    def set_body_pose(
        self,
        name: str,
        pos: Sequence[float],
        quat: Optional[Sequence[float]] = None,
    ) -> None:
        """Set a free body pose. quat is xyzw when provided."""
    def save_model_xml(self, path: str) -> None: ...


class Viewer:
    """The C++ MuJoCo Studio viewer on an Env: open() it, then env.step() and pace() as in C++.
    Its pause, reset, perturbation and edits reach the Env inside env.step(), which returns
    False once the window is closed."""
    def __init__(self) -> None: ...
    def open(self, env: Env, title: str = "MuJoCo") -> None:
        """Open the window on env; raises when there is no display."""
    def close(self) -> None: ...
    def __enter__(self) -> Viewer: ...
    def __exit__(self, *args: Any) -> None: ...
    def is_running(self) -> bool: ...
    def pace(self) -> None:
        """Sleep out one timestep at the real-time factor; nothing while closed."""
    def key_pressed(self, key: int) -> bool:
        """True while the key (an ImGuiKey code, as in imgui.h) is held in the window."""
    def capture_key(self, key: int, capture: bool = True) -> None: ...
    def use_camera(self, name: str = "") -> bool: ...
    def set_free_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    realtime_factor: float
    """pace()'s speed: 1.0 real time, 0.5 half speed, 0.0 uncapped."""
    paused: bool
    """Set before open() to open paused; while open, env.step() keeps it equal to the window's
    pause."""
    @property
    def fps(self) -> float: ...
    # Read by open(); later changes reach the next window.
    window_width: int
    window_height: int
    min_panel_width: float
    """Narrowest a side panel may be dragged, in font heights."""
    theme: ViewerTheme
    """Theme unless the settings file names another."""
    ini_path: str
    """Layout and theme file; empty uses Studio's ~/.mujoco.ini."""
    record_path: str
    record_resolution: VideoResolution
    record_fps: int
    framing_frames: int
    """Frames the start camera is re-fitted while the layout settles."""
    speed_window: float
    """The measured speed is sim time over this much wall time [s]."""
    plugin_max_geoms: int
    """Geoms Studio's ScenePlugins may add per frame."""
    joint_limit_margin: float
    """Part of a joint's range counted as near a limit (0..0.5)."""
    joint_limit_highlight: bool
    """Red sphere on a joint near a limit (window only, never recorded)."""
    manipulability_ellipsoid: bool
    """TCP velocity ellipsoid per robot (window only, never recorded)."""
    manipulability_scale: float
    """Drawn size of 1 m/s per rad/s of joint speed [m]."""


class ViewerTheme(_Enum):
    Light: ClassVar[ViewerTheme]
    Dark: ClassVar[ViewerTheme]
    Classic: ClassVar[ViewerTheme]


class VideoResolution(_Enum):
    R360p: ClassVar[VideoResolution]
    R480p: ClassVar[VideoResolution]
    R720p: ClassVar[VideoResolution]
    R1080p: ClassVar[VideoResolution]
    R2K: ClassVar[VideoResolution]
    R4K: ClassVar[VideoResolution]


class VideoRecorder:
    """The C++ offscreen recorder (Filament) on an Env: an MP4 through ffmpeg, or RGB frames."""
    @staticmethod
    def open(
        env: Env, out_path: str, width: int = 1280, height: int = 720, fps: int = 60
    ) -> VideoRecorder: ...
    @staticmethod
    def open_preset(
        env: Env,
        out_path: str,
        resolution: VideoResolution = ...,
        fps: int = 60,
    ) -> VideoRecorder: ...
    @staticmethod
    def open_offscreen(env: Env, width: int, height: int) -> VideoRecorder: ...
    def render_rgb(self) -> npt.NDArray[np.uint8]:
        """The Env's current state as a (height, width, 3) array, top row first."""
    def record_frame(self) -> bool: ...
    def use_camera(self, name: str = "") -> bool: ...
    def set_free_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> VideoRecorder: ...
    def __exit__(self, *args: Any) -> None: ...


__version__: str
__mujoco_version__: str

def set_log_level(level: LogLevel) -> None:
    """Print wrapper messages at level and above; NONE prints nothing."""
def get_log_level() -> LogLevel: ...
