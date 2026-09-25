from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import PyKDL as kdl

_JointValues = Union[Sequence[float], npt.NDArray[np.float64]]


class LogLevel(Enum):
    NONE: "LogLevel"
    INFO: "LogLevel"
    WARN: "LogLevel"
    ERROR: "LogLevel"


class AttachKind(Enum):
    World: "AttachKind"
    Body: "AttachKind"
    Site: "AttachKind"
    Frame: "AttachKind"


class Shape(Enum):
    Unspecified: "Shape"
    BOX: "Shape"
    SPHERE: "Shape"
    CYLINDER: "Shape"


class Condim(Enum):
    Tangential: "Condim"
    Torsional: "Condim"
    Rolling: "Condim"


class CtrlMode(Enum):
    POSITION: "CtrlMode"
    TORQUE: "CtrlMode"
    VELOCITY: "CtrlMode"


class CtrlModeSpec:
    """A control mode build_scene adds to a robot, on its own actuator group."""
    mode: CtrlMode
    joints: list[str]
    kv: float
    def __init__(
        self, mode: CtrlMode = CtrlMode.TORQUE, joints: Sequence[str] = (), kv: float = 0.0
    ) -> None: ...


class VideoResolution(Enum):
    R360p: "VideoResolution"
    R480p: "VideoResolution"
    R720p: "VideoResolution"
    R1080p: "VideoResolution"
    R2K: "VideoResolution"
    R4K: "VideoResolution"


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
    """MJCF asset or primitive object. Primitive size, rgba, mass, and friction are required;
    rgba on an asset recolors its geoms."""
    name: str
    mjcf_path: str
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
    def __init__(self) -> None: ...


class SiteSpec:
    """A frame marked on a body of the assembled scene."""
    body: str
    name: str
    pos: list[float]
    quat: list[float]
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
    """Full scene description. timestep, add_floor, and add_skybox are required."""
    robots: list[RobotSpec]
    timestep: Optional[float]
    gravity_z: float
    add_floor: Optional[bool]
    floor_z: float
    add_skybox: Optional[bool]
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
    used_keyframe: bool
    keyframe: int


class ResetContext:
    """Context passed to Env.on_reset after MuJoCo data is reset."""
    @property
    def options(self) -> ResetOptions: ...
    @property
    def info(self) -> ResetInfo: ...


class Robot:
    """Robot registered with an Env (Env.create_robot) and backed by a wrapper-built KDL chain.
    Env.update() reads and commands it; Env.close() closes it."""
    ctrl_mode: CtrlMode
    paused: bool
    n_joints: int
    joint_names: list[str]
    joint_limits: list[tuple[float, float]]
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
    def set_control_mode(self, mode: CtrlMode) -> None: ...
    """Switch mode without a jump: seeds the new mode's commands from the current state."""
    def set_joint_pos(self, q: Union[Sequence[float], kdl.JntArray]) -> None: ...
    """Write MuJoCo joint positions; frames read afterwards follow them."""
    def gravity_torques(self, gravity_z: float = -9.81) -> list[float]: ...
    def kdl_chain(self) -> kdl.Chain: ...
    """Return the wrapper-built chain as a PyKDL.Chain."""
    @property
    def ft_sensor_names(self) -> list[str]: ...
    def ft_sensor(self, name: str) -> kdl.Wrench: ...
    """Return the latest measured force-torque sensor value as a PyKDL.Wrench."""
    def ft_sensor_frame(self, name: str) -> kdl.Frame: ...
    """Return the configured FT sensor frame_site pose as a PyKDL.Frame."""
    def fk_frame(
        self,
        q: Optional[Union[Sequence[float], kdl.JntArray]] = None,
    ) -> kdl.Frame: ...
    """Return FK terminal pose as PyKDL.Frame."""
    @property
    def has_tcp_frame(self) -> bool: ...
    @property
    def tcp_site(self) -> str: ...
    @property
    def tip_T_tcp(self) -> kdl.Frame: ...
    def joint_force_limits(self, fallback: float = 1e6) -> npt.NDArray[np.float64]: ...
    """Per-joint force/torque limit of the active mode's actuators; fallback where unlimited."""


class Viewer:
    """The simulate UI of an Env, opened by Env.open_viewer(); inert while closed."""
    realtime_factor: float
    def is_running(self) -> bool: ...
    def key_pressed(self, key: int) -> bool: ...
    """True while the GLFW key code is held; False headless."""
    def capture_key(self, key: int, capture: bool = True) -> None: ...
    """Claim a GLFW key so the UI does not act on it; capture=False gives it back."""
    def clear_trace(self) -> None: ...
    def add_trace_segment(
        self,
        a: Sequence[float],
        b: Sequence[float],
        rgba: Optional[Sequence[float]] = None,
    ) -> None: ...
    def use_camera(self, name: str = "") -> bool: ...
    def set_free_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None: ...


class VideoRecorder:
    """Offscreen MuJoCo video recorder for an Env."""
    @staticmethod
    def open(
        env: "Env",
        out_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ) -> "VideoRecorder": ...
    @staticmethod
    def open_preset(
        env: "Env",
        out_path: str,
        resolution: VideoResolution = VideoResolution.R720p,
        fps: int = 60,
    ) -> "VideoRecorder": ...
    @staticmethod
    def open_offscreen(env: "Env", width: int, height: int) -> "VideoRecorder": ...
    """Open an offscreen renderer (no video file) for render_rgb()."""
    def render_rgb(self) -> npt.NDArray[np.uint8]: ...
    """The Env's current state as a (height, width, 3) array, top row first."""
    def __enter__(self) -> "VideoRecorder": ...
    def __exit__(self, *args: object) -> None: ...
    def record_frame(self) -> bool: ...
    def use_camera(self, name: str = "") -> bool: ...
    def set_free_camera(
        self,
        distance: float,
        azimuth: float,
        elevation: float,
        lookat: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    def close(self) -> None: ...


class Env:
    """The simulation: compiled scene, its robots, scene slots and viewer. The loop is
    step(), then update() (read every robot and slot, apply their commands), then pace()."""
    spec: SceneSpec
    on_reset: Optional[Callable[[ResetContext], None]]
    @staticmethod
    def build(spec: SceneSpec) -> "Env": ...
    def close(self) -> None: ...
    """Close the viewer and free the model; robots become closed."""
    def __enter__(self) -> "Env": ...
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
    ) -> Robot: ...
    """Register a robot driven by chain; joint_names are the MuJoCo joints in chain order."""
    def step(self) -> bool: ...
    """Advance one timestep; False once the viewer window is closed."""
    def update(self) -> None: ...
    """Read every robot and scene slot, then apply their commands."""
    def pace(self) -> None: ...
    """Sleep out this step's share of wall time; no-op headless. step() never sleeps."""
    def open_viewer(self, title: str = "MuJoCo") -> None: ...
    """Open the simulate UI; step() drives it, close() closes it."""
    @property
    def viewer(self) -> Viewer: ...
    def reset(self, options: Optional[ResetOptions] = None) -> ResetInfo: ...
    """Reset MuJoCo state, then on_reset, then every robot and scene slot."""
    def add_object(self, object: SceneObject) -> None: ...
    """Rebuild with an added object; robots and scene slots follow the new model."""
    def set_control_mode(self, robot: int, mode: CtrlMode) -> None: ...
    """Switch robot (its SceneSpec.robots index) to mode, for robots driven without a Robot."""
    def remove_object(self, name: str) -> None: ...
    """Rebuild without the named object; robots and scene slots follow the new model."""
    def camera_names(self) -> list[str]: ...
    def time(self) -> float: ...
    def timestep(self) -> float: ...
    def body_frame(self, name: str) -> kdl.Frame: ...
    def site_frame(self, name: str) -> kdl.Frame: ...
    def set_body_pose(
        self,
        name: str,
        pos: Sequence[float],
        quat: Optional[Sequence[float]] = None,
    ) -> None: ...
    """Set a free body pose. quat is xyzw when provided."""
    def set_actuator_ctrl(self, name: str, value: float) -> None: ...
    """Command an actuator (or the one driving a joint); takes effect at the next update()."""
    def set_body_wrench(
        self,
        name: str,
        force: Sequence[float],
        torque: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None: ...
    """Push a body with a world-frame wrench; takes effect at the next update()."""
    def actuator_ctrl(self, name: str) -> float: ...
    """The actuator's current ctrl value."""
    def has_actuator(self, name: str) -> bool: ...
    def save_model_xml(self, path: str) -> None: ...
    def save_binary(self, path: str) -> None: ...


__version__: str
__mujoco_version__: str

def set_log_level(level: LogLevel) -> None: ...
def get_log_level() -> LogLevel: ...
def mujoco_version() -> str: ...
def scene_object_site_name(object: SceneObject, site_name: str) -> str: ...
