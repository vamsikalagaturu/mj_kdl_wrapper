from __future__ import annotations

from enum import Enum
from typing import Optional, Sequence

import PyKDL as kdl


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
    euler: list[float]
    contact_exclusions: list[tuple[str, str]]
    def __init__(self) -> None: ...


class RobotSpec:
    """Robot root MJCF plus placement and ordered attachment specs."""
    path: str
    prefix: str
    attach_to: AttachTarget
    pos: list[float]
    euler: list[float]
    attachments: list[AttachmentSpec]
    def __init__(self) -> None: ...


class SceneObject:
    """MJCF asset or primitive object. Primitive size, rgba, mass, and friction are required."""
    name: str
    mjcf_path: str
    attach_to: AttachTarget
    shape: Shape
    size: Optional[list[float]]
    pos: list[float]
    rgba: Optional[list[float]]
    fixed: bool
    mass: Optional[float]
    condim: Condim
    friction: Optional[list[float]]
    def __init__(self) -> None: ...


class CameraSpec:
    """Named fixed world camera. pos and fovy are required."""
    name: str
    pos: Optional[list[float]]
    euler: list[float]
    fovy: Optional[float]
    def __init__(self) -> None: ...


class SceneSpec:
    """Full scene description. timestep, add_floor, and add_skybox are required."""
    robots: list[RobotSpec]
    timestep: Optional[float]
    gravity_z: float
    add_floor: Optional[bool]
    add_skybox: Optional[bool]
    objects: list[SceneObject]
    cameras: list[CameraSpec]
    def __init__(self) -> None: ...


class ToolFrameSpec:
    """Optional tool mass/TCP description for building the robot KDL chain."""
    tool_body: str
    tcp_site: str
    def __init__(self) -> None: ...


class ResetOptions:
    keyframe: str
    use_keyframe: bool
    def __init__(self) -> None: ...


class ResetInfo:
    used_keyframe: bool
    keyframe: str


class Scene:
    """Owned MuJoCo model/data built from SceneSpec."""
    spec: SceneSpec
    @staticmethod
    def build(spec: SceneSpec) -> "Scene": ...
    """Build and compile a MuJoCo scene."""
    def close(self) -> None: ...
    def save_xml(self, path: str) -> None: ...
    def save_binary(self, path: str) -> None: ...
    def time(self) -> float: ...
    def timestep(self) -> float: ...
    def step(self) -> None: ...
    def step_n(self, n: int) -> None: ...
    def camera_names(self) -> list[str]: ...
    def body_frame(self, name: str) -> kdl.Frame: ...
    """Return a body world pose as PyKDL.Frame."""
    def site_frame(self, name: str) -> kdl.Frame: ...
    """Return a site world pose as PyKDL.Frame."""
    def set_body_pose(
        self,
        name: str,
        pos: Sequence[float],
        quat: Optional[Sequence[float]] = None,
    ) -> None: ...
    def set_actuator_ctrl(self, name: str, value: float) -> None: ...
    def actuator_ctrl(self, name: str) -> float: ...
    def has_actuator(self, name: str) -> bool: ...
    def add_object(self, object: SceneObject) -> None: ...
    def remove_object(self, name: str) -> None: ...


class Robot:
    """Robot handle synchronized with a Scene or Env and backed by a wrapper-built KDL chain."""
    ctrl_mode: CtrlMode
    paused: bool
    n_joints: int
    joint_names: list[str]
    joint_limits: list[tuple[float, float]]
    jnt_pos_msr: list[float]
    jnt_vel_msr: list[float]
    jnt_trq_msr: list[float]
    jnt_pos_cmd: list[float]
    jnt_trq_cmd: list[float]
    @staticmethod
    def from_scene(
        scene: Scene,
        base_body: str,
        tip_body: str,
        prefix: str = "",
        tool: Optional[ToolFrameSpec] = None,
    ) -> "Robot": ...
    def update(self) -> None: ...
    def step(self) -> bool: ...
    def step_n(self, n: int) -> bool: ...
    def set_joint_pos(self, q: Sequence[float], call_forward: bool = True) -> None: ...
    def gravity_torques(self, gravity_z: float = -9.81) -> list[float]: ...
    def kdl_chain(self) -> kdl.Chain: ...
    """Return the wrapper-built chain as a PyKDL.Chain."""
    def fk_frame(self, q: Optional[Sequence[float]] = None) -> kdl.Frame: ...
    """Return FK terminal pose as PyKDL.Frame."""


class SimulateViewer:
    """Wrapper for the custom MuJoCo simulate UI."""
    realtime_factor: float
    @staticmethod
    def open(robot: Robot, title: str = "MuJoCo") -> "SimulateViewer": ...
    def close(self) -> None: ...
    def is_running(self) -> bool: ...
    def step(self) -> bool: ...
    def step_n(self, n: int) -> bool: ...
    def clear_trace(self) -> None: ...
    def add_trace_segment(
        self,
        a: Sequence[float],
        b: Sequence[float],
        rgba: Optional[Sequence[float]] = None,
    ) -> None: ...
    def use_camera(self, name: str = "") -> bool: ...


class VideoRecorder:
    """Offscreen MuJoCo video recorder for a Scene."""
    @staticmethod
    def open(
        scene: Scene,
        out_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 60,
    ) -> "VideoRecorder": ...
    @staticmethod
    def open_preset(
        scene: Scene,
        out_path: str,
        resolution: VideoResolution = VideoResolution.R720p,
        fps: int = 60,
    ) -> "VideoRecorder": ...
    def record_frame(self) -> bool: ...
    def use_camera(self, name: str = "") -> bool: ...
    def close(self) -> None: ...


class Env:
    """Resettable scene environment that keeps registered Robot handles synchronized."""
    spec: SceneSpec
    @staticmethod
    def build(spec: SceneSpec) -> "Env": ...
    def close(self) -> None: ...
    def create_robot(
        self,
        base_body: str,
        tip_body: str,
        prefix: str = "",
        tool: Optional[ToolFrameSpec] = None,
    ) -> Robot: ...
    def reset(self, options: Optional[ResetOptions] = None) -> ResetInfo: ...
    def add_object(self, object: SceneObject) -> None: ...
    def remove_object(self, name: str) -> None: ...
    def camera_names(self) -> list[str]: ...


__version__: str
__mujoco_version__: str

def set_log_level(level: LogLevel) -> None: ...
def get_log_level() -> LogLevel: ...
def mujoco_version() -> str: ...
def scene_object_site_name(object: SceneObject, site_name: str) -> str: ...
