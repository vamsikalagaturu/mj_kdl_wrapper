/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

namespace mj_kdl {

/**
 * @ingroup grp_logging
 * Log verbosity level.  Each level includes all levels below it:
 *   NONE   - nothing printed.
 *   INFO   - informational messages only (scene/chain construction progress).
 *   WARN   - INFO + recoverable warnings (e.g. fallback to headless mode).
 *   ERROR  - all messages, including errors that cause functions to fail.  Default.
 */
enum class LogLevel { NONE = 0, INFO = 1, WARN = 2, ERROR = 3 };

/** @ingroup grp_logging
 *  Library-wide log verbosity (inline so one shared instance across all TUs). */
inline LogLevel g_log_level = LogLevel::ERROR;

/** @ingroup grp_logging
 *  Set the library-wide log verbosity. */
inline void set_log_level(LogLevel level) { g_log_level = level; }
/** @ingroup grp_logging
 *  Get the library-wide log verbosity. */
inline LogLevel get_log_level() { return g_log_level; }

} // namespace mj_kdl

/* @ingroup grp_logging
 * Internal logging macros, exposed so wrapper users (examples, tests, and
 * downstream code) can emit messages through the same stream/level filter.
 * MJ_LOG_ is the primitive; LOG_INFO/LOG_WARN/LOG_ERROR are the entry points.
 * `expr` may use << to build the message: LOG_INFO("count=" << n).
 *
 * Defined at file scope (not inside the mj_kdl namespace) because macros are
 * not namespaced; the MJ_ prefix avoids collisions.
 */
#define MJ_FILENAME_ (::strrchr(__FILE__, '/') ? ::strrchr(__FILE__, '/') + 1 : __FILE__)

#define MJ_LOG_(lvl_enum, color, label, expr)                         \
    do {                                                              \
        if (::mj_kdl::g_log_level >= ::mj_kdl::LogLevel::lvl_enum) {  \
            std::ostringstream _mj_oss;                               \
            _mj_oss << expr; /* NOLINT(bugprone-macro-parentheses) */ \
            std::fprintf(                                             \
              stderr,                                                 \
              color "[mj_kdl " label "] %s:%d (%s): %s\033[0m\n",     \
              MJ_FILENAME_,                                           \
              __LINE__,                                               \
              __func__,                                               \
              _mj_oss.str().c_str()                                   \
            );                                                        \
        }                                                             \
    } while (0)

#define LOG_INFO(expr) MJ_LOG_(INFO, "", "INFO ", expr)
#define LOG_WARN(expr) MJ_LOG_(WARN, "\033[33m", "WARN ", expr)
#define LOG_ERROR(expr) MJ_LOG_(ERROR, "\033[31m", "ERROR", expr)

namespace mj_kdl {

/**
 * @ingroup grp_types
 * Kind of element an AttachTarget references in the accumulated scene spec.
 * World selects the worldbody (the default); Body, Site, and Frame each look
 * up a named element of the corresponding type.
 */
enum class AttachKind { World, Body, Site, Frame };

/**
 * @ingroup grp_types
 * Where to attach a robot, object, or attachment in the accumulated scene spec.
 * Tagged so exactly one alternative is encoded; defaulting to World keeps
 * callers that omit attach_to anchored to the worldbody.
 * For Site, the site's own pos/quat becomes the placement frame and the
 * accompanying pos/quat are an additional offset on top of it (matches MJCF).
 */
struct AttachTarget
{
    AttachKind  kind = AttachKind::World;
    const char *name = nullptr; // ignored when kind == World
};

/**
 * @ingroup grp_types
 * One link in an ordered attachment chain for a robot.
 * An attachment is any MJCF body (end effector, mount, FT sensor, tool, additional arm
 * on a mobile base, etc.) attached under a named element in the accumulated robot spec.
 * Attachments are applied in declaration order; attach_to may reference any body, site,
 * or frame present after all prior attachments have been applied.
 */
struct AttachmentSpec
{
    const char  *mjcf_path = nullptr;      // MJCF file for this attachment
    AttachTarget attach_to;                // parent in root or prior attachment (default: world)
    const char  *prefix  = "";             // element name prefix (avoids name conflicts)
    double       pos[3]  = { 0, 0, 0 };    // position offset [m]
    double       quat[4] = { 0, 0, 0, 1 }; // orientation offset [x, y, z, w]

    /* Contact exclusion pairs registered by attach_to_spec(). */
    std::vector<std::pair<std::string, std::string>> contact_exclusions; // (body1, body2) pairs
};

/**
 * @ingroup grp_types
 * Joint-space control mode. Each mode drives its own actuator, in its own actuator group:
 *   POSITION - jnt_pos_cmd to a position servo's ctrl.
 *   TORQUE   - jnt_trq_cmd to a motor's ctrl.
 *   VELOCITY - jnt_vel_cmd to a velocity actuator's ctrl.
 */
enum class CtrlMode { POSITION, TORQUE, VELOCITY };

/**
 * @ingroup grp_types
 * A control mode for RobotSpec::modes. joints empty = every joint the robot's own MJCF
 * actuates. kv is the VELOCITY actuator's gain [N m s/rad], required for VELOCITY.
 */
struct CtrlModeSpec
{
    CtrlMode                 mode = CtrlMode::TORQUE;
    std::vector<std::string> joints;
    double                   kv = 0.0;
};

/**
 * @ingroup grp_types
 * One robot in a scene: a root MJCF (arm, mobile base, ...) with an ordered attachment
 * chain and a placement target.
 *
 * attachments is applied in order by build_scene() / attach_to_spec(): each entry's
 * attach_to may reference any body, site, or frame in the accumulated spec (root + all
 * prior attachments). This naturally supports: fixed arm, arm+gripper, arm+mount+FT+
 * gripper, mobile base, mobile manipulator (base root, arm as first attachment), etc.
 *
 * attach_to selects where the robot root is placed in the scene; the default is the
 * worldbody. Set it to e.g. { AttachKind::Site, "table_mount" } to place the robot on
 * a tabletop site exported by a prior scene object. pos/quat are offsets in the
 * resolved parent frame.
 *
 * path is the root MJCF passed to build_scene(). prefix must be unique per robot
 * in multi-robot scenes.
 *
 * modes lists the control modes the robot offers beyond the one its own actuators give
 * (a <position> servo gives POSITION, a <motor> gives TORQUE); by default every robot also gets
 * TORQUE. build_scene() adds one actuator per extra mode on each listed joint and puts each mode
 * in its own actuator group, switched with set_control_mode(). Only the robot's own joints take
 * modes, never its attachments. With no joint list, joints that cannot take modes are skipped.
 * Actuator groups 1-30 are reserved for this (group 1 + 3 * robot index + mode); the robot's MJCF
 * must not assign actuator groups itself.
 */
struct RobotSpec
{
    const char                 *path   = nullptr;         // root MJCF path
    const char                 *prefix = "";              // element name prefix
    AttachTarget                attach_to;                // placement parent (default: world)
    double                      pos[3]  = { 0, 0, 0 };    // offset in parent frame [m]
    double                      quat[4] = { 0, 0, 0, 1 }; // orientation offset [x, y, z, w]
    std::vector<AttachmentSpec> attachments;              // ordered attachment chain; empty = none
    std::vector<CtrlModeSpec>   modes = { CtrlModeSpec{} };  // TORQUE; {} = native mode only
};

/** @ingroup grp_types
 *  Shape type for scene objects. Unspecified is the sentinel value;
 *  build_scene rejects a primitive SceneObject whose shape is Unspecified. */
enum class Shape { Unspecified, BOX, SPHERE, CYLINDER };

/**
 * @ingroup grp_types
 * Contact-friction dimensionality, matching MuJoCo's `condim` integer values.
 *   Tangential (3) - sliding friction only (default).
 *   Torsional  (4) - +torsion about the contact normal.
 *   Rolling    (6) - +torsion and +rolling resistance.
 * Values 1 (frictionless) and 2 (1D friction) exist in MuJoCo but are
 * uncommon; if needed, pass `static_cast<Condim>(1)` etc.
 */
enum class Condim : int { Tangential = 3, Torsional = 4, Rolling = 6 };

/**
 * @ingroup grp_types
 * A free-floating or fixed rigid body to place in the scene.
 *
 * size:
 *   BOX       - half-extents (x, y, z)
 *   SPHERE    - {radius, 0, 0}
 *   CYLINDER  - {radius, half-length, 0}
 *   Ignored when mjcf_path is set.
 *
 * attach_to:
 *   Parent in the accumulated scene spec. Default is the worldbody. A child
 *   object must appear after its parent in SceneSpec::objects.
 *   MuJoCo constraint: a body that carries a freejoint must be a direct child
 *   of the worldbody. So a non-fixed primitive (fixed == false) and any
 *   mjcf_path asset whose root body owns a freejoint must use AttachKind::World.
 *   Fixed primitives and articulated subtrees (no freejoint on the root) may
 *   use any kind. mj_compile reports the violation if this rule is broken.
 *
 * pos:
 *   Offset in the resolved parent frame. For MJCF assets, this is the placement
 *   frame for the asset's first root body.
 *
 * fixed:
 *   If true the body is welded to its parent (no freejoint); useful for
 *   static obstacles or fixtures. Ignored when mjcf_path is set.
 *
 * size, rgba, mass and friction start unset (NaN) and must be set explicitly
 * by the caller; build_scene() fails on a primitive that leaves one unset.
 * They are arbitrary visual/material/dynamic choices, not neutral identities,
 * so the API refuses to invent placeholder values. A fixed primitive may leave
 * mass unset.
 *
 * rgba / has_rgba:
 *   A primitive's colour, required. On an MJCF asset it is optional: with
 *   has_rgba set, every geom under the asset's root body takes it, so a scene
 *   can state the colour an object is drawn in without editing the asset.
 */
struct SceneObject
{
    std::string  name;
    std::string  mjcf_path; // optional MJCF asset; when set, shape/size/mass/friction are ignored
    AttachTarget attach_to; // placement parent (default: world)
    Shape shape = Shape::Unspecified; // required for primitives; rejected at build time if not set
    // half-extents (BOX) / {radius, 0, 0} (SPHERE) / {radius, half-len, 0} (CYL)
    double size[3]     = { NAN, NAN, NAN };
    double pos[3]      = { 0.0, 0.0, 0.0 };      // offset in resolved parent frame [m]
    double quat[4]     = { 0.0, 0.0, 0.0, 1.0 }; // orientation offset [x, y, z, w]
    float  rgba[4]     = { NAN, NAN, NAN, NAN }; // [r, g, b, a]; required for primitives
    bool   has_rgba    = false;                  // rgba is set; recolours an asset's geoms
    bool   fixed       = false;
    double mass        = NAN; // [kg]; required for non-fixed primitives
    Condim condim      = Condim::Tangential;
    double friction[3] = { NAN, NAN, NAN }; // [slide, spin, roll]; required for primitives
};

/**
 * @ingroup grp_types
 * A frame to mark on a body of the assembled scene, as a MuJoCo site.
 *
 * Sites are what a model's own frames become: the scene states where they sit, and the
 * runtime can then address and draw them. Added after every robot and object, so `body`
 * may name anything the assembled scene holds. A site the asset already declares under
 * this name is left alone -- the asset's own is authoritative.
 */
struct SiteSpec
{
    std::string body;                             // body to add the site to, by name
    std::string name;                             // site name, unique within the scene
    double      pos[3]  = { 0.0, 0.0, 0.0 };      // offset in the body frame [m]
    double      quat[4] = { 0.0, 0.0, 0.0, 1.0 }; // orientation in the body frame [x, y, z, w]
};

/**
 * @ingroup grp_types
 * A named fixed camera to add to the world body of the scene.
 * After build_scene() the camera is accessible by name via get_camera_names()
 * and can be activated on a Viewer or VideoRecorder with use_camera().
 *
 * pos and fovy start unset (NaN): there is no neutral camera position or field of view, so
 * the caller must specify both, and build_scene() fails otherwise. quat defaults to identity.
 */
struct CameraSpec
{
    std::string name;
    std::string body;                             // anchor body; empty = worldbody
    double      pos[3]  = { NAN, NAN, NAN };      // position in the anchor body's frame [m]
    double      quat[4] = { 0.0, 0.0, 0.0, 1.0 }; // orientation [x, y, z, w]
    double      fovy    = NAN;                    // vertical field of view [degrees]
};

/** @ingroup grp_types
 *  Full scene description passed to build_scene().
 *  timestep (unset: NaN), add_floor, and add_skybox have no defaults: the caller must
 *  choose a physics step and an explicit yes/no for each decoration so the
 *  resulting scene is never silently misconfigured. gravity_z defaults to
 *  Earth gravity. */
struct SceneSpec
{
    std::vector<RobotSpec>   robots;
    double                   timestep  = NAN;   // required; suggested 0.002 [s]
    double                   gravity_z = -9.81; // Earth gravity [m/s^2]
    bool                     add_floor;         // required; checker groundplane geom
    double                   floor_z = 0.0;     // floor plane height in the world frame [m]
    bool                     add_skybox;        // required; gradient sky + directional light
    std::vector<SceneObject> objects;
    std::vector<SiteSpec>    sites;   // frames marked on the assembled scene's bodies
    std::vector<CameraSpec>  cameras; // static world cameras added to worldbody
};

/**
 * @ingroup grp_types
 * Logical force-torque sensor backed by MuJoCo's separate <force> and <torque>
 * sensors. If force_sensor/torque_sensor are omitted, init_robot_from_mjcf()
 * resolves "{name}_force" and "{name}_torque".
 */
struct ForceTorqueSensorSpec
{
    const char *name          = nullptr; // logical wrapper name
    const char *force_sensor  = nullptr; // MuJoCo <force> sensor name
    const char *torque_sensor = nullptr; // MuJoCo <torque> sensor name
    const char *frame_site    = nullptr; // optional site that defines the sensor frame
};

/**
 * @ingroup grp_types
 * What a force-torque sensor measures. reset() assigns a fresh one, so every field here is reset.
 */
struct ForceTorqueReading
{
    KDL::Wrench wrench = KDL::Wrench::Zero(); // updated by update(Env *)
};

/** @ingroup grp_types
 *  A resolved force-torque sensor: its names and addresses, plus the reading. */
struct ForceTorqueSensor : ForceTorqueReading
{
    std::string name;
    std::string force_sensor;
    std::string torque_sensor;
    std::string frame_site;

    int force_adr     = -1;
    int torque_adr    = -1;
    int frame_site_id = -1;
};

/**
 * @ingroup grp_types
 * Optional tool/end-effector description used while building the KDL chain.
 *
 * tool_body names the root of the attached tool subtree whose mass/inertia is
 * lumped into the arm dynamics.  tcp_site names an authored MuJoCo site that
 * becomes the KDL terminal frame for FK/IK (takes priority when set).  When
 * the model has no suitable site, tcp_frame provides an equivalent manual
 * transform expressed in the tip body's local frame.
 * For the prefixed Robotiq 2F-85 this is typically {"g_base", "g_pinch"}.
 */
struct ToolFrameSpec
{
    const char *tool_body = nullptr;
    const char *tcp_site  = nullptr;                // MuJoCo site name (takes priority)
    KDL::Frame  tcp_frame = KDL::Frame::Identity(); // manual TCP in tip frame (fallback)
    std::vector<ForceTorqueSensorSpec> ft_sensors;
};


/**
 * @ingroup grp_types
 * A robot's control ports. reset() assigns a freshly seeded one, so every field here is reset.
 */
struct RobotPorts
{
    CtrlMode             ctrl_mode = CtrlMode::POSITION;
    std::vector<double>  jnt_pos_msr;   // [rad]   measured joint positions (update())
    std::vector<double>  jnt_vel_msr;   // [rad/s] measured joint velocities (update())
    std::vector<double>  jnt_trq_msr;   // [Nm]    actuator output torques (update())
    std::vector<double>  jnt_pos_cmd;   // [rad]   position setpoints (POSITION)
    std::vector<double>  jnt_vel_cmd;   // [rad/s] velocity setpoints (VELOCITY)
    std::vector<double>  jnt_trq_cmd;   // [Nm]    torque commands (TORQUE)
    std::vector<uint8_t> jnt_saturated; // 1 if the command was clamped to ctrlrange (update())
};

struct Env;
struct RobotInternals;

/**
 * @ingroup grp_types
 * One KDL-tracked articulation in an Env. init_robot_from_mjcf() / init_robot_from_chain()
 * configure it and register it with the Env, which then reads, commands and resets it.
 * Registered by address, so it is neither copied nor moved.
 */
struct Robot : RobotPorts
{
    /* Configuration - set by init_robot_from_mjcf() / init_robot_from_chain(). */
    mjModel                               *model = nullptr; // the Env's; updated on a rebuild
    mjData                                *data  = nullptr;
    KDL::Chain                             chain;
    KDL::Frame                             tip_T_tcp     = KDL::Frame::Identity();
    bool                                   has_tcp_frame = false;
    std::string                            tcp_site;
    int                                    n_joints = 0;
    std::vector<std::string>               joint_names;
    std::vector<std::pair<double, double>> joint_limits; // [lo, hi]; +-inf for unlimited joints
    std::vector<ForceTorqueSensor>         ft_sensors;
    bool                                   paused = false; // step() leaves physics alone

    Robot();
    ~Robot();
    Robot(const Robot &)            = delete;
    Robot &operator=(const Robot &) = delete;

    std::unique_ptr<RobotInternals> _impl; // internal
};

/**
 * @ingroup grp_viewer
 * The simulate UI window of an Env. Opened by open_viewer(); closed by cleanup(Env *).
 */
struct Viewer
{
    mjvCamera cam{};
    /* Real-time factor for pace_realtime(): 1.0 = real time, 0.5 = half speed, 0.0 = uncapped.
     * The ',' and '.' keys adjust it. */
    double                                realtime_factor = 1.0;
    std::chrono::steady_clock::time_point _tick_t{};         // internal: pacing
    void                                 *_sim_ui = nullptr; // internal: SimUiState*, open
};

/**
 * @ingroup grp_recorder
 * Standard output resolution presets for init_video_recorder().
 * Each maps to a 16:9 frame size at the named quality level.
 */
enum class VideoResolution {
    R360p  = 360,  // 640  x 360
    R480p  = 480,  // 854  x 480
    R720p  = 720,  // 1280 x 720
    R1080p = 1080, // 1920 x 1080
    R2K    = 1440, // 2560 x 1440
    R4K    = 2160, // 3840 x 2160
};

/**
 * @ingroup grp_recorder
 * Headless video recorder.  Renders frames to an EGL offscreen buffer and
 * pipes raw RGB data to an ffmpeg process, producing an H.264 MP4 without a
 * display server or GLFW window.
 *
 * Requirements: EGL (libegl-dev) and ffmpeg available in PATH.
 *
 * Typical usage:
 *
 *   VideoRecorder vr;
 *   init_video_recorder(&vr, model, "sim.mp4", VideoResolution::R1080p);
 *   vr.cam.azimuth = 135;  vr.cam.elevation = -20;  vr.cam.distance = 2.5;
 *
 *   for (int i = 0; i < steps; ++i) {
 *       mj_step(model, data);
 *       record_frame(&vr, model, data);
 *   }
 *
 *   cleanup(&vr);
 */
struct VideoRecorder
{
    mjvCamera cam{};           // camera configuration; modify freely between frames
    mjvOption opt{};           // rendering options; modify freely between frames
    void     *_impl = nullptr; // opaque EGL + ffmpeg state
};

struct Env;

/** @ingroup grp_env
 * Options controlling an environment reset. */
struct ResetOptions
{
    int  keyframe     = 0;    // keyframe index to use when available
    bool use_keyframe = true; // fall back to mj_resetData when false or invalid
};

/** @ingroup grp_env
 * Information returned by reset(). */
struct ResetInfo
{
    bool used_keyframe = false;
    int  keyframe      = -1;
};

/** @ingroup grp_env
 * Runtime context passed to Env::on_reset, which runs after everything is re-seeded: it may
 * prime commands, and a robot it moves needs its commands set too. */
struct ResetContext
{
    Env                *env     = nullptr;
    mjModel            *model   = nullptr;
    mjData             *data    = nullptr;
    const ResetOptions *options = nullptr;
    ResetInfo          *info    = nullptr;
};

using ResetHook = std::function<void(ResetContext *)>;

/* Scene slots: each derives from what it reads or commands, which reset() assigns afresh. */

/** @ingroup grp_scene
 *  What a scene joint slot reads. */
struct SceneJointReading
{
    double        position = 0.0;
    double        velocity = 0.0;
    std::uint64_t seq      = 0; // bumped by each read
};

/** @ingroup grp_scene
 *  A joint the world model reads that no Robot samples: a gripper mimic, an object hinge. */
struct SceneJointSlot : SceneJointReading
{
    std::string name;
    int         qpos_adr = -1;
    int         dof_adr  = -1;
};

/** @ingroup grp_scene
 *  What a free-body slot reads. */
struct SceneFreeBodyReading
{
    KDL::Frame    pose;
    std::uint64_t seq = 0; // bumped by each read
};

/** @ingroup grp_scene
 *  A free body: its freejoint's qpos is the body pose in the world frame. */
struct SceneFreeBodySlot : SceneFreeBodyReading
{
    std::string name;
    int         qpos_adr = -1;
};

/** @ingroup grp_scene
 *  What a wrench slot commands. */
struct SceneWrenchCommand
{
    KDL::Wrench wrench = KDL::Wrench::Zero();
};

/** @ingroup grp_scene
 *  A body a controller pushes: the wrench is applied as xfrc_applied at the body origin. */
struct SceneWrenchSlot : SceneWrenchCommand
{
    std::string name;
    int         body_id = -1;
};

/** @ingroup grp_scene
 *  What an actuator slot commands. */
struct SceneActuatorCommand
{
    double command   = 0.0;   // in ctrl units
    bool   saturated = false; // command was clamped to ctrlrange (update(Env *))
};

/** @ingroup grp_scene
 *  An actuator a controller commands directly, outside the KDL chain: a gripper drive. */
struct SceneActuatorSlot : SceneActuatorCommand
{
    std::string name;
    int         ctrl_id = -1;
};

/**
 * @ingroup grp_scene
 * Every non-robot primitive of the scene, read and applied by update(Env *).
 * Deques: a pointer to a slot stays valid as more are bound.
 */
struct SceneState
{
    const mjModel                *model = nullptr;
    std::deque<SceneJointSlot>    joints;
    std::deque<SceneFreeBodySlot> free_bodies;
    std::deque<SceneWrenchSlot>   wrenches;
    std::deque<SceneActuatorSlot> actuators;
};

struct EnvInternals;

/** @ingroup grp_env
 * The simulation: the compiled scene and everything that reads, commands or shows it.
 * init_env() builds it; step(), update() and reset() drive it; cleanup(Env *) frees it.
 * Robots are borrowed: registered by init_robot_from_mjcf() / init_robot_from_chain(), never
 * deleted here. Not copied or moved (robots and slots point into it).
 */
struct Env
{
    SceneSpec            spec;
    mjModel             *model = nullptr;
    mjData              *data  = nullptr;
    std::vector<Robot *> robots;
    SceneState           scene;  // bind slots with bind_scene_*(&env.scene, ...)
    Viewer               viewer; // closed until open_viewer()
    ResetHook            on_reset;

    Env();
    ~Env();
    Env(const Env &)            = delete;
    Env &operator=(const Env &) = delete;

    std::unique_ptr<EnvInternals> _impl; // internal
};

/**
 * @ingroup grp_scene
 * Save a model to an MJCF XML file, including runtime changes to its real-valued fields.
 * Works for any live model from build_scene()/compile_and_make_data(), and for the last model
 * loaded with mj_loadXML. Typical use: build a combined scene once, save it, reload it later.
 * @param model  Model to save.
 * @param path   Output path for the MJCF XML file.
 * @return true on success.
 */
bool save_model_xml(const mjModel *model, const char *path);

/**
 * @ingroup grp_robot
 * Build KDL chain from a compiled MuJoCo model and optional tool/TCP metadata.
 *
 * If tool->tcp_site is set, that authored site becomes the KDL terminal frame
 * for FK/IK.  The joint count and MuJoCo joint/actuator maps still cover only
 * the controllable joints from base_body to tip_body.
 * Pass tool = nullptr (the default) for an arm with no attached tool.
 * Registers r with env, which reads, commands and resets it from then on.
 */
bool init_robot_from_mjcf(
  Robot               *r,
  Env                 *env,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix = "",
  const ToolFrameSpec *tool   = nullptr
);

/**
 * @ingroup grp_robot
 * Adopt an externally built KDL chain and wire it to a compiled MuJoCo model.
 *
 * Same as init_robot_from_mjcf(), except the chain is supplied rather than derived
 * from the model: use this when the chain comes from the scene description that also
 * produced the MJCF, so the solvers compute with the authored dynamics.  The chain is
 * taken as given - no tool inertia is lumped onto it, since such a chain already
 * carries its tool as explicit segments.
 *
 * joint_names lists the MuJoCo joint names in KDL chain order, one per chain joint;
 * they drive the same qpos/dof/ctrl index maps init_robot_from_mjcf() builds, and
 * prefix is applied to each as there.  tool is used only to resolve FT sensors;
 * tool->tool_body and tool->tcp_site are ignored. Registers r with env.
 */
bool init_robot_from_chain(
  Robot                          *r,
  Env                            *env,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix = "",
  const ToolFrameSpec            *tool   = nullptr
);

/** @ingroup grp_robot Find a configured logical force-torque sensor by name. */
const ForceTorqueSensor *find_ft_sensor(const Robot *r, const char *name);

/**
 * @ingroup grp_robot
 * Per-joint torque/force saturation limit in KDL joint order for the actuator
 * of the robot's `ctrl_mode`: max(|lo|, |hi|) of its `forcerange` times |gear|,
 * and in TORQUE mode at most its `ctrlrange` times |gear|. Joints with no
 * actuator for that mode or an unlimited one fall back to `fallback`.
 *
 * @param r         Initialized robot (init_robot_from_mjcf() already called).
 * @param fallback  Bound used for joints without a force-limited actuator;
 *                  default is a large, effectively non-limiting value.
 * @return          One entry per KDL joint, same order as r->joint_names.
 */
std::vector<double> joint_force_limits(const Robot *r, double fallback = 1e6);

/**
 * @ingroup grp_scene
 * Apply one attachment to an arm spec using the MuJoCo spec API (mjs_attach).
 * Parses a->mjcf_path, attaches its first root body under a->attach_to with the given
 * pos/quat offset, prefixes all element names with a->prefix, and registers contact
 * exclusions via mjs_addExclude.  Can be called repeatedly to build a chain: each
 * subsequent a->attach_to may reference any body added by prior calls.
 * @param[in,out] robot_spec  Accumulated robot spec to attach into.
 * @param[in]     a           Attachment; a->mjcf_path must not be null.
 * @return true on success.
 */
bool attach_to_spec(mjSpec *robot_spec, const AttachmentSpec *a);

/**
 * @ingroup grp_scene
 * Build a MuJoCo scene from one or more robots using the MuJoCo spec API.
 * This is the primary scene-building function.
 *
 * For each RobotSpec: mj_parseXML loads the root MJCF, then attach_to_spec() applies
 * each entry in RobotSpec::attachments in order (mount, sensor, gripper, etc.),
 * and mjs_attach places the complete robot spec at the given position.  A single
 * mj_compile produces the final model -- no intermediate XML files are written.
 *
 * @param[out] out_model  Newly allocated MuJoCo model; caller frees via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller frees via destroy_scene().
 * @param[in]  spec       Scene description: robots (with attachment chains), table,
 *                        objects, timestep, gravity, floor, skybox.
 * @return true on success.
 */
bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *spec);

/**
 * @ingroup grp_scene
 * Free a model/data pair allocated by any scene-building function.
 * @param[in] model  Model to free (may be null).
 * @param[in] data   Data to free (may be null).
 */
void destroy_scene(mjModel *model, mjData *data);

/**
 * @ingroup grp_env
 * Build the scene from spec into env, which owns the model/data until cleanup(Env *).
 */
bool init_env(Env *env, const SceneSpec *spec);

/**
 * @ingroup grp_env
 * Reset everything env holds: MuJoCo data to the keyframe (or the model default); every
 * registered robot's ports (holding the reset pose, a requested ctrl_mode kept) and F/T readings,
 * and every scene slot, re-seeded; then Env::on_reset; then measurements read from the result.
 * The simulate UI's reset button does the same.
 */
ResetInfo reset(Env *env, const ResetOptions *options = nullptr);

/**
 * @ingroup grp_viewer
 * Open the simulate UI (panels, physics controls) on env, rendered on a background thread;
 * step() then drives physics, pause, perturbation and recording. Linux (X11 / Wayland) only.
 * @return false when there is no display or the window cannot be created.
 */
bool open_viewer(Env *env, const char *title = "MuJoCo");

/**
 * @ingroup grp_viewer
 * Reset the viewer's user-scene geom count to 0.
 * Call once per frame before appending trace segments with add_trace_segment().
 * No-op while the viewer is closed (e.g. headless).
 * @param[in,out] v  An Env's viewer.
 */
void clear_trace(Viewer *v);

/**
 * @ingroup grp_viewer
 * Append a single line segment to the viewer's user scene. Thread-safe.
 * The render thread merges the user scene into each frame automatically.
 * Silently drops the segment once the user-scene geom buffer is full.
 * No-op while the viewer is closed (e.g. headless).
 * @param[in,out] v     An Env's viewer.
 * @param[in]     a     Segment start point (world frame) [m].
 * @param[in]     b     Segment end point (world frame) [m].
 * @param[in]     rgba  Optional [r, g, b, a] colour; nullptr -> warm orange.
 */
void add_trace_segment(
  Viewer            *v,
  const KDL::Vector &a,
  const KDL::Vector &b,
  const float        rgba[4] = nullptr
);

/**
 * @ingroup grp_viewer
 * Append a world-space arrow to the viewer's user scene. Thread-safe.
 * Shares the user scene with add_trace_segment(), so clear_trace() clears both
 * and one call per frame is enough for either.
 * dir need not be normalised; a zero-length dir draws nothing.
 * Silently drops the arrow once the user-scene geom buffer is full.
 * No-op while the viewer is closed (e.g. headless).
 * @param[in,out] v       An Env's viewer.
 * @param[in]     from    Arrow tail (world frame) [m].
 * @param[in]     dir     Direction the arrow points; normalised internally.
 * @param[in]     length  Arrow length [m].
 * @param[in]     rgba    Optional [r, g, b, a] colour; nullptr -> warm orange.
 */
void add_overlay_arrow(
  Viewer            *v,
  const KDL::Vector &from,
  const KDL::Vector &dir,
  double             length,
  const float        rgba[4] = nullptr
);

/**
 * @ingroup grp_robot
 * Unregister r from its Env and clear it. The Env keeps running without it.
 */
void cleanup(Robot *r);

/**
 * @ingroup grp_env
 * Close the viewer, free the model/data and forget the robots (which are not deleted).
 */
void cleanup(Env *env);

/**
 * @ingroup grp_recorder
 * Initialise a headless EGL video recorder.
 * Creates an EGL context, an offscreen render target, and launches an ffmpeg
 * process (H.264/MP4) via a pipe.  The MuJoCo model is used to size the scene
 * and initialise the rendering context; it must remain valid until cleanup().
 *
 * @param vr        VideoRecorder to initialise; freed by cleanup(VideoRecorder*).
 * @param model     MuJoCo model for the rendering context.
 * @param out_path  Output MP4 path (e.g. "sim.mp4").
 * @param width     Frame width in pixels (default 1280).
 * @param height    Frame height in pixels (default 720).
 * @param fps       Playback frame rate (default 60).
 * @return true on success; false if EGL init or ffmpeg launch fails.
 */
bool init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width  = 1280,
  int            height = 720,
  int            fps    = 60
);

/**
 * @ingroup grp_recorder
 * Convenience overload: initialise a VideoRecorder using a named resolution preset.
 * Frame width is derived from the preset at 16:9 aspect ratio.
 *
 * @param vr         VideoRecorder to initialise.
 * @param model      MuJoCo model.
 * @param out_path   Output MP4 path.
 * @param resolution VideoResolution preset (e.g. VideoResolution::R1080p).
 * @param fps        Playback frame rate (default 60).
 * @return true on success.
 */
bool init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps = 60
);

/**
 * @ingroup grp_recorder
 * Render env's current state and write one frame to the video stream.
 *
 * @param vr   VideoRecorder initialised by init_video_recorder().
 * @param env  Env to render.
 * @return true on success; false on render or pipe write error.
 */
bool record_frame(VideoRecorder *vr, Env *env);

/**
 * @ingroup grp_recorder
 * Initialise offscreen rendering only: EGL context and MuJoCo render buffers,
 * no ffmpeg process and no output file. Use with render_rgb() to grab frames;
 * use_camera(VideoRecorder*, ...) and cleanup(VideoRecorder*) work unchanged.
 *
 * @param vr      VideoRecorder to initialise; freed by cleanup(VideoRecorder*).
 * @param model   MuJoCo model for the rendering context.
 * @param width   Frame width in pixels.
 * @param height  Frame height in pixels.
 * @return true on success; false if EGL init fails.
 */
bool init_offscreen(VideoRecorder *vr, mjModel *model, int width, int height);

/**
 * @ingroup grp_recorder
 * Render env's current state into a caller-owned top-down RGB8 buffer
 * of width*height*3 bytes.
 *
 * @param vr   VideoRecorder initialised by init_offscreen() or init_video_recorder().
 * @param env  Env to render.
 * @param out  Destination buffer, width*height*3 bytes.
 * @return true on success.
 */
bool render_rgb(VideoRecorder *vr, Env *env, std::uint8_t *out);

/**
 * @ingroup grp_recorder
 * Flush the ffmpeg pipe, finalise the MP4, and release all EGL resources.
 * After this call vr->_impl is null and the VideoRecorder may be discarded.
 *
 * @param vr  VideoRecorder to tear down.
 */
void cleanup(VideoRecorder *vr);

/**
 * @ingroup grp_env
 * Advance env one timestep: mj_step2() then mj_step1() (MuJoCo's split mj_step()). Afterwards
 * joint state, body frames and position/velocity sensors all describe the new state; force and
 * acceleration sensors describe the step just taken. With the viewer open it also honours its
 * pause, perturbation and recording, and does nothing while every robot is paused.
 * @return false once the viewer window is closed; always true headless.
 */
bool step(Env *env);

/**
 * @ingroup grp_viewer
 * Sleep out this step's share of wall time at the viewer's real-time factor. step() never
 * sleeps; a loop that paces itself reads realtime_factor_of() instead. No-op headless.
 */
void pace_realtime(Env *env);

/**
 * @ingroup grp_viewer
 * The viewer's current real-time factor, as the user has set it with the speed keys.
 * @param[in] v  Viewer, or nullptr.
 * @return the factor; 0.0 means uncapped ("RTF: MAX"), 1.0 if @p v is nullptr.
 */
double realtime_factor_of(const Viewer *v);

/**
 * @ingroup grp_viewer
 * Returns true while the viewer window is open.
 * @param[in] v  An Env's viewer.
 */
bool is_running(const Viewer *v);

/**
 * @ingroup grp_viewer
 * Whether a key is currently held down in the viewer's window.
 *
 * The simulate UI opened by open_viewer() owns its GLFW window on the
 * render thread, so a caller driving physics on its own thread must not call
 * glfwGetKey() itself. This reads the key state that the UI's own key callback
 * records, which is safe from any thread.
 *
 * Keys the UI consumes for itself (',' and '.' for the speed control) are
 * reported like any other.
 *
 * @param[in] v         Viewer, or nullptr.
 * @param[in] glfw_key  A GLFW key code, e.g. GLFW_KEY_UP.
 * @return true while the key is held; false for a nullptr or headless viewer,
 *         or an out-of-range key code.
 */
bool key_pressed(const Viewer *v, int glfw_key);

/**
 * @ingroup grp_viewer
 * Claim a key for the caller, so the simulate UI never acts on it.
 *
 * The UI binds keys of its own: the left and right arrows scrub the history
 * and single-step, escape restores the free camera, space pauses. A caller
 * that drives a robot with those keys would otherwise fight the UI for them.
 * A captured key is still reported by key_pressed(); it is only withheld from
 * the UI's own handler.
 *
 * @param[in,out] v         An Env's viewer, or nullptr.
 * @param[in]     glfw_key  A GLFW key code, e.g. GLFW_KEY_LEFT.
 * @param[in]     capture   true to claim the key, false to give it back.
 */
void capture_key(Viewer *v, int glfw_key, bool capture = true);

/**
 * @ingroup grp_env
 * One control cycle for everything env holds: read, then apply.
 * Robots: qpos -> jnt_pos_msr, qvel -> jnt_vel_msr, qfrc_actuator -> jnt_trq_msr (only the active
 * mode's actuators produce force, so this is the drive torque in every mode), F/T wrenches; then
 * the active mode's command to its actuators' ctrl (gear applied, ctrlrange clamped). If
 * ctrl_mode was changed directly, the switch runs first (set_control_mode()).
 * Scene slots: joints and free bodies sampled from qpos/qvel; wrenches into xfrc_applied (zeros
 * included), actuator commands into ctrl.
 */
void update(Env *env);

/**
 * @ingroup grp_robot
 * Switch the robot to mode: seed the new actuators so nothing jumps, enable their group,
 * disable the robot's other mode groups. @return false if the robot has no actuator for mode.
 */
bool set_control_mode(Robot *r, CtrlMode mode);

/**
 * @ingroup grp_robot
 * The same switch for a robot driven outside a Robot chain (e.g. through env.scene):
 * robot is its index in SceneSpec::robots. @return false if it has no actuators for mode.
 */
bool set_control_mode(Env *env, int robot, CtrlMode mode);

/**
 * @ingroup grp_scene
 * Resolve a scalar joint by name and keep a slot for it.
 * Rejects an unknown name, a free or ball joint, and a name already bound.
 * @return the slot, or nullptr on failure. The address stays valid across further binds.
 */
SceneJointSlot *bind_scene_joint(SceneState *s, const char *joint_name);

/**
 * @ingroup grp_scene
 * Resolve a free-floating body by name and keep a slot for its pose.
 * Rejects an unknown name, a body that owns no mjJNT_FREE joint, and a name already bound.
 * @return the slot, or nullptr on failure. The address stays valid across further binds.
 */
SceneFreeBodySlot *bind_scene_free_body(SceneState *s, const char *body_name);

/**
 * @ingroup grp_scene
 * Resolve a body by name and keep a slot for the wrench applied to it.
 * Rejects an unknown name and a name already bound.
 * @return the slot, or nullptr on failure. The address stays valid across further binds.
 */
SceneWrenchSlot *bind_scene_wrench(SceneState *s, const char *body_name);

/**
 * @ingroup grp_scene
 * Resolve the actuator named, or the one driving the joint of that name, and keep a slot for
 * its command. Rejects a name that reaches no actuator, and a name already bound.
 * @return the slot, or nullptr on failure. The address stays valid across further binds.
 */
SceneActuatorSlot *bind_scene_actuator(SceneState *s, const char *name);

/**
 * @ingroup grp_robot
 * Write KDL joint positions into MuJoCo qpos (KDL chain order -> MuJoCo addresses). Frames
 * read afterwards follow the new positions.
 * @param[in,out] r  Registered robot.
 * @param[in]     q  Joint positions in KDL chain order; size must equal r->n_joints.
 */
void set_joint_pos(Robot *r, const KDL::JntArray &q);

/**
 * @ingroup grp_env
 * Teleport a free-floating body to a world-frame position and optionally a
 * world-frame orientation, then zero its velocity.
 * body_name must identify a body that owns a mjJNT_FREE joint.
 * quat is MuJoCo convention [w, x, y, z]; pass nullptr to keep identity orientation.
 */
void set_body_pose(
  Env          *env,
  const char   *body_name,
  const double  pos[3],
  const double *quat = nullptr
);

/**
 * @ingroup grp_scene
 * Append obj to env->spec.objects and rebuild. Robots, scene slots and the viewer follow the
 * new model; a slot whose name is gone is unbound and skipped.
 * @return true on success; env unchanged on failure.
 */
bool scene_add_object(Env *env, const SceneObject &obj);

/**
 * @ingroup grp_scene
 * Remove the named object from env->spec.objects and rebuild, as scene_add_object() does.
 * @return true on success; false if name not found or rebuild fails.
 */
bool scene_remove_object(Env *env, const std::string &name);

/**
 * @ingroup grp_scene
 * Return the compiled MuJoCo name for a site inside an MJCF-backed SceneObject.
 * build_scene() prefixes all MJCF asset element names with obj.name + "_".
 */
std::string scene_object_site_name(const SceneObject &obj, const char *site_name);

/**
 * @ingroup grp_scene
 * Read a named MuJoCo site as a world-frame KDL frame. Recomputes the kinematics only when the
 * state has changed since they were last computed (a direct qpos write included).
 */
bool get_site_frame(Env *env, const char *site_name, KDL::Frame *out);

/**
 * @ingroup grp_scene
 * Read a named MuJoCo body as a world-frame KDL frame, as get_site_frame() does.
 */
bool get_body_frame(Env *env, const char *body_name, KDL::Frame *out);

/**
 * @ingroup grp_scene
 * Read a joint's position (qpos) in physical units (rad or m), by joint name.
 * If the name is not a joint, it is treated as an actuator name and resolved to
 * its transmission joint (direct joint, or the first tendon-wrapped joint).
 */
bool get_joint_position(Env *env, const char *name, double *out);

/**
 * @ingroup grp_scene
 * Read a joint's velocity (qvel) in physical units (rad/s or m/s), by joint name.
 * Resolves the name exactly as get_joint_position() does.
 */
bool get_joint_velocity(Env *env, const char *name, double *out);

/**
 * @ingroup grp_scene
 * Return the names of all cameras in a compiled model.
 * Includes cameras from robot MJCFs (e.g. the Kinova wrist camera) and any
 * cameras added via SceneSpec::cameras.
 */
std::vector<std::string> get_camera_names(const mjModel *model);

/**
 * @ingroup grp_viewer
 * Switch the viewer to a named fixed camera defined in the model.
 * Pass nullptr or an empty string to return to the free camera.
 * @return true if the camera name was found; false if not found (viewer unchanged).
 */
bool use_camera(Viewer *v, const mjModel *model, const char *name);

/**
 * Configure the viewer's free orbit camera.
 */
void set_free_camera(
  Viewer                      *v,
  double                       distance,
  double                       azimuth,
  double                       elevation,
  const std::array<double, 3> &lookat
);

/**
 * @ingroup grp_recorder
 * Switch the video recorder to a named fixed camera defined in the model.
 * @return true if the camera name was found; false if not found (recorder unchanged).
 */
bool use_camera(VideoRecorder *vr, const mjModel *model, const char *name);

/**
 * Internal spec-building helpers.
 *
 * These are used internally by build_scene() but are exposed here for advanced
 * callers that construct mjSpec objects directly. They are not part of the
 * stable public API and may change between releases.
 */

/**
 * @ingroup grp_advanced
 * Add a sky gradient texture and overhead directional light to spec.
 * Corresponds to SceneSpec::add_skybox.
 */
void add_skybox_to_spec(mjSpec *spec);

/**
 * @ingroup grp_advanced
 * Add a checker groundplane texture, material, and floor plane geom to spec.
 * Corresponds to SceneSpec::add_floor, placed at floor_z along the world z axis
 * so a scene whose world frame is not at ground level still gets a ground.
 */
void add_floor_to_spec(mjSpec *spec, double floor_z = 0.0);

/**
 * @ingroup grp_advanced
 * Add free-floating or fixed rigid bodies to the world body of spec.
 * @param spec     MuJoCo spec to modify.
 * @param objects  List of objects to add.
 * @return false (logged) on the first object that cannot be added, e.g. a field left unset.
 */
bool add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects);

/**
 * @ingroup grp_advanced
 * Compile spec into a model and create its data buffer. Takes ownership of spec: freed on
 * failure, otherwise kept with the model (for save_model_xml) and freed by destroy_scene().
 * @param[in]  spec       MuJoCo spec to compile; owned by the wrapper from here on.
 * @param[out] out_model  Newly allocated model on success; null on failure.
 * @param[out] out_data   Newly allocated data on success; null on failure.
 * @return true on success.
 */
bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data);

/**
 * @ingroup grp_advanced
 * Load MuJoCo decoder plugins (STL, OBJ, ...) once at first use.
 * Required for external mesh decoder plugin libraries.
 * Called automatically by all scene-building functions; call explicitly only
 * when building a scene via raw mjSpec APIs without going through the library.
 */
void ensure_plugins_loaded();

} // namespace mj_kdl
