/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <cstdio>
#include <cstring>
#include <functional>
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

#define MJ_LOG_(lvl_enum, color, label, expr)                        \
    do {                                                             \
        if (::mj_kdl::g_log_level >= ::mj_kdl::LogLevel::lvl_enum) { \
            std::ostringstream _mj_oss;                              \
            _mj_oss << expr; /* NOLINT(bugprone-macro-parentheses) */ \
            std::fprintf(                                            \
              stderr,                                                \
              color "[mj_kdl " label "] %s:%d (%s): %s\033[0m\n",    \
              MJ_FILENAME_,                                          \
              __LINE__,                                              \
              __func__,                                              \
              _mj_oss.str().c_str()                                  \
            );                                                       \
        }                                                            \
    } while (0)

#define LOG_INFO(expr)  MJ_LOG_(INFO,  "",          "INFO ", expr)
#define LOG_WARN(expr)  MJ_LOG_(WARN,  "\033[33m",  "WARN ", expr)
#define LOG_ERROR(expr) MJ_LOG_(ERROR, "\033[31m",  "ERROR", expr)

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
 * accompanying pos/euler are an additional offset on top of it (matches MJCF).
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
    const char  *mjcf_path = nullptr;     // MJCF file for this attachment
    AttachTarget attach_to;               // parent in root or prior attachment (default: world)
    const char  *prefix    = "";          // element name prefix (avoids name conflicts)
    double       pos[3]    = { 0, 0, 0 }; // position offset [m]
    double       euler[3]  = { 0, 0, 0 }; // extrinsic XYZ Euler offset [degrees]

    /* Contact exclusion pairs registered by attach_to_spec(). */
    std::vector<std::pair<std::string, std::string>> contact_exclusions; // (body1, body2) pairs
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
 * a tabletop site exported by a prior scene object. pos/euler are offsets in the
 * resolved parent frame.
 *
 * path is the root MJCF passed to build_scene(). prefix must be unique per robot
 * in multi-robot scenes.
 */
struct RobotSpec
{
    const char                 *path     = nullptr;     // root MJCF path
    const char                 *prefix   = "";          // element name prefix
    AttachTarget                attach_to;              // placement parent (default: world)
    double                      pos[3]   = { 0, 0, 0 }; // offset in parent frame [m]
    double                      euler[3] = { 0, 0, 0 }; // extrinsic XYZ Euler offset [degrees]
    std::vector<AttachmentSpec> attachments;            // ordered attachment chain; empty = none
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
 * Fields without an inline default (size, rgba, mass, friction) must be set
 * explicitly by the caller. They are arbitrary visual/material/dynamic
 * choices, not neutral identities, so the API refuses to invent placeholder
 * values.
 */
struct SceneObject
{
    std::string  name;
    std::string  mjcf_path;  // optional MJCF asset; when set, shape/size/mass/friction are ignored
    AttachTarget attach_to;  // placement parent (default: world)
    Shape        shape    = Shape::Unspecified; // required for primitives; rejected at build time if not set
    double       size[3]; // half-extents (BOX) / {radius, 0, 0} (SPHERE) / {radius, half-len, 0} (CYL)
    double       pos[3]  = { 0.0, 0.0, 0.0 }; // offset in resolved parent frame [m]
    float        rgba[4];                     // [r, g, b, a]; required for primitives
    bool         fixed    = false;
    double       mass;                        // [kg]; required for primitives
    Condim       condim   = Condim::Tangential;
    double       friction[3];                 // [slide, spin, roll]; required for primitives
};

/**
 * @ingroup grp_types
 * A named fixed camera to add to the world body of the scene.
 * After build_scene() the camera is accessible by name via get_camera_names()
 * and can be activated on a Viewer or VideoRecorder with use_camera().
 *
 * pos and fovy have no defaults: there is no neutral camera position or
 * field of view, so the caller must specify both. euler defaults to identity.
 */
struct CameraSpec
{
    std::string name;
    double      pos[3];                       // world-frame position [m]
    double      euler[3] = { 0.0, 0.0, 0.0 }; // extrinsic XYZ Euler [degrees]
    double      fovy;                         // vertical field of view [degrees]
};

/** @ingroup grp_types
 *  Full scene description passed to build_scene().
 *  timestep, add_floor, and add_skybox have no defaults: the caller must
 *  choose a physics step and an explicit yes/no for each decoration so the
 *  resulting scene is never silently misconfigured. gravity_z defaults to
 *  Earth gravity. */
struct SceneSpec
{
    std::vector<RobotSpec>   robots;
    double                   timestep;          // required; suggested 0.002 [s]
    double                   gravity_z = -9.81; // Earth gravity [m/s^2]
    bool                     add_floor;         // required; checker groundplane geom
    bool                     add_skybox;        // required; gradient sky + directional light
    std::vector<SceneObject> objects;
    std::vector<CameraSpec>  cameras; // static world cameras added to worldbody
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
};

/**
 * @ingroup grp_types
 * Joint-space control mode for update().
 *   POSITION - writes jnt_pos_cmd to actuator ctrl inputs.
 *   TORQUE   - writes jnt_trq_cmd to qfrc_applied (generalized forces).
 */
enum class CtrlMode { POSITION, TORQUE };

/**
 * @ingroup grp_types
 * Runtime handle for one KDL-tracked articulation inside a MuJoCo scene.
 * model/data are borrowed (never freed by cleanup()); call destroy_scene() separately.
 *
 * Workflow:
 *   1. Call init_robot_from_mjcf() - populates configuration and sizes port vectors to n_joints.
 *   2. Each control step: read *_msr ports (updated by update()), fill *_cmd ports,
 *      call update() to apply commands to MuJoCo and read back sensor state.
 */
struct Robot
{
    /* Configuration - set once by init_robot() / init_from_mjcf(). */
    mjModel                               *model = nullptr;
    mjData                                *data  = nullptr;
    KDL::Chain                             chain;
    KDL::Frame                             tip_T_tcp     = KDL::Frame::Identity();
    bool                                   has_tcp_frame = false;
    std::string                            tcp_site;
    int                                    n_joints = 0;
    std::vector<std::string>               joint_names;
    std::vector<std::pair<double, double>> joint_limits;

    /* Ports - read/written each control cycle. */
    CtrlMode            ctrl_mode = CtrlMode::POSITION;
    bool                paused    = false;
    std::vector<double> jnt_pos_msr; // [rad]   - measured joint positions   (written by update())
    std::vector<double> jnt_vel_msr; // [rad/s] - measured joint velocities  (written by update())
    std::vector<double> jnt_trq_msr; // [Nm]    - actuator output torques    (written by update())
    std::vector<double> jnt_pos_cmd; // [rad] - position setpoints  (POSITION mode)
    std::vector<double> jnt_trq_cmd; // [Nm]  - torque commands     (TORQUE mode)

    /* Internal state - populated by init_robot() / init_from_mjcf(). */
    std::vector<int> kdl_to_mj_qpos; // KDL index -> MuJoCo qpos address
    std::vector<int> kdl_to_mj_dof;  // KDL index -> MuJoCo dof address
    std::vector<int> kdl_to_mj_ctrl; // KDL index -> MuJoCo ctrl index (-1 if none)
};

/**
 * @ingroup grp_viewer
 * GLFW window and MuJoCo visualization state for the manual render loop.
 * Created by init_window(); freed by cleanup(Viewer *).
 */
struct Viewer
{
    GLFWwindow *window = nullptr;
    mjvScene    scn{};
    mjvCamera   cam{};
    mjvOption   opt{};
    mjvPerturb  pert{};
    mjrContext  con{};
    /* Real-time factor controlling simulation speed in step()/tick().
     * 1.0 = real-time (default), 2.0 = 2x faster, 0.5 = half speed.
     * Keyboard: ',' slows down, '.' speeds up in both viewer modes.
     * 0.0 means run as fast as possible (no sleep). */
    double realtime_factor = 1.0;
    /* internal: real-time pacing state used by tick(). */
    std::chrono::steady_clock::time_point _tick_t{};
    /* internal: non-null when init_window_sim() is used; holds SimUiState*. */
    void *_sim_ui = nullptr;
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
 * Runtime context passed to Env::on_reset after MuJoCo data has been reset and
 * before mj_forward() and robot command-port synchronisation. */
struct ResetContext
{
    Env                *env     = nullptr;
    mjModel            *model   = nullptr;
    mjData             *data    = nullptr;
    const ResetOptions *options = nullptr;
    ResetInfo          *info    = nullptr;
};

using ResetHook = std::function<void(ResetContext *)>;

/** @ingroup grp_env
 * Runtime environment instance: declarative SceneSpec plus compiled MuJoCo
 * model/data and Robot handles that should be synchronised after reset.
 *
 * Env owns model/data created by init_env(); registered Robot pointers are
 * borrowed and are never deleted by cleanup(Env *). Robot::model/data remain
 * borrowed aliases for compatibility with the existing robot-centric API.
 */
struct Env
{
    SceneSpec            spec;
    mjModel             *model = nullptr;
    mjData              *data  = nullptr;
    std::vector<Robot *> robots;
    ResetHook            on_reset;
};

/**
 * @ingroup grp_scene
 * Save the compiled model to an MJCF XML file for later reloading via build_scene().
 * Must be called with the model returned by the most recent build_scene() call -
 * MuJoCo only retains the last compiled model's XML internally.
 * Typical use: build a combined scene (dual-arm, arm+gripper, ...) once, save it,
 * then reload via build_scene() in subsequent runs to skip all build steps.
 * @param model  Model to save; must be the most recently compiled model.
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
 */
bool init_robot_from_mjcf(
  Robot               *r,
  mjModel             *model,
  mjData              *data,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix = "",
  const ToolFrameSpec *tool   = nullptr
);

/**
 * @ingroup grp_scene
 * Apply one attachment to an arm spec using the MuJoCo spec API (mjs_attach).
 * Parses a->mjcf_path, attaches its first root body under a->attach_to with the given
 * pos/euler offset, prefixes all element names with a->prefix, and registers contact
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
 * Build a runtime environment from a declarative SceneSpec.
 * The resulting model/data are owned by env and freed by cleanup(Env *).
 */
bool init_env(Env *env, const SceneSpec *spec);

/**
 * @ingroup grp_env
 * Register a Robot handle to be synchronised after environment reset.
 * The robot is borrowed; the Env does not delete or clean it up.
 */
void env_add_robot(Env *env, Robot *robot);

/**
 * @ingroup grp_env
 * Reset the whole environment to a keyframe/default state, call Env::on_reset
 * for user-specific robot/object/task restoration, then sync all registered
 * Robot command ports and clear stale robot forces.
 */
ResetInfo reset(Env *env, const ResetOptions *options = nullptr);


/**
 * @ingroup grp_viewer
 * Open a GLFW window and initialise MuJoCo visualization contexts.
 * Must be called after init_robot() or init_from_mjcf().
 * @param[out] v      Viewer to initialise; must be zero-initialised before call.
 * @param[in]  r      Robot whose model drives the rendering context.
 * @param[in]  title  Window title string.
 * @param[in]  width  Window width in pixels.
 * @param[in]  height Window height in pixels.
 * @return true on success, false if GLFW or MuJoCo context creation fails.
 */
bool init_window(
  Viewer     *v,
  Robot      *r,
  const char *title  = "MuJoCo",
  int         width  = 1280,
  int         height = 720
);

/**
 * @ingroup grp_viewer
 * Open the full MuJoCo simulate UI (panels, physics controls, joint viewer)
 * in a background render thread, then return so the caller can drive the
 * physics loop with tick().
 *
 * Use this instead of init_window() when you want the simulate UI panels
 * alongside a user-owned loop.  tick() automatically acquires the render
 * mutex, steps physics, and handles pause / perturbation / speed controls.
 *
 * Note: the render thread owns the GLFW window; on Linux (X11 / Wayland)
 * this works correctly.  Not supported on macOS.
 *
 * @param[out] v      Viewer to initialise; freed by cleanup(Viewer *).
 * @param[in]  r      Robot to simulate.  r is registered globally; pass the same
 *                    Robot to every subsequent step() call so that keyboard and
 *                    mouse perturbation callbacks operate on the correct model.
 *                    Only one (Viewer, Robot) pair may be active at a time.
 * @param[in]  title  Label shown in the window title bar (default "MuJoCo").
 * @return true on success.
 */
bool init_window_sim(Viewer *v, Robot *r, const char *title = "MuJoCo");

/**
 * @ingroup grp_viewer
 * Reset the viewer's user-scene geom count to 0.
 * Call once per frame before appending trace segments with add_trace_segment().
 * No-op when v is not backed by an init_window_sim() window (e.g. headless).
 * @param[in,out] v  Viewer initialised by init_window_sim().
 */
void clear_trace(Viewer *v);

/**
 * @ingroup grp_viewer
 * Append a single line segment to the viewer's user scene. Thread-safe.
 * The render thread merges the user scene into each frame automatically.
 * Silently drops the segment once the user-scene geom buffer is full.
 * No-op when v is not backed by an init_window_sim() window (e.g. headless).
 * @param[in,out] v     Viewer initialised by init_window_sim().
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
 * @ingroup grp_robot
 * Zero all Robot fields.  Does not free model or data; call destroy_scene() for that.
 * @param[in,out] r  Robot to tear down.
 */
void cleanup(Robot *r);

/**
 * @ingroup grp_env
 * Destroy model/data owned by Env and clear borrowed Robot registrations.
 * Registered Robot objects are not deleted.
 */
void cleanup(Env *env);

/**
 * @ingroup grp_viewer
 * Release the GLFW window and MuJoCo visualization contexts owned by v.
 * @param[in,out] v  Viewer to tear down; all pointers set to null afterwards.
 */
void cleanup(Viewer *v);

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
 * Render the current simulation state and write one frame to the video stream.
 * Call mj_step() (or equivalent) before each record_frame() call.
 *
 * @param vr    VideoRecorder initialised by init_video_recorder().
 * @param model MuJoCo model.
 * @param data  MuJoCo data (current state).
 * @return true on success; false on render or pipe write error.
 */
bool record_frame(VideoRecorder *vr, mjModel *model, mjData *data);

/**
 * @ingroup grp_recorder
 * Flush the ffmpeg pipe, finalise the MP4, and release all EGL resources.
 * After this call vr->_impl is null and the VideoRecorder may be discarded.
 *
 * @param vr  VideoRecorder to tear down.
 */
void cleanup(VideoRecorder *vr);

/**
 * @ingroup grp_robot
 * Advance one physics timestep.
 *
 * Headless (no viewer active): calls mj_step() and returns true.
 * GUI (init_window_sim() was called): advances physics, renders, syncs to real
 * time, and polls GLFW events -- exactly what tick() used to do.  Returns false
 * once the user closes the window.
 *
 * This replaces the old headless/GUI split:
 *
 *   // Before:
 *   if (headless) { mj_kdl::step(&r); }
 *   else if (!mj_kdl::tick(&v, m, d)) break;
 *
 *   // After:
 *   if (!mj_kdl::step(&r)) break;
 *
 * Coupling note: in GUI mode the keyboard and mouse perturbation callbacks
 * operate on the Robot registered via init_window_sim().  Always pass the
 * same Robot to both init_window_sim() and step(); passing a different Robot
 * causes perturbation forces to be applied to the wrong model.
 *
 * @param[in,out] s  Simulation state; must be the Robot passed to init_window_sim().
 * @return true while the window is open (or always true in headless mode).
 */
bool step(Robot *s);

/**
 * @ingroup grp_robot
 * Advance the simulation by n timesteps.
 * Returns false immediately if the viewer window is closed mid-sequence.
 * @param[in,out] s  Simulation state.
 * @param[in]     n  Number of steps.
 */
bool step_n(Robot *s, int n);

/**
 * @ingroup grp_viewer
 * Model/data overload of step() for multi-robot or no-robot GUI loops.
 * Equivalent to the former tick(Viewer*, mjModel*, mjData*).
 * @param[in,out] v  Viewer initialised by init_window() or init_window_sim().
 * @param[in]     m  Shared MuJoCo model.
 * @param[in]     d  Shared MuJoCo data.
 * @return true while the window is open; false once the user closes it.
 */
bool step(Viewer *v, mjModel *m, mjData *d);

/**
 * @ingroup grp_viewer
 * Returns true if the viewer window is open and not scheduled for closing.
 * @param[in] v  Viewer created by init_window().
 */
bool is_running(const Viewer *v);

/**
 * @ingroup grp_viewer
 * Render the current simulation frame to the viewer window.
 * @param[in,out] v  Viewer created by init_window().
 * @param[in]     r  Robot whose model and data are rendered.
 * @return true if the window is still open after rendering.
 */
bool render(Viewer *v, const Robot *r);

/**
 * @ingroup grp_viewer
 * Render the current simulation frame to the viewer window.
 * Model/data overload -- use when no single Robot owns the scene (e.g. multi-robot).
 * @param[in,out] v  Viewer created by init_window().
 * @param[in]     m  MuJoCo model.
 * @param[in]     d  MuJoCo data.
 * @return true if the window is still open after rendering.
 */
bool render(Viewer *v, mjModel *m, mjData *d);

/**
 * @ingroup grp_robot
 * One control cycle: read MuJoCo into *_msr, then apply *_cmd to MuJoCo.
 * Read step: qpos -> jnt_pos_msr, qvel -> jnt_vel_msr, qfrc_actuator -> jnt_trq_msr.
 * Apply step: POSITION -> data->ctrl,
 *             TORQUE   -> qfrc_applied; also sets ctrl = qpos to neutralize
 *             position actuators (zeroes kp*(ctrl-qpos) restoring force).
 * Joints with kdl_to_mj_ctrl[i] == -1 are skipped for ctrl writes.
 */
void update(Robot *r);

/**
 * @ingroup grp_robot
 * Write KDL joint positions into MuJoCo qpos (KDL chain order -> MuJoCo addresses).
 * @param[in,out] r            Robot with a valid data pointer.
 * @param[in]     q            Joint positions in KDL chain order; size must equal r->n_joints.
 * @param[in]     call_forward If true (default), calls mj_forward() after writing qpos
 *                             so that body poses and sensor data are updated immediately.
 */
void set_joint_pos(Robot *r, const KDL::JntArray &q, bool call_forward = true);

/**
 * @ingroup grp_robot
 * Teleport a free-floating body to a world-frame position and optionally a
 * world-frame orientation, then zero its velocity.
 * body_name must identify a body that owns a mjJNT_FREE joint.
 * quat is MuJoCo convention [w, x, y, z]; pass nullptr to keep identity orientation.
 * @param[in,out] model      MuJoCo model.
 * @param[in,out] data       MuJoCo data.
 * @param[in]     body_name  Name of the free-floating body to teleport.
 * @param[in]     pos        World-frame position [x, y, z].
 * @param[in]     quat       World-frame orientation [w, x, y, z], or nullptr for identity.
 */
void set_body_pose(
  mjModel      *model,
  mjData       *data,
  const char   *body_name,
  const double  pos[3],
  const double *quat = nullptr
);

/**
 * @ingroup grp_scene
 * Add an object to the scene by appending it to spec->objects and rebuilding
 * the model. The old model/data are freed; new ones replace them.
 * Any Robot handles sharing the old model/data become stale  - call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; obj is appended to spec->objects.
 * @param[in]     obj    Object to add.
 * @return true on success; model/data and spec->objects unchanged on failure.
 */
bool scene_add_object(mjModel **model, mjData **data, SceneSpec *spec, const SceneObject &obj);

/**
 * @ingroup grp_scene
 * Env overload: adds obj, rebuilds, and re-initialises all robots registered in env.
 * env->model, env->data, and each Robot's model/data pointers are updated automatically.
 * @return true on success; env unchanged on failure.
 */
bool scene_add_object(Env *env, const SceneObject &obj);

/**
 * @ingroup grp_scene
 * Remove a named object from the scene by erasing it from spec->objects and
 * rebuilding the model. The old model/data are freed; new ones replace them.
 * Any Robot handles sharing the old model/data become stale  - call init_robot()
 * again on the new model/data after this call.
 * @param[in,out] model  Current model pointer; updated to new model on success.
 * @param[in,out] data   Current data pointer; updated to new data on success.
 * @param[in,out] spec   Scene spec; named object removed from spec->objects.
 * @param[in]     name   Name of the object to remove.
 * @return true on success; false if name not found or rebuild fails.
 */
bool scene_remove_object(mjModel **model, mjData **data, SceneSpec *spec, const std::string &name);

/**
 * @ingroup grp_scene
 * Env overload: removes the named object, rebuilds, and re-initialises all robots registered
 * in env. env->model, env->data, and each Robot's model/data pointers are updated automatically.
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
 * Read a named MuJoCo site as a world-frame KDL frame.
 * Calls mj_forward() before reading site_xpos/site_xmat.
 */
bool get_site_frame(const mjModel *model, mjData *data, const char *site_name, KDL::Frame *out);

/**
 * @ingroup grp_scene
 * Read a named MuJoCo body as a world-frame KDL frame.
 * Calls mj_forward() before reading xpos/xmat.
 */
bool get_body_frame(const mjModel *model, mjData *data, const char *body_name, KDL::Frame *out);

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
 * Works for both init_window() and init_window_sim() paths.
 * Pass nullptr or an empty string to return to the free camera.
 * @return true if the camera name was found; false if not found (viewer unchanged).
 */
bool use_camera(Viewer *v, const mjModel *model, const char *name);

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
 * Corresponds to SceneSpec::add_floor.
 */
void add_floor_to_spec(mjSpec *spec);

/**
 * @ingroup grp_advanced
 * Add free-floating or fixed rigid bodies to the world body of spec.
 * @param spec     MuJoCo spec to modify.
 * @param objects  List of objects to add.
 */
void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects);

/**
 * @ingroup grp_advanced
 * Compile spec into a model and create its data buffer.
 * spec is always deleted (on success and failure).
 * @param[in]  spec       MuJoCo spec to compile; always freed by this call.
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
