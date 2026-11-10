/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <mujoco/mujoco.h>
#include <kdl/chain.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mj_kdl {

/**
 * @ingroup grp_logging
 * Least severe message printed: INFO prints everything (default), WARN warnings and errors,
 * ERROR errors only, NONE nothing.
 */
enum class LogLevel { INFO = 0, WARN = 1, ERROR = 2, NONE = 3 };

/** @ingroup grp_logging
 *  Library-wide log threshold. */
inline LogLevel g_log_level = LogLevel::INFO;

/** @ingroup grp_logging
 *  Set the library-wide log verbosity. */
inline void set_log_level(LogLevel level) { g_log_level = level; }
/** @ingroup grp_logging
 *  Get the library-wide log verbosity. */
inline LogLevel get_log_level() { return g_log_level; }

} // namespace mj_kdl

// Log through the library's threshold; expr may stream: MJ_LOG_INFO("count=" << n).
#define MJ_FILENAME_ (::strrchr(__FILE__, '/') ? ::strrchr(__FILE__, '/') + 1 : __FILE__)

#define MJ_LOG_(lvl_enum, color, label, expr)                         \
    do {                                                              \
        if (::mj_kdl::LogLevel::lvl_enum >= ::mj_kdl::g_log_level) {  \
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

#define MJ_LOG_INFO(expr) MJ_LOG_(INFO, "", "INFO ", expr)
#define MJ_LOG_WARN(expr) MJ_LOG_(WARN, "\033[33m", "WARN ", expr)
#define MJ_LOG_ERROR(expr) MJ_LOG_(ERROR, "\033[31m", "ERROR", expr)

// Log a failure, then return it as the Status of the function that failed.
#define MJ_FAIL(expr)                                              \
    do {                                                           \
        std::ostringstream _mj_fail;                               \
        _mj_fail << expr; /* NOLINT(bugprone-macro-parentheses) */ \
        MJ_LOG_ERROR(_mj_fail.str());                              \
        return ::mj_kdl::Status{ _mj_fail.str() };                 \
    } while (0)

namespace mj_kdl {

/**
 * @ingroup grp_types
 * Result of a call that can fail on its input: empty error = success. The error is the same
 * text the call logs.
 */
struct Status
{
    std::string error;

    explicit operator bool() const { return error.empty(); }
};

/**
 * @ingroup grp_env
 * The library's version: "X.Y.Z" (MJ_KDL_VERSION in cmake/Versions.cmake) for a build of the
 * vX.Y.Z tag or of a source package, else "X.Y.Z-dev+<short commit>", with ".dirty" when
 * tracked files had uncommitted changes.
 */
const char *version();

/**
 * @ingroup grp_types
 * Kind of element an AttachTarget names: World is the worldbody (the default); Body, Site and
 * Frame look up a named element of that type.
 */
enum class AttachKind { World, Body, Site, Frame };

/**
 * @ingroup grp_types
 * Where to attach a robot, object, or attachment in the accumulated scene spec.
 * For Site, the site's own pos/quat becomes the placement frame and the
 * accompanying pos/quat are an additional offset on top of it (matches MJCF).
 */
struct AttachTarget
{
    AttachKind  kind = AttachKind::World;
    std::string name; // ignored when kind == World
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
 * A control mode for RobotSpec::modes or AttachmentSpec::modes. joints empty = every joint the
 * robot's or attachment's own MJCF actuates. kv is the VELOCITY actuator's gain [N m s/rad],
 * required for VELOCITY.
 */
struct CtrlModeSpec
{
    CtrlMode                 mode = CtrlMode::TORQUE;
    std::vector<std::string> joints;
    double                   kv = 0.0;
};

/**
 * @ingroup grp_types
 * One link in a robot's ordered attachment chain: any MJCF body (gripper, mount, F/T sensor,
 * tool, ...). Applied in declaration order; attach_to may reference any body, site or frame of
 * the root or a prior attachment.
 * modes: control modes for its own joints, in groups of their own so it switches apart from its
 * robot; empty by default, unlike RobotSpec::modes.
 */
struct AttachmentSpec
{
    std::string               mjcf_path;
    AttachTarget              attach_to;
    std::string               prefix;                // prepended to the attachment's element names
    double                    pos[3]  = { 0, 0, 0 }; // position offset [m]
    double                    quat[4] = { 0, 0, 0, 1 }; // orientation offset [x, y, z, w]
    std::vector<CtrlModeSpec> modes;                    // {} = native mode only

    // (body1, body2) pairs build_scene() excludes from contact
    std::vector<std::pair<std::string, std::string>> contact_exclusions;
};

/**
 * @ingroup grp_types
 * One robot in a scene: a root MJCF (arm, mobile base, ...) with an ordered attachment
 * chain and a placement target.
 *
 * attach_to places the robot root (default: the worldbody), e.g.
 * { AttachKind::Site, "table_mount" } on a site of a prior scene object; pos/quat are offsets
 * in that parent frame. prefix must be unique per robot in multi-robot scenes.
 *
 * modes lists the control modes the robot offers beyond the one its own actuators give
 * (a `<position>` servo gives POSITION, a `<motor>` gives TORQUE); by default every robot also gets
 * TORQUE. build_scene() adds one actuator per extra mode on each listed joint and puts each mode
 * in its own actuator group, switched with set_control_mode(). These are the robot's own joints;
 * an attachment's take modes through AttachmentSpec::modes. With no joint list, joints that
 * cannot take modes are skipped. Actuator groups 1-30 are reserved for this: group
 * 1 + 3 * owner + mode, where the owners are the robots in order and then every attachment with
 * modes in order, ten at most. The robot's MJCF must not assign actuator groups itself.
 */
struct RobotSpec
{
    std::string                 path; // root MJCF
    std::string                 prefix;
    AttachTarget                attach_to;
    double                      pos[3]  = { 0, 0, 0 };    // offset in parent frame [m]
    double                      quat[4] = { 0, 0, 0, 1 }; // orientation offset [x, y, z, w]
    std::vector<AttachmentSpec> attachments;
    std::vector<CtrlModeSpec>   modes = { CtrlModeSpec{} }; // TORQUE; {} = native mode only
};

/** @ingroup grp_types
 *  Shape of a primitive SceneObject; build_scene() rejects one left Unspecified. */
enum class Shape { Unspecified, BOX, SPHERE, CYLINDER, ELLIPSOID, MESH };

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
 * MuJoCo's <option integrator> for SceneSpec::integrator; Robot keeps the first robot's (Gen3:
 * ImplicitFast). Cloth and other flex elasticity needs Discrete (or Euler, RK4).
 */
enum class Integrator { Robot, Euler, RK4, Implicit, ImplicitFast, Discrete };

/**
 * @ingroup grp_types
 * MuJoCo's <option solver> for SceneSpec::solver; Robot keeps the first robot's (Gen3: Newton).
 */
enum class Solver { Robot, PGS, CG, Newton };

/**
 * @ingroup grp_types
 * A free-floating or fixed rigid body to place in the scene.
 *
 * Names: an MJCF asset's elements keep their authored names, with prefix prepended when set;
 * its root body is renamed to name. The same asset used twice needs distinct prefixes, or
 * the build fails on the repeated names.
 *
 * size:
 *   BOX       - half-extents (x, y, z)
 *   SPHERE    - {radius, 0, 0}
 *   CYLINDER  - {radius, half-length, 0}
 *   ELLIPSOID - radii (x, y, z)
 *   MESH      - scale (x, y, z) applied to the mesh file's vertices
 *   Ignored when mjcf_path is set.
 *
 * mesh_path:
 *   The mesh file (STL, OBJ or MSH) of a MESH primitive; required for MESH, ignored otherwise.
 *   Added as the mesh asset `<name>_mesh`; a relative path resolves against the working
 *   directory, never against an earlier MJCF's meshdir.
 *
 * attach_to:
 *   Parent in the accumulated scene spec (default: the worldbody). A child object must appear
 *   after its parent in SceneSpec::objects. A body with a freejoint must be a direct child of
 *   the worldbody (MuJoCo), so a non-fixed primitive, and an mjcf_path asset whose root owns a
 *   freejoint, must use AttachKind::World; the build fails otherwise.
 *
 * pos:
 *   Offset in the resolved parent frame; for an MJCF asset, of its first root body.
 *
 * fixed:
 *   If true the body is welded to its parent (no freejoint). If false, an MJCF asset whose root
 *   has no joint gets a freejoint named `<name>_free`.
 *
 * size, rgba, mass and friction start unset (NaN). build_scene() fails on a primitive that
 * leaves size unset, and on a non-fixed primitive that leaves mass or friction unset. What a
 * primitive may leave unset gets MuJoCo's own default:
 *   rgba     - MuJoCo's default geom colour, on any primitive.
 *   mass     - fixed primitives only: the geom's mass follows MuJoCo's default density.
 *              Set 0 for a geom that adds no mass, e.g. one drawn on a body whose inertia
 *              is already stated.
 *   friction - fixed primitives only: MuJoCo's default geom friction.
 * A vector field is either fully set or fully unset; a partly set one is rejected.
 *
 * rgba / has_rgba:
 *   On a primitive, rgba is the colour when its four channels are set. On an MJCF asset,
 *   has_rgba makes every geom under the asset's root body take rgba, so a scene can state
 *   the colour an object is drawn in without editing the asset.
 */
struct SceneObject
{
    std::string  name;
    std::string  mjcf_path; // optional MJCF asset; when set, shape/size/mass/friction are ignored
    std::string  prefix;    // prepended to the asset's element names; empty = as authored
    AttachTarget attach_to;
    Shape        shape = Shape::Unspecified; // required for primitives
    std::string  mesh_path;
    double       size[3]     = { NAN, NAN, NAN };
    double       pos[3]      = { 0.0, 0.0, 0.0 };      // offset in resolved parent frame [m]
    double       quat[4]     = { 0.0, 0.0, 0.0, 1.0 }; // orientation offset [x, y, z, w]
    float        rgba[4]     = { NAN, NAN, NAN, NAN }; // [r, g, b, a]; unset = MuJoCo's default
    bool         has_rgba    = false;                  // rgba is set; recolours an asset's geoms
    bool         fixed       = false;
    double       mass        = NAN; // [kg]; required for non-fixed primitives
    Condim       condim      = Condim::Tangential;
    double       friction[3] = { NAN, NAN, NAN }; // [slide, spin, roll]; required unless fixed
    int          contype     = 1;                 // primitive's collision bitmask; MuJoCo's default
    int          conaffinity = 1;                 // primitive's collision bitmask; MuJoCo's default
};

/**
 * @ingroup grp_types
 * A frame to mark on a body of the assembled scene, as a MuJoCo site.
 * Added after every robot and object, so `body` may name anything the assembled scene holds.
 * A site the asset already declares under this name is left alone.
 */
struct SiteSpec
{
    std::string body;
    std::string name;                             // unique within the scene
    double      pos[3]  = { 0.0, 0.0, 0.0 };      // offset in the body frame [m]
    double      quat[4] = { 0.0, 0.0, 0.0, 1.0 }; // orientation in the body frame [x, y, z, w]
    double      size    = 0.005;                  // marker sphere radius [m]
    int         group   = 4; // visibility group, so frame sites hide without an asset's own
};

/**
 * @ingroup grp_types
 * A named fixed camera to add to the world body of the scene.
 * After build_scene() the camera is accessible by name (mj_name2id(model, mjOBJ_CAMERA, name))
 * and can be activated on a Viewer with use_camera(), or on a VideoRecorder through vr.cam.
 *
 * pos and fovy start unset (NaN) and are required: build_scene() fails without them.
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
 *  timestep (unset: NaN) and use_stage have no defaults; the caller must set both.
 *
 *  The stage is an MJCF file whose floor, sky, haze, lights and visual settings every scene
 *  is built on; the bundled assets/stage/default.xml is used when `stage` is empty. */
struct SceneSpec
{
    std::vector<RobotSpec> robots;
    double                 timestep  = NAN;   // required; suggested 0.001 [s]
    double                 gravity_z = -9.81; // [m/s^2]
    bool                   use_stage;         // required; false builds on an empty world
    std::string            stage;             // stage MJCF path; empty = the bundled default
    double                 floor_z  = 0.0;    // added to the height of the stage's planes [m]
    bool       speculative_contacts = false; // MuJoCo's speculative flag: contacts before an impact
    Integrator integrator           = Integrator::Robot; // the first robot's unless set
    Solver     solver               = Solver::Robot;     // the first robot's unless set
    std::vector<SceneObject> objects;
    std::vector<SiteSpec>    sites;
    std::vector<CameraSpec>  cameras;
};

/**
 * @ingroup grp_types
 * Logical force-torque sensor backed by MuJoCo's separate `<force>` and `<torque>`
 * sensors. If force_sensor/torque_sensor are omitted, init_robot_from_mjcf()
 * resolves "{name}_force" and "{name}_torque".
 */
struct ForceTorqueSensorSpec
{
    std::string name;          // logical wrapper name
    std::string force_sensor;  // MuJoCo <force> sensor name
    std::string torque_sensor; // MuJoCo <torque> sensor name
    std::string frame_site;    // optional site that defines the sensor frame
};

/**
 * @ingroup grp_types
 * What a force-torque sensor measures; reset() assigns a fresh one, so every field here is reset.
 *
 * The wrench is the load on the sensor, as a physical F/T sensor reports it: a tool hanging below
 * reads its weight pointing down. MuJoCo's `<force>`/`<torque>` sensors give the opposite (the
 * parent's force on the child), so the reading is their negation.
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
 * tool_body is the root of the tool subtree whose mass/inertia is lumped into the arm dynamics
 * once, at its pose at init (a 2F-85's finger opening then is what the chain keeps). tcp_site
 * becomes the KDL terminal frame for FK/IK; without one, tcp_frame gives it in the tip body's
 * frame. For the prefixed Robotiq 2F-85 this is {"g_base_mount", "g_pinch"}: the mount carries
 * mass too.
 */
struct ToolFrameSpec
{
    std::string tool_body;
    std::string tcp_site;                           // MuJoCo site name (takes priority)
    KDL::Frame  tcp_frame = KDL::Frame::Identity(); // manual TCP in tip frame (fallback)
    std::vector<ForceTorqueSensorSpec> ft_sensors;
    double tcp_identity_tol = 1e-12; // a tcp_frame this close to identity means no TCP
};


/**
 * @ingroup grp_types
 * A robot's control ports. Sized once by init_robot_*(); reset() rewrites their values in place,
 * so a pointer to an element stays valid until the robot is re-initialised or destroyed.
 */
struct RobotPorts
{
    CtrlMode             ctrl_mode = CtrlMode::POSITION;
    std::vector<double>  jnt_pos_msr;   // [rad]   measured joint positions (update())
    std::vector<double>  jnt_vel_msr;   // [rad/s] measured joint velocities (update())
    std::vector<double>  jnt_trq_msr;   // [Nm]    active mode's qfrc_actuator (update())
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
    std::string name; // set by the caller, before or after init; the viewer shows it

    // Set by init_robot_from_mjcf() / init_robot_from_chain().
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
 *  A body a controller pushes: the wrench is applied as xfrc_applied, at the body's centre
 *  of mass. */
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
 * What a component outside the core (the viewer of mj_kdl_wrapper::viewer, a recorder) does
 * when the Env it follows changes. Each hook runs on the thread driving the Env, inside the call
 * named; an unset one is skipped. Kept in Env::hooks under the component's address.
 */
struct EnvHooks
{
    /* step(), before it integrates: may write the state and commands, and clears *integrate to
     * hold physics this step. Returns false once the component has stopped (a window closed),
     * which step() returns in turn. */
    std::function<bool(Env *env, bool *integrate)> on_step;
    // scene_add_object() / scene_remove_object(), after env->model and env->data were replaced.
    std::function<void(Env *env)> on_rebuild;
    // reset(), after the robots and scene slots were re-seeded and before Env::on_reset.
    std::function<void(Env *env)> on_reset;
    // cleanup(Env *) and init_env(): the Env is going away; Env::hooks is cleared right after.
    std::function<void(Env *env)> on_close;
};

/** @ingroup grp_env
 * The simulation: the compiled scene and everything that reads or commands it.
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
    SceneState           scene; // bind slots with bind_scene_*(&env.scene, ...)
    ResetHook            on_reset;
    // Takes each compiled pair; returns the pair to run on, which the Env never frees.
    std::function<std::pair<mjModel *, mjData *>(mjModel *, mjData *)> adopt;
    /* Components following the Env, keyed by their own address. A hook may add or erase entries
     * while it runs; each entry is looked up again before it is called. */
    std::map<const void *, EnvHooks> hooks;
    std::size_t          overlay_max_geoms = 8192; // overlay geoms beyond this are dropped
    std::array<float, 4> overlay_rgba      = { 1.0f, 0.5f, 0.1f, 1.0f }; // when a call gives none

    Env();
    ~Env();
    Env(const Env &)            = delete;
    Env &operator=(const Env &) = delete;

    std::unique_ptr<EnvInternals> _impl; // internal
};

/**
 * @ingroup grp_scene
 * Save a model to an MJCF XML file, including runtime changes to its real-valued fields.
 * Works for any live model from build_scene() or init_env() (adopted ones too), and for the
 * last model loaded with mj_loadXML.
 * @param model  Model to save.
 * @param path   Output path for the MJCF XML file.
 * @return an empty Status on success, else the error.
 */
Status save_model_xml(const mjModel *model, const char *path);

/**
 * @ingroup grp_robot
 * Build KDL chain from a compiled MuJoCo model and optional tool/TCP metadata.
 *
 * If tool->tcp_site is set, that authored site becomes the KDL terminal frame
 * for FK/IK.  The joint count and MuJoCo joint/actuator maps still cover only
 * the controllable joints from base_body to tip_body.
 * Pass tool = nullptr (the default) for an arm with no attached tool.
 * prefix is prepended to every name the call resolves: base_body, tip_body, the tool body,
 * TCP site and F/T sensor names, e.g. ("base_link", "bracelet_link", "r2_") is the second arm.
 * Registers r with env, which reads, commands and resets it from then on; on failure r is left
 * as it was.
 */
Status init_robot_from_mjcf(
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
 * Same as init_robot_from_mjcf(), except the chain is supplied rather than derived from the
 * model, and taken as given: no tool inertia is lumped onto it.
 *
 * joint_names lists the MuJoCo joint names in KDL chain order, one per chain joint. prefix is
 * prepended to each joint name and F/T sensor name. tool is used only to resolve FT sensors;
 * tool->tool_body and tool->tcp_site are ignored. Registers r with env, as init_robot_from_mjcf().
 */
Status init_robot_from_chain(
  Robot                          *r,
  Env                            *env,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix = "",
  const ToolFrameSpec            *tool   = nullptr
);

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
 * Build a MuJoCo scene from one or more robots using the MuJoCo spec API; no intermediate XML
 * files are written.
 *
 * @param[out] out_model  Newly allocated MuJoCo model; caller frees via destroy_scene().
 * @param[out] out_data   Newly allocated MuJoCo data; caller frees via destroy_scene().
 * @param[in]  spec       Scene description: robots (with attachment chains), objects,
 *                        sites, cameras, timestep, gravity, floor, skybox.
 * @return an empty Status on success, else the error.
 */
Status build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *spec);

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
 * Env::on_reset and Env::adopt set beforehand are kept.
 */
Status init_env(Env *env, const SceneSpec *spec);

/**
 * @ingroup grp_env
 * Reset everything env holds: MuJoCo data to the keyframe (or the model default); every
 * registered robot's ports (holding the reset pose, a requested ctrl_mode kept) and F/T readings,
 * and every scene slot, re-seeded; then Env::on_reset; then measurements read from the result.
 * The viewer's reset button does the same.
 */
ResetInfo reset(Env *env, const ResetOptions *options = nullptr);

/**
 * @ingroup grp_overlay
 * Remove every overlay geom of env: trace segments and arrows alike.
 * Call once per frame before appending this frame's overlays. Thread-safe.
 */
void clear_trace(Env *env);

/**
 * @ingroup grp_overlay
 * Append a world-space line segment to env's overlay geoms, which the viewer and every
 * recorder draw on top of the model. Thread-safe. Silently dropped once the overlay holds
 * its maximum of geoms.
 * @param[in,out] env   Env the overlay belongs to.
 * @param[in]     a     Segment start point (world frame) [m].
 * @param[in]     b     Segment end point (world frame) [m].
 * @param[in]     rgba  Optional [r, g, b, a] colour; nullptr -> warm orange.
 */
void add_trace_segment(
  Env               *env,
  const KDL::Vector &a,
  const KDL::Vector &b,
  const float        rgba[4] = nullptr
);

/**
 * @ingroup grp_overlay
 * Append a world-space arrow to env's overlay geoms, as add_trace_segment() does.
 * dir need not be normalised; a zero-length dir draws nothing.
 * @param[in,out] env     Env the overlay belongs to.
 * @param[in]     from    Arrow tail (world frame) [m].
 * @param[in]     dir     Direction the arrow points.
 * @param[in]     length  Arrow length [m].
 * @param[in]     rgba    Optional [r, g, b, a] colour; nullptr -> warm orange.
 */
void add_overlay_arrow(
  Env               *env,
  const KDL::Vector &from,
  const KDL::Vector &dir,
  double             length,
  const float        rgba[4] = nullptr
);

/**
 * @ingroup grp_overlay
 * A copy of env's overlay geoms, for whatever draws them. Thread-safe.
 */
std::vector<mjvGeom> overlay_geoms(const Env *env);

/**
 * @ingroup grp_robot
 * Unregister r from its Env and clear it. The Env keeps running without it.
 */
void cleanup(Robot *r);

/**
 * @ingroup grp_env
 * Close every component following env (EnvHooks::on_close), free the model/data and forget the
 * robots (which are not deleted).
 */
void cleanup(Env *env);

/**
 * @ingroup grp_env
 * Advance env one timestep: mj_step2() then mj_step1() (MuJoCo's split mj_step()). Afterwards
 * joint state, body frames and position/velocity sensors all describe the new state; force and
 * acceleration sensors describe the step just taken. Each EnvHooks::on_step runs first and may
 * hold physics (a paused viewer); nothing integrates while every robot is paused either.
 * @return false once a component following env has stopped (its window closed), else true.
 */
bool step(Env *env);

/**
 * @ingroup grp_env
 * One control cycle for everything env holds: read, then apply.
 * Robots: qpos -> jnt_pos_msr, qvel -> jnt_vel_msr, qfrc_actuator -> jnt_trq_msr (only the active
 * mode's actuators produce force, so this is the drive torque in every mode), F/T wrenches; then
 * the active mode's command to its actuators' ctrl (gear applied, ctrlrange clamped). If
 * ctrl_mode was changed directly, the switch runs first (set_control_mode()). Outside TORQUE,
 * gravity on the bodies a robot's joints carry is compensated (body_gravcomp), so a servo holds
 * its setpoint; in TORQUE the command carries gravity itself.
 * Scene slots: joints and free bodies sampled from qpos/qvel; wrenches into xfrc_applied (zeros
 * included), actuator commands into ctrl.
 */
void update(Env *env);

/**
 * @ingroup grp_robot
 * Switch the robot to mode: seed the new actuators so nothing jumps, enable their group,
 * disable the robot's other mode groups. @return an error if the robot has no actuator for mode.
 */
Status set_control_mode(Robot *r, CtrlMode mode);

/**
 * @ingroup grp_robot
 * The same switch for a robot driven outside a Robot chain (e.g. through env.scene):
 * robot is its index in SceneSpec::robots. @return an error if it has no actuators for mode.
 */
Status set_control_mode(Env *env, int robot, CtrlMode mode);

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
 * quat is [x, y, z, w] like every quaternion in this API; nullptr sets identity.
 */
void set_body_pose(
  Env          *env,
  const char   *body_name,
  const double  pos[3],
  const double *quat = nullptr
);

/**
 * @ingroup grp_scene
 * Append obj to env->spec.objects and rebuild. Time, and qpos/qvel/act/ctrl of every joint and
 * actuator whose name survives, carry over; robots, scene slots, recorders and the viewer follow
 * the new model; a slot whose name is gone is unbound and skipped.
 * @return an error on failure (a robot or F/T sensor that no longer resolves), env unchanged.
 */
Status scene_add_object(Env *env, const SceneObject &obj);

/**
 * @ingroup grp_scene
 * Remove the named object from env->spec.objects and rebuild, as scene_add_object() does.
 * @return an error if name is not found or the rebuild fails, env unchanged.
 */
Status scene_remove_object(Env *env, const std::string &name);

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

} // namespace mj_kdl
