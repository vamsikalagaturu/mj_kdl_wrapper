/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "simulate.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#include "glfw_adapter.h"

#ifdef MJ_KDL_HAS_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif
#ifdef MJ_KDL_RELOCATABLE_PLUGINS
#include <dlfcn.h>
#endif

#include <kdl/frames.hpp>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace mj_kdl {

static std::string default_mujoco_plugin_dir()
{
#ifdef MJ_KDL_RELOCATABLE_PLUGINS
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&default_mujoco_plugin_dir), &info) && info.dli_fname) {
        std::string library_path(info.dli_fname);
        const auto  slash = library_path.find_last_of('/');
        if (slash != std::string::npos) { return library_path.substr(0, slash) + "/mujoco_plugin"; }
    }
#endif
    return MUJOCO_PLUGIN_DIR;
}

void ensure_plugins_loaded()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        /* MuJoCo's plugin registry is global to the loaded libmujoco. When this
         * library shares that libmujoco with the official mujoco Python package,
         * importing mujoco already registers the bundled plugins; loading them
         * again is a fatal "plugin already registered" error. Only load when the
         * registry is still empty (e.g. standalone C++ use). */
        if (mjp_pluginCount() > 0) return;
        const char       *env      = std::getenv("MUJOCO_PLUGIN_DIR");
        const std::string fallback = default_mujoco_plugin_dir();
        const char       *dir      = env ? env : fallback.c_str();
        mj_loadAllPluginLibraries(dir, nullptr);
    });
}

// A Robot's MuJoCo addresses in KDL joint order, and its mode-group bookkeeping.
struct RobotInternals
{
    Env             *env = nullptr;
    std::vector<int> kdl_to_mj_qpos;
    std::vector<int> kdl_to_mj_dof;
    std::vector<int> kdl_to_mj_ctrl;    // -1 if none
    std::vector<int> mode_ctrl[3];      // per CtrlMode: the joint's actuator, -1 if none
    int              robot_index  = -1; // SceneSpec::robots index owning mode groups; -1 none
    CtrlMode         applied_mode = CtrlMode::POSITION; // mode whose group is enabled
    bool             mode_applied = false;              // applied_mode is live in the model
    std::string      mj_prefix;                         // re-resolves the joints after a rebuild
};

struct EnvInternals
{
    // What the last mj_step1/mj_forward read; its results stay current while the state matches.
    std::vector<mjtNum>                  computed_from;
    bool                                 computed = false;
    std::unordered_map<std::string, int> names[mjNOBJECT]; // mj_name2id per object type
};

Robot::Robot() : _impl(std::make_unique<RobotInternals>()) {}
Robot::~Robot() { cleanup(this); }
Env::Env() : _impl(std::make_unique<EnvInternals>()) {}
Env::~Env() { cleanup(this); }

// Held while touching env's mjData, which the Simulate render thread also reads.
static std::unique_lock<std::recursive_mutex> lock_env(const Env *env);

/* Poses and position/velocity sensors (what mj_step1 computes, and mj_forward with it) are
 * recomputed only when the state they came from has changed. Compared, not trusted: a caller
 * may write qpos, qvel or mocap between two calls. */
static constexpr int kKinematicsInputs =
  mjSTATE_FULLPHYSICS | mjSTATE_EQ_ACTIVE | mjSTATE_MOCAP_POS | mjSTATE_MOCAP_QUAT;

static bool kinematics_current(const Env *env)
{
    const EnvInternals &in = *env->_impl;
    if (!in.computed) return false;
    static std::vector<mjtNum> now;
    now.resize(mj_stateSize(env->model, kKinematicsInputs));
    mj_getState(env->model, env->data, now.data(), kKinematicsInputs);
    return now == in.computed_from;
}

static void record_kinematics(Env *env)
{
    EnvInternals &in = *env->_impl;
    in.computed_from.resize(mj_stateSize(env->model, kKinematicsInputs));
    mj_getState(env->model, env->data, in.computed_from.data(), kKinematicsInputs);
    in.computed = true;
}

static void ensure_kinematics(Env *env)
{
    if (kinematics_current(env)) return;
    mj_forward(env->model, env->data);
    record_kinematics(env);
}

// mj_step1: frames and position/velocity sensors for the current state.
static void begin_step(Env *env)
{
    if (kinematics_current(env)) return;
    mj_step1(env->model, env->data);
    record_kinematics(env);
}

// mj_step2: integrate the commands set since begin_step().
static void end_step(Env *env)
{
    begin_step(env);
    mj_step2(env->model, env->data);
    env->_impl->computed = false;
}

static int cached_name2id(Env *env, mjtObj type, const char *name)
{
    auto      &names = env->_impl->names[type];
    const auto found = names.find(name);
    if (found != names.end()) return found->second;
    const int id = mj_name2id(env->model, type, name);
    names.emplace(name, id);
    return id;
}


// Spec-API helpers

// Buffer size for MuJoCo error strings returned by mj_parseXML / mj_saveLastXML.
static constexpr size_t kMjErrBuf = 2048;

// Default scene-decoration constants used by add_floor_to_spec /
// add_skybox_to_spec. Keep here so callers reading the code see one source.
static constexpr int    kFloorHalfSize    = 10;   // floor plane half-extents [m]
static constexpr double kFloorThickness   = 0.05; // floor geom thickness   [m]
static constexpr int    kFloorTexSize     = 300;  // checker texture size   [px]
static constexpr int    kFloorTexRepeat   = 5;    // checker tile repeat
static constexpr float  kFloorReflectance = 0.2f; // floor material reflectance
static constexpr int    kSkyTexSize       = 200;  // skybox gradient texture size [px]
static constexpr double kSunHeight        = 4.0;  // overhead directional light z [m]
// Frame sites are markers, not geometry: small, and in their own group so the viewer can
// switch them off without hiding the sites an asset declares for attaching.
static constexpr double kFrameSiteSize  = 0.005; // frame-site marker radius [m]
static constexpr int    kFrameSiteGroup = 4;     // viewer group for frame sites

// Numerical tolerances. Frame-equality test treats relative deviations below
// kIdentityTol as zero; sim-time guard rejects steps shorter than kSimTimeEps
// caused by floating-point round-trip through MuJoCo.
static constexpr double kIdentityTol = 1e-12;
static constexpr double kSimTimeEps  = 1e-9;

// Default MuJoCo collision category bitmask for wrapper-authored geoms (sets
// category bit 0, matching MuJoCo's MJCF compiler defaults for contype and
// conaffinity). This is a bitmask, not a boolean.
static constexpr int kContactCategoryAll = 1;

// Pixel layout for the offscreen recorder (RGB24 / ffmpeg rgb24 input).
static constexpr int kRgbBytesPerPixel = 3;

// Defaults applied when the Simulate UI omits recorder fields.
static constexpr int kRecorderDefaultResIndex = 2; // 720p (see recorder_resolution_from_index)
static constexpr int kRecorderDefaultFps      = 30;

/* An ffmpeg process taking raw RGB24 on stdin and writing H.264 MP4. Both recordings feed one:
   the offscreen renderer, and the window capture that reads back the frame already drawn. */
struct FfmpegSink
{
    FILE       *pipe = nullptr;
    std::string path;
    int         width  = 0;
    int         height = 0;
    int         frames = 0;
};

/* Start ffmpeg reading in_w x in_h frames. An out_w x out_h that differs is scaled by ffmpeg,
   so a view of any size can be recorded at the resolution that was asked for.

   `flip` is for frames straight out of mjr_readPixels, which MuJoCo fills bottom-to-top: the
   filter chain turns them over, as the record sample in the MuJoCo docs does, rather than the
   caller walking every frame to swap its rows. */
static bool sink_open(
  FfmpegSink *sink,
  const char *out_path,
  int         in_w,
  int         in_h,
  int         out_w,
  int         out_h,
  int         fps,
  bool        flip
)
{
    // -g <fps>: ~1s GOP (vs x264's 250-frame default) so scrubbing seeks land near a keyframe.
    const int gop = std::max(1, fps);

    /* libx264 in yuv420p refuses an odd width or height, and both a window's size and a 16:9
       width for an odd height can be odd, so drop the stray row or column here. */
    const int target_w = ((out_w > 0 ? out_w : in_w) / 2) * 2;
    const int target_h = ((out_h > 0 ? out_h : in_h) / 2) * 2;
    if (target_w <= 0 || target_h <= 0) {
        LOG_ERROR("recording: frame is " << in_w << "x" << in_h << ", too small to encode");
        return false;
    }

    const bool scaling    = target_w != in_w || target_h != in_h;
    char       chain[160] = "";
    if (flip && scaling) {
        snprintf(chain, sizeof(chain), "vflip,scale=%d:%d", target_w, target_h);
    } else if (flip) {
        snprintf(chain, sizeof(chain), "vflip");
    } else if (scaling) {
        snprintf(chain, sizeof(chain), "scale=%d:%d", target_w, target_h);
    }
    char filters[192] = "";
    if (chain[0]) snprintf(filters, sizeof(filters), "-vf \"%s\" ", chain);

    char cmd[2048];
    snprintf(
      cmd,
      sizeof(cmd),
      "ffmpeg -hide_banner -loglevel error -nostats -y "
      "-f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s %dx%d -r %d "
      "-i pipe:0 -an %s-vcodec libx264 -pix_fmt yuv420p -preset medium -crf 18 -g %d \"%s\"",
      in_w,
      in_h,
      fps,
      filters,
      gop,
      out_path
    );
    /* An ffmpeg that dies mid-recording must not take the simulation down with it: without this
       the next write raises SIGPIPE, whose default is to kill the process. */
    std::signal(SIGPIPE, SIG_IGN);

    sink->pipe   = popen(cmd, "w");
    sink->path   = out_path;
    sink->width  = in_w;
    sink->height = in_h;
    sink->frames = 0;
    if (!sink->pipe) {
        LOG_ERROR("popen(ffmpeg) failed - is ffmpeg installed and in PATH?");
        return false;
    }
    return true;
}

static bool sink_write(FfmpegSink *sink, const std::uint8_t *rgb, std::size_t bytes)
{
    if (!sink->pipe) return false;
    if (fwrite(rgb, 1, bytes, sink->pipe) != bytes) {
        LOG_ERROR("fwrite to ffmpeg pipe failed");
        return false;
    }
    ++sink->frames;
    return true;
}

static void sink_close(FfmpegSink *sink)
{
    if (!sink->pipe) return;
    const int status = pclose(sink->pipe);
    sink->pipe       = nullptr;
    if (status == 0 && sink->frames > 0) {
        std::fprintf(stderr, "[mj_kdl] recording saved to %s\n", sink->path.c_str());
    } else if (status != 0) {
        LOG_ERROR("ffmpeg failed while saving recording to '" << sink->path << "'");
    }
}

#ifdef MJ_KDL_HAS_EGL
// EGL framebuffer attribute values for the headless recorder.
static constexpr EGLint kEglChannelBits = 8;
static constexpr EGLint kEglDepthBits   = 24;
#endif

// Sole owner of the MuJoCo worldbody name. Every other site that needs the
// worldbody (skybox light, floor geom, resolve_parent with kind == World,
// cameras) calls this so the literal "world" exists in one place.
// The name is MuJoCo's, not ours to choose: its compiler decides what is top level by this
// name, so renaming the root body makes every free joint under it fail to compile.
static mjsBody *world_body(mjSpec *spec) { return mjs_findBody(spec, "world"); }

// MuJoCo quaternion [w,x,y,z] <-> KDL::Rotation [x,y,z,w]. Single source so
// the w/x/y/z swap is never re-derived inline.
static KDL::Rotation mj_quat_to_kdl_rot(const double *q)
{
    return KDL::Rotation::Quaternion(q[1], q[2], q[3], q[0]);
}

static void kdl_rot_to_mj_quat(const KDL::Rotation &R, double q[4])
{
    double qx, qy, qz, qw;
    R.GetQuaternion(qx, qy, qz, qw);
    q[0] = qw;
    q[1] = qx;
    q[2] = qy;
    q[3] = qz;
}

static KDL::Rotation mj_xmat_to_kdl_rot(const double *m)
{
    return KDL::Rotation(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);
}

// RAII guard for transient mjSpec pointers built/parsed inside scene-building
// functions. unique_ptr handles every early-return error path automatically.
using MjSpecPtr = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;
static MjSpecPtr make_spec_ptr(mjSpec *s) { return { s, &mj_deleteSpec }; }

/* A parsed spec's asset files are relative to its own file and meshdir/texturedir, which do not
 * survive mjs_attach: resolve them now, or a scene saved with mj_saveXML cannot be reloaded. */
static void absolutize_asset_files(mjSpec *spec)
{
    // Relative when the model path was, and a relative file does not resolve from the scene.
    const std::filesystem::path base = std::filesystem::absolute(mjs_getString(spec->modelfiledir));
    const auto resolve = [&base](mjString *file, const mjString *dir) {
        const std::filesystem::path f = mjs_getString(file);
        if (f.empty() || f.is_absolute()) return;
        const std::filesystem::path d = mjs_getString(dir);
        mjs_setString(file, ((d.is_absolute() ? d : base / d) / f).lexically_normal().c_str());
    };
    for (mjsElement *e = mjs_firstElement(spec, mjOBJ_MESH); e; e = mjs_nextElement(spec, e))
        resolve(mjs_asMesh(e)->file, spec->compiler.meshdir);
    for (mjsElement *e = mjs_firstElement(spec, mjOBJ_HFIELD); e; e = mjs_nextElement(spec, e))
        resolve(mjs_asHField(e)->file, spec->compiler.meshdir);
    for (mjsElement *e = mjs_firstElement(spec, mjOBJ_TEXTURE); e; e = mjs_nextElement(spec, e))
        resolve(mjs_asTexture(e)->file, spec->compiler.texturedir);
}

// Resolve an AttachTarget to its element in the accumulated spec.
// Returns nullptr (and logs) when a non-World name is missing.
static mjsElement *resolve_parent(mjSpec *spec, const AttachTarget &t)
{
    switch (t.kind) {
    case AttachKind::World: {
        mjsBody *wb = world_body(spec);
        return wb ? wb->element : nullptr;
    }
    case AttachKind::Body: {
        if (!t.name) {
            LOG_ERROR("AttachTarget kind=Body has null name");
            return nullptr;
        }
        mjsBody *b = mjs_findBody(spec, t.name);
        if (!b) LOG_ERROR("attach parent body '" << t.name << "' not found");
        return b ? b->element : nullptr;
    }
    case AttachKind::Site: {
        if (!t.name) {
            LOG_ERROR("AttachTarget kind=Site has null name");
            return nullptr;
        }
        mjsElement *e = mjs_findElement(spec, mjOBJ_SITE, t.name);
        if (!e) LOG_ERROR("attach parent site '" << t.name << "' not found");
        return e;
    }
    case AttachKind::Frame: {
        if (!t.name) {
            LOG_ERROR("AttachTarget kind=Frame has null name");
            return nullptr;
        }
        mjsFrame *f = mjs_findFrame(spec, t.name);
        if (!f) LOG_ERROR("attach parent frame '" << t.name << "' not found");
        return f ? f->element : nullptr;
    }
    }
    return nullptr;
}

// Reorder a [x, y, z, w] quaternion into MuJoCo's [w, x, y, z].
static void quat_xyzw_to_mj_quat(const double q[4], double out_quat[4])
{
    out_quat[0] = q[3];
    out_quat[1] = q[0];
    out_quat[2] = q[1];
    out_quat[3] = q[2];
}

// Attach a child body element under the resolved parent at the given offset.
// For body parents, an intermediate mjsFrame carries the offset so the child's
// authored pos/quat is preserved. For site/frame parents (which do not accept
// mjs_addFrame), the offset is written into the child root's pos/quat.
// Returns the attached body element in the scene spec on success (so callers
// can rename it or inspect it), or nullptr on failure.
static mjsBody *attach_child(
  mjSpec             *spec,
  const AttachTarget &target,
  const double        pos[3],
  const double        quat[4],
  mjsBody            *child_root,
  const char         *prefix
)
{
    if (!spec || !child_root) return nullptr;
    mjsElement *parent = resolve_parent(spec, target);
    if (!parent) return nullptr;

    double mj_quat[4];
    quat_xyzw_to_mj_quat(quat, mj_quat);

    mjsElement *attach_parent = parent;
    if (mjsBody *body_parent = mjs_asBody(parent)) {
        // Intermediate frame carries the user offset; child keeps its authored pose.
        mjsFrame *frame = mjs_addFrame(body_parent, nullptr);
        frame->pos[0]   = pos[0];
        frame->pos[1]   = pos[1];
        frame->pos[2]   = pos[2];
        for (int i = 0; i < 4; ++i) frame->quat[i] = mj_quat[i];
        attach_parent = frame->element;
    } else {
        // Site/frame parents do not accept mjs_addFrame, so the offset has to
        // ride on the child root. Compose user_offset * child_authored so the
        // child's MJCF-authored pos/quat is not silently dropped.
        KDL::Frame user(mj_quat_to_kdl_rot(mj_quat), KDL::Vector(pos[0], pos[1], pos[2]));
        KDL::Frame child(
          mj_quat_to_kdl_rot(child_root->quat),
          KDL::Vector(child_root->pos[0], child_root->pos[1], child_root->pos[2])
        );
        KDL::Frame composed = user * child;
        child_root->pos[0]  = composed.p.x();
        child_root->pos[1]  = composed.p.y();
        child_root->pos[2]  = composed.p.z();
        kdl_rot_to_mj_quat(composed.M, child_root->quat);
    }

    const char *pfx = prefix ? prefix : "";
    // The child's unnamed top-level default class is written back nameless by mj_saveXML, which
    // mj_loadXML then rejects; give it the child model's name.
    mjSpec     *child     = mjs_getSpec(child_root->element);
    mjsDefault *child_def = child ? mjs_getSpecDefault(child) : nullptr;
    if (child_def && child->modelname) {
        mjs_setName(child_def->element, (std::string(pfx) + mjs_getString(child->modelname)).c_str());
    }
    mjsElement *attached = mjs_attach(attach_parent, child_root->element, pfx, "");
    if (!attached) {
        LOG_ERROR("mjs_attach failed: " << mjs_getError(spec));
        return nullptr;
    }
    return mjs_asBody(attached);
}

// Extract the first root body from a freshly parsed or built mjSpec
// (i.e. the first body child of its worldbody).
static mjsBody *first_root_body(mjSpec *spec)
{
    mjsBody    *wb    = world_body(spec);
    mjsElement *first = wb ? mjs_firstChild(wb, mjOBJ_BODY, 0) : nullptr;
    return first ? mjs_asBody(first) : nullptr;
}

void add_skybox_to_spec(mjSpec *spec)
{
    mjsBody *wb = world_body(spec);

    mjsTexture *sky = mjs_addTexture(spec);
    mjs_setString(mjs_getName(sky->element), "skybox");
    sky->type    = mjTEXTURE_SKYBOX;
    sky->builtin = mjBUILTIN_GRADIENT;
    sky->rgb1[0] = 0.3f;
    sky->rgb1[1] = 0.45f;
    sky->rgb1[2] = 0.65f; // top: mid blue
    sky->rgb2[0] = 0.65f;
    sky->rgb2[1] = 0.80f;
    sky->rgb2[2] = 0.95f; // bottom: pale blue
    sky->width   = kSkyTexSize;
    sky->height  = kSkyTexSize;

    mjsLight *sun = mjs_addLight(wb, nullptr);
    sun->type     = mjLIGHT_DIRECTIONAL;
    sun->pos[0]   = 0;
    sun->pos[1]   = 0;
    sun->pos[2]   = kSunHeight;
}

void add_floor_to_spec(mjSpec *spec, double floor_z)
{
    mjsBody *wb = world_body(spec);

    mjsTexture *tex = mjs_addTexture(spec);
    mjs_setString(mjs_getName(tex->element), "groundplane");
    tex->type    = mjTEXTURE_2D;
    tex->builtin = mjBUILTIN_CHECKER;
    tex->rgb1[0] = 0.2;
    tex->rgb1[1] = 0.3;
    tex->rgb1[2] = 0.4;
    tex->rgb2[0] = 0.1;
    tex->rgb2[1] = 0.2;
    tex->rgb2[2] = 0.3;
    tex->width   = kFloorTexSize;
    tex->height  = kFloorTexSize;

    mjsMaterial *mat = mjs_addMaterial(spec, nullptr);
    mjs_setString(mjs_getName(mat->element), "groundplane");
    // Set texture at slot mjTEXROLE_RGB (1); vector is pre-initialised with 10 empty strings
    mjs_setInStringVec(mat->textures, mjTEXROLE_RGB, "groundplane");
    mat->texrepeat[0] = kFloorTexRepeat;
    mat->texrepeat[1] = kFloorTexRepeat;
    mat->reflectance  = kFloorReflectance;

    // Left unnamed on purpose: nothing here looks the ground plane up, and naming it "floor"
    // makes the scene fail to compile against any asset that has a geom of that name (a tray
    // bottom, a room model) unless the caller happens to attach it under a prefix.
    mjsGeom *floor = mjs_addGeom(wb, nullptr);
    mjs_setString(floor->material, "groundplane");
    floor->type        = mjGEOM_PLANE;
    floor->pos[2]      = floor_z;
    floor->size[0]     = kFloorHalfSize;
    floor->size[1]     = kFloorHalfSize;
    floor->size[2]     = kFloorThickness;
    floor->contype     = kContactCategoryAll;
    floor->conaffinity = kContactCategoryAll;
    floor->condim      = static_cast<int>(Condim::Tangential);
}

template<std::size_t N, class T> static bool all_set(const T (&values)[N])
{
    return std::none_of(values, values + N, [](T v) { return std::isnan(v); });
}

bool add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects)
{
    for (const auto &obj : objects) {
        if (!obj.mjcf_path.empty()) {
            if (obj.has_rgba && !all_set(obj.rgba)) {
                LOG_ERROR("SceneObject '" << obj.name << "' sets has_rgba but not .rgba");
                return false;
            }
            char      err[kMjErrBuf] = {};
            MjSpecPtr asset =
              make_spec_ptr(mj_parseXML(obj.mjcf_path.c_str(), nullptr, err, sizeof(err)));
            if (!asset) {
                LOG_ERROR("mj_parseXML failed for object asset '" << obj.mjcf_path << "': " << err);
                return false;
            }
            absolutize_asset_files(asset.get());

            // Compile now so mesh files load while this spec's meshdir is alive:
            // mjs_attach defers file loading to the parent compile, but `asset`
            // is freed before then.
            if (mjModel *compiled = mj_compile(asset.get(), nullptr)) {
                mj_deleteModel(compiled);
            } else {
                LOG_ERROR(
                  "failed to compile object asset '" << obj.mjcf_path
                                                     << "': " << mjs_getError(asset.get())
                );
                return false;
            }

            mjsBody *root = first_root_body(asset.get());
            if (!root) {
                LOG_ERROR("no root body found in object asset '" << obj.mjcf_path << "'");
                return false;
            }

            std::string prefix = obj.name.empty() ? "" : obj.name + "_";
            mjsBody    *attached =
              attach_child(spec, obj.attach_to, obj.pos, obj.quat, root, prefix.c_str());
            // Rename the asset's root body to obj.name so callers can write
            // attach_to = { Body, obj.name } without knowing the MJCF-internal
            // root body name. Other elements keep the obj.name + "_" prefix.
            if (attached && !obj.name.empty()) {
                mjs_setString(mjs_getName(attached->element), obj.name.c_str());
            }
            // The scene's colour wins over the asset's: every geom under the root takes it.
            if (attached && obj.has_rgba) {
                std::function<void(mjsBody *)> recolour = [&](mjsBody *body) {
                    for (mjsElement *el = mjs_firstChild(body, mjOBJ_GEOM, 0); el;
                         el             = mjs_nextChild(body, el, 0)) {
                        mjsGeom *g = mjs_asGeom(el);
                        for (int k = 0; k < 4; ++k) g->rgba[k] = obj.rgba[k];
                    }
                    for (mjsElement *el = mjs_firstChild(body, mjOBJ_BODY, 0); el;
                         el             = mjs_nextChild(body, el, 0)) {
                        recolour(mjs_asBody(el));
                    }
                };
                recolour(attached);
            }
            // A non-fixed MJCF object stands free, exactly like a non-fixed primitive: honor
            // the flag with a free joint unless the asset already roots one of its own.
            if (attached && !obj.fixed) {
                bool has_joint = false;
                for (mjsElement *el = mjs_firstChild(attached, mjOBJ_JOINT, 0); el;
                     el             = mjs_nextChild(attached, el, 0)) {
                    has_joint = true;
                    break;
                }
                if (!has_joint) {
                    mjsJoint *fj = mjs_addJoint(attached, nullptr);
                    mjs_setString(mjs_getName(fj->element), (obj.name + "_free").c_str());
                    fj->type = mjJNT_FREE;
                }
            }
            continue;
        }

        if (obj.shape == Shape::Unspecified) {
            LOG_ERROR(
              "primitive SceneObject '"
              << obj.name << "' has Shape::Unspecified; set .shape explicitly (BOX/SPHERE/CYLINDER)"
            );
            return false;
        }
        // Validate fields the user must set on a primitive (unset is NaN, which fails every > 0).
        // Free-jointed bodies also need mass > 0; a fixed body may leave it unset.
        const int n_size = obj.shape == Shape::BOX ? 3 : obj.shape == Shape::CYLINDER ? 2 : 1;
        if (!std::all_of(obj.size, obj.size + n_size, [](double s) { return s > 0.0; })) {
            LOG_ERROR(
              "primitive SceneObject '"
              << obj.name << "' has an unset or non-positive .size for its shape; set its dimensions"
            );
            return false;
        }
        if (!obj.fixed && !(obj.mass > 0.0)) {
            LOG_ERROR(
              "primitive SceneObject '" << obj.name << "' has .mass=" << obj.mass
                                        << "; non-fixed bodies require mass > 0"
            );
            return false;
        }
        if (!all_set(obj.rgba) || !all_set(obj.friction)) {
            LOG_ERROR("primitive SceneObject '" << obj.name << "' leaves .rgba or .friction unset");
            return false;
        }

        // Build the primitive body inside a throwaway spec so it can be attached
        // under any parent kind (body, site, frame, world) via the same helper.
        MjSpecPtr tmp    = make_spec_ptr(mj_makeSpec());
        mjsBody  *tmp_wb = world_body(tmp.get());
        mjsBody  *ob     = mjs_addBody(tmp_wb, nullptr);
        mjs_setString(mjs_getName(ob->element), obj.name.c_str());

        if (!obj.fixed) {
            mjsJoint *fj = mjs_addJoint(ob, nullptr);
            mjs_setString(mjs_getName(fj->element), (obj.name + "_joint").c_str());
            fj->type = mjJNT_FREE;
        }

        mjsGeom *g = mjs_addGeom(ob, nullptr);
        mjs_setString(mjs_getName(g->element), (obj.name + "_geom").c_str());
        switch (obj.shape) {
        case Shape::BOX:
            g->type = mjGEOM_BOX;
            break;
        case Shape::SPHERE:
            g->type = mjGEOM_SPHERE;
            break;
        case Shape::CYLINDER:
            g->type = mjGEOM_CYLINDER;
            break;
        case Shape::Unspecified:
            break; // unreachable, guarded above
        }
        for (int k = 0; k < 3; ++k) g->size[k] = k < n_size ? obj.size[k] : 0.0;
        if (!std::isnan(obj.mass)) g->mass = obj.mass;
        for (int k = 0; k < 4; ++k) g->rgba[k] = obj.rgba[k];
        for (int k = 0; k < 3; ++k) g->friction[k] = obj.friction[k];
        g->contype     = kContactCategoryAll;
        g->conaffinity = kContactCategoryAll;
        g->condim      = static_cast<int>(obj.condim);

        attach_child(spec, obj.attach_to, obj.pos, obj.quat, ob, "");
    }
    return true;
}

static bool add_cameras_to_spec(mjSpec *spec, const std::vector<CameraSpec> &cameras)
{
    mjsBody *wb = world_body(spec);
    for (const auto &cs : cameras) {
        if (!all_set(cs.pos) || std::isnan(cs.fovy)) {
            LOG_ERROR("camera '" << cs.name << "' leaves .pos or .fovy unset");
            return false;
        }
        // The asset's own camera wins: it is the one its author placed, and re-adding the
        // name would fail the compile on a duplicate.
        if (mjs_findElement(spec, mjOBJ_CAMERA, cs.name.c_str())) continue;
        mjsBody *anchor = cs.body.empty() ? wb : mjs_findBody(spec, cs.body.c_str());
        if (!anchor) {
            LOG_WARN("camera '" << cs.name << "': no body '" << cs.body << "' in the scene");
            continue;
        }
        mjsCamera *cam = mjs_addCamera(anchor, nullptr);
        mjs_setString(mjs_getName(cam->element), cs.name.c_str());
        cam->pos[0] = cs.pos[0];
        cam->pos[1] = cs.pos[1];
        cam->pos[2] = cs.pos[2];
        cam->fovy   = cs.fovy;
        quat_xyzw_to_mj_quat(cs.quat, cam->quat);
    }
    return true;
}

static void add_site_to_spec(mjSpec *spec, mjsBody *body, const SiteSpec &ss)
{
    // The asset's own site wins: it is the one its author placed, and re-adding the
    // name would fail the compile on a duplicate.
    if (mjs_findElement(spec, mjOBJ_SITE, ss.name.c_str())) return;

    mjsSite *site = mjs_addSite(body, nullptr);
    mjs_setString(mjs_getName(site->element), ss.name.c_str());
    site->type    = mjGEOM_SPHERE;
    site->size[0] = kFrameSiteSize;
    site->size[1] = kFrameSiteSize;
    site->size[2] = kFrameSiteSize;
    site->group   = kFrameSiteGroup;
    site->pos[0]  = ss.pos[0];
    site->pos[1]  = ss.pos[1];
    site->pos[2]  = ss.pos[2];
    quat_xyzw_to_mj_quat(ss.quat, site->quat);
}

// Every pending site whose body has arrived, added now and struck off the list.
//
// Sites cannot all wait until the end: a site marks a frame the scene states, and a robot may
// have to bolt to one -- an arm to the `left_arm_attachment` frame on the platform's base_link.
// An attach target has to exist before the attach, which is the same reason objects go in ahead
// of robots. What no body carries yet stays pending for the next round.
static void add_ready_sites(mjSpec *spec, std::vector<SiteSpec> &pending)
{
    for (auto it = pending.begin(); it != pending.end();) {
        mjsBody *body = mjs_findBody(spec, it->body.c_str());
        if (!body) {
            ++it;
            continue;
        }
        add_site_to_spec(spec, body, *it);
        it = pending.erase(it);
    }
}

static void add_sites_to_spec(mjSpec *spec, const std::vector<SiteSpec> &sites)
{
    for (const auto &ss : sites) {
        mjsBody *body = mjs_findBody(spec, ss.body.c_str());
        if (!body) {
            LOG_WARN("site '" << ss.name << "': no body '" << ss.body << "' in the scene");
            continue;
        }
        add_site_to_spec(spec, body, ss);
    }
}

// Compile spec into a model and data; spec is always deleted.
// A compiled model cannot be written back to XML on its own; mj_saveXML needs its spec.
static std::unordered_map<const mjModel *, MjSpecPtr> g_model_specs;

bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data)
{
    MjSpecPtr owned = make_spec_ptr(spec);
    *out_model      = mj_compile(spec, nullptr);
    if (!*out_model) {
        LOG_ERROR("mj_compile failed: " << mjs_getError(spec));
        return false;
    }
    LOG_INFO(
      "scene compiled: nq=" << (*out_model)->nq << " nv=" << (*out_model)->nv
                            << " nbody=" << (*out_model)->nbody
    );
    *out_data = mj_makeData(*out_model);
    if (!*out_data) {
        mj_deleteModel(*out_model);
        *out_model = nullptr;
        return false;
    }
    g_model_specs.insert_or_assign(*out_model, std::move(owned));
    return true;
}

// KDL helpers

static bool
  get_site_frame_in_body(Env *env, const char *body_name, const char *site_name, KDL::Frame *out)
{
    if (!body_name || !site_name || !out) return false;

    ensure_kinematics(env);
    const mjModel *model   = env->model;
    const mjData  *data    = env->data;
    int            body_id = mj_name2id(model, mjOBJ_BODY, body_name);
    int            site_id = mj_name2id(model, mjOBJ_SITE, site_name);
    if (body_id < 0 || site_id < 0) return false;

    const double *body_pos = data->xpos + 3 * body_id;
    const double *site_pos = data->site_xpos + 3 * site_id;

    KDL::Frame world_T_body(
      mj_xmat_to_kdl_rot(data->xmat + 9 * body_id),
      KDL::Vector(body_pos[0], body_pos[1], body_pos[2])
    );
    KDL::Frame world_T_site(
      mj_xmat_to_kdl_rot(data->site_xmat + 9 * site_id),
      KDL::Vector(site_pos[0], site_pos[1], site_pos[2])
    );

    *out = world_T_body.Inverse() * world_T_site;
    return true;
}

// Collect all body IDs in the subtree rooted at root_bid (inclusive).
// MuJoCo stores bodies in topological order (parent always precedes children),
// so a single forward pass is sufficient.
static std::vector<int> collect_subtree(const mjModel *model, int root_bid)
{
    std::vector<bool> mark(model->nbody, false);
    mark[root_bid] = true;
    for (int b = root_bid + 1; b < model->nbody; ++b)
        if (mark[model->body_parentid[b]]) mark[b] = true;
    std::vector<int> result;
    for (int b = root_bid; b < model->nbody; ++b)
        if (mark[b]) result.push_back(b);
    return result;
}

/*
 * Compute the lumped KDL::RigidBodyInertia for a set of bodies, expressed
 * in tip_body's local frame.
 *
 * Requires data->xpos and data->xmat to be valid (mj_forward must have been
 * called beforehand).  xmat[9*b] is the body-to-world rotation matrix stored
 * row-major, so:
 *   v_world  = R * v_body   where R[r][c] = xmat[9*b + 3*r + c]
 *   v_body   = R^T * v_world
 */
static KDL::RigidBodyInertia compute_tool_inertia(
  const mjModel          *model,
  const mjData           *data,
  int                     tip_bid,
  const std::vector<int> &tool_bodies
)
{
    // Step 1: total mass and world-frame COM.
    double total_mass = 0.0;
    double com_w[3]   = {};
    for (int b : tool_bodies) {
        double m = model->body_mass[b];
        total_mass += m;
        const double *xp = &data->xpos[3 * b];
        const double *xm = &data->xmat[9 * b];
        const double *ip = &model->body_ipos[3 * b]; // COM in body frame
        for (int a = 0; a < 3; ++a)
            com_w[a] +=
              m * (xp[a] + xm[3 * a] * ip[0] + xm[3 * a + 1] * ip[1] + xm[3 * a + 2] * ip[2]);
    }
    if (total_mass <= 0.0) return KDL::RigidBodyInertia::Zero();
    for (double &v : com_w) v /= total_mass;

    // Step 2: combined inertia about com_w, in world frame.
    double I_w[3][3] = {};
    for (int b : tool_bodies) {
        double        m  = model->body_mass[b];
        const double *xp = &data->xpos[3 * b];
        const double *xm = &data->xmat[9 * b];
        const double *ip = &model->body_ipos[3 * b];

        // Body COM in world frame.
        double r[3];
        for (int a = 0; a < 3; ++a)
            r[a] = xp[a] + xm[3 * a] * ip[0] + xm[3 * a + 1] * ip[1] + xm[3 * a + 2] * ip[2];

        /*
         * Body inertia in world frame: Rf * diag(id) * Rf^T
         * where Rf = xmat * iquat_R maps from principal (inertia) axes to world.
         */
        KDL::Rotation iR = mj_quat_to_kdl_rot(&model->body_iquat[4 * b]);
        const double *id = &model->body_inertia[3 * b];
        double        Rf[3][3];
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col) {
                Rf[row][col] = 0.0;
                for (int k = 0; k < 3; ++k) Rf[row][col] += xm[3 * row + k] * iR(k, col);
            }
        double I_b[3][3] = {};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                for (int k = 0; k < 3; ++k) I_b[row][col] += Rf[row][k] * id[k] * Rf[col][k];

        // Parallel-axis theorem: d = body_com - total_com.
        double d[3] = { r[0] - com_w[0], r[1] - com_w[1], r[2] - com_w[2] };
        double d2   = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        for (int a = 0; a < 3; ++a)
            for (int c = 0; c < 3; ++c) {
                I_w[a][c] += I_b[a][c];
                I_w[a][c] += m * ((a == c ? d2 : 0.0) - d[a] * d[c]);
            }
    }

    // Step 3: transform COM and inertia into tip_body's local frame.
    const double *txm = &data->xmat[9 * tip_bid];
    const double *txp = &data->xpos[3 * tip_bid];

    // com_tip = R_tip^T * (com_w - tip_pos);   R_tip^T[a][j] = txm[3*j+a]
    double      dv[3] = { com_w[0] - txp[0], com_w[1] - txp[1], com_w[2] - txp[2] };
    KDL::Vector com_tip(
      txm[0] * dv[0] + txm[3] * dv[1] + txm[6] * dv[2],
      txm[1] * dv[0] + txm[4] * dv[1] + txm[7] * dv[2],
      txm[2] * dv[0] + txm[5] * dv[1] + txm[8] * dv[2]
    );

    // I_tip = R_tip^T * I_w * R_tip;   I_tip[a][c] = sum_{j,k} txm[3j+a]*I_w[j][k]*txm[3k+c]
    double I_t[3][3] = {};
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 3; ++c)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    I_t[a][c] += txm[3 * j + a] * I_w[j][k] * txm[3 * k + c];

    return KDL::RigidBodyInertia(
      total_mass,
      com_tip,
      KDL::RotationalInertia(I_t[0][0], I_t[1][1], I_t[2][2], I_t[0][1], I_t[0][2], I_t[1][2])
    );
}

// Extract the full rigid-body inertia for body bid from a compiled mjModel.
static KDL::RigidBodyInertia mj_body_inertia(const mjModel *model, int bid)
{
    double        mass    = model->body_mass[bid];
    const double *ip      = &model->body_ipos[3 * bid];
    KDL::Rotation iR      = mj_quat_to_kdl_rot(&model->body_iquat[4 * bid]);
    const double *id      = &model->body_inertia[3 * bid];
    double        I[3][3] = {};
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 3; c++) I[a][b] += iR(a, c) * id[c] * iR(b, c);
    return KDL::RigidBodyInertia(
      mass,
      KDL::Vector(ip[0], ip[1], ip[2]),
      KDL::RotationalInertia(I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2])
    );
}


// The mode an actuator gives on its own: a position servo (fixed gain kp, affine bias
// [0, -kp, -kv]) is POSITION, a motor (fixed gain, no bias) is TORQUE; -1 for anything else.
static int native_mode(int gaintype, const double *gainprm, int biastype, const double *biasprm,
                       int dyntype)
{
    if (dyntype != mjDYN_NONE || gaintype != mjGAIN_FIXED || gainprm[0] == 0.0) return -1;
    if (biastype == mjBIAS_NONE) return static_cast<int>(CtrlMode::TORQUE);
    if (biastype == mjBIAS_AFFINE && biasprm[0] == 0.0 && biasprm[1] == -gainprm[0])
        return static_cast<int>(CtrlMode::POSITION);
    return -1;
}

static int mode_group(int robot, int mode) { return 1 + 3 * robot + mode; }

static bool build_index_map(Robot *s, const std::string &pfx = "")
{
    RobotInternals &in = *s->_impl;
    in.kdl_to_mj_qpos.clear();
    in.kdl_to_mj_dof.clear();
    in.kdl_to_mj_ctrl.clear();
    for (auto &ctrl : in.mode_ctrl) ctrl.assign(s->joint_names.size(), -1);
    in.robot_index  = -1;
    in.mode_applied = false;
    in.mj_prefix    = pfx;
    if (!s->model) return false;
    const mjModel *m = s->model;
    for (size_t j = 0; j < s->joint_names.size(); ++j) {
        const std::string &name = s->joint_names[j];
        int                id   = mj_name2id(m, mjOBJ_JOINT, (pfx + name).c_str());
        if (id < 0) {
            LOG_ERROR(
              "joint '" << pfx << name
                        << "' not found in MuJoCo model - check robot prefix or URDF joint names"
            );
            return false;
        }
        in.kdl_to_mj_qpos.push_back(m->jnt_qposadr[id]);
        in.kdl_to_mj_dof.push_back(m->jnt_dofadr[id]);
        // A mode group says which mode an actuator serves; ungrouped ones serve their native mode.
        int first = -1;
        for (int ai = 0; ai < m->nu; ++ai) {
            if (m->actuator_trntype[ai] != mjTRN_JOINT || m->actuator_trnid[2 * ai] != id) continue;
            if (first < 0) first = ai;
            const int group = m->actuator_group[ai];
            int       mode  = -1;
            if (group >= 1 && group <= 30) {
                mode           = (group - 1) % 3;
                in.robot_index = (group - 1) / 3;
            } else {
                mode = native_mode(
                  m->actuator_gaintype[ai],
                  m->actuator_gainprm + mjNGAIN * ai,
                  m->actuator_biastype[ai],
                  m->actuator_biasprm + mjNBIAS * ai,
                  m->actuator_dyntype[ai]
                );
            }
            if (mode >= 0 && in.mode_ctrl[mode][j] < 0) in.mode_ctrl[mode][j] = ai;
        }
        const int pos = in.mode_ctrl[static_cast<int>(CtrlMode::POSITION)][j];
        in.kdl_to_mj_ctrl.push_back(pos >= 0 ? pos : first);
    }

    // The mode the model is in now: every joint has its actuator and its group is enabled.
    const int n = s->n_joints;
    for (int mode = 0; mode < 3; ++mode) {
        const bool driven =
          std::all_of(in.mode_ctrl[mode].begin(), in.mode_ctrl[mode].end(), [](int a) {
              return a >= 0;
          });
        const bool enabled =
          in.robot_index < 0 || !(m->opt.disableactuator & (1 << mode_group(in.robot_index, mode)));
        if (driven && enabled && n > 0) {
            s->ctrl_mode    = static_cast<CtrlMode>(mode);
            in.applied_mode = s->ctrl_mode;
            in.mode_applied = true;
            break;
        }
    }
    return true;
}

// A joint the model leaves unlimited (a continuous wrist) is unlimited here too, not +-pi.
static std::pair<double, double> joint_range(const mjModel *model, int jid)
{
    if (jid >= 0 && model->jnt_limited[jid])
        return { model->jnt_range[2 * jid], model->jnt_range[2 * jid + 1] };
    LOG_INFO(
      "joint '" << (jid >= 0 ? mj_id2name(model, mjOBJ_JOINT, jid) : "?") << "' is unlimited"
    );
    const double inf = std::numeric_limits<double>::infinity();
    return { -inf, inf };
}

// Build KDL chain from compiled mjModel (no URDF needed)

static bool
  build_kdl_from_model(Robot *s, mjModel *model, const char *base_body, const char *tip_body)
{
    int base_bid = mj_name2id(model, mjOBJ_BODY, base_body);
    int tip_bid  = mj_name2id(model, mjOBJ_BODY, tip_body);
    if (base_bid < 0) {
        LOG_ERROR("base body '" << base_body << "' not found in compiled model");
        return false;
    }
    if (tip_bid < 0) {
        LOG_ERROR("tip body '" << tip_body << "' not found in compiled model");
        return false;
    }

    std::vector<int> bids;
    for (int b = tip_bid; b != base_bid; b = model->body_parentid[b]) {
        if (b == 0) {
            LOG_ERROR(
              "'" << tip_body << "' is not a descendant of '" << base_body
                  << "'  - check body hierarchy in the model"
            );
            return false;
        }
        bids.push_back(b);
    }
    std::reverse(bids.begin(), bids.end());

    s->chain = KDL::Chain();
    s->joint_names.clear();
    s->joint_limits.clear();

    for (int bid : bids) {
        const char   *bname = mj_id2name(model, mjOBJ_BODY, bid);
        KDL::Rotation bR    = mj_quat_to_kdl_rot(&model->body_quat[4 * bid]);
        KDL::Vector   bv(
          model->body_pos[3 * bid], model->body_pos[3 * bid + 1], model->body_pos[3 * bid + 2]
        );
        KDL::Frame F(bR, bv);

        KDL::Joint jnt(KDL::Joint::None);
        for (int jid = model->body_jntadr[bid];
             jid < model->body_jntadr[bid] + model->body_jntnum[bid];
             ++jid) {
            if (model->jnt_type[jid] != mjJNT_HINGE && model->jnt_type[jid] != mjJNT_SLIDE)
                continue;
            const char *jname = mj_id2name(model, mjOBJ_JOINT, jid);
            KDL::Vector jp(
              model->jnt_pos[3 * jid], model->jnt_pos[3 * jid + 1], model->jnt_pos[3 * jid + 2]
            );
            KDL::Vector ja(
              model->jnt_axis[3 * jid], model->jnt_axis[3 * jid + 1], model->jnt_axis[3 * jid + 2]
            );
            KDL::Vector           origin = bv + bR * jp;
            KDL::Vector           axis   = bR * ja;
            KDL::Joint::JointType jtype =
              (model->jnt_type[jid] == mjJNT_HINGE) ? KDL::Joint::RotAxis : KDL::Joint::TransAxis;
            // Rotor inertia (armature) is part of what the joint must drive; KDL's dynamics
            // solvers add it, so the chain matches the simulated model.
            const double armature = model->dof_armature[model->jnt_dofadr[jid]];
            jnt = KDL::Joint(jname ? jname : "", origin, axis, jtype, 1.0, 0.0, armature);
            if (jname) {
                s->joint_names.push_back(jname);
                s->joint_limits.push_back(joint_range(model, jid));
            }
            break;
        }

        KDL::RigidBodyInertia inertia = mj_body_inertia(model, bid);

        s->chain.addSegment(KDL::Segment(bname ? bname : "", jnt, F, inertia));
    }
    s->n_joints = (int)s->chain.getNrOfJoints();
    return true;
}

// Scene API

bool save_model_xml(const mjModel *model, const char *path)
{
    char       err[kMjErrBuf] = {};
    int        ok             = 0;
    const auto spec           = g_model_specs.find(model);
    if (spec != g_model_specs.end()) {
        ok = mj_copyBack(spec->second.get(), model)
             && mj_saveXML(spec->second.get(), path, err, sizeof(err)) == 0;
    } else {
        ok = mj_saveLastXML(path, model, err, sizeof(err));
    }
    if (!ok) {
        LOG_ERROR("mj_saveLastXML failed for '" << path << "': " << err);
    } else {
        LOG_INFO("model saved to '" << path << "'");
    }
    return ok != 0;
}

void destroy_scene(mjModel *model, mjData *data)
{
    if (model) g_model_specs.erase(model);
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
}

static void close_viewer(Env *env);

// Every cache is tied to the model it was taken from.
static void forget_model(Env *env)
{
    env->_impl->computed = false;
    for (auto &names : env->_impl->names) names.clear();
}

bool init_env(Env *env, const SceneSpec *spec)
{
    if (!env || !spec) return false;

    cleanup(env);
    env->spec = *spec;
    if (!build_scene(&env->model, &env->data, &env->spec)) {
        env->model = nullptr;
        env->data  = nullptr;
        return false;
    }
    env->scene       = SceneState{};
    env->scene.model = env->model;
    return true;
}

void cleanup(Env *env)
{
    if (!env) return;
    close_viewer(env);
    for (Robot *r : env->robots) {
        r->_impl->env = nullptr;
        r->model      = nullptr;
        r->data       = nullptr;
    }
    env->robots.clear();
    destroy_scene(env->model, env->data);
    env->model = nullptr;
    env->data  = nullptr;
    env->scene = SceneState{};
    forget_model(env);
    env->on_reset = nullptr;
}

bool attach_to_spec(mjSpec *robot_spec, const AttachmentSpec *a)
{
    if (!robot_spec || !a || !a->mjcf_path) return false;
    ensure_plugins_loaded();
    LOG_INFO(
      "attach_to_spec: parent='" << (a->attach_to.name ? a->attach_to.name : "(world)")
                                 << "' prefix='" << (a->prefix ? a->prefix : "") << "'"
    );

    char      err[kMjErrBuf] = {};
    MjSpecPtr att            = make_spec_ptr(mj_parseXML(a->mjcf_path, nullptr, err, sizeof(err)));
    if (!att) {
        LOG_ERROR("mj_parseXML failed for attachment '" << a->mjcf_path << "': " << err);
        return false;
    }
    absolutize_asset_files(att.get());

    mjsBody *att_root = first_root_body(att.get());
    if (!att_root) {
        LOG_ERROR("no root body found in attachment spec '" << a->mjcf_path << "'");
        return false;
    }

    if (!attach_child(robot_spec, a->attach_to, a->pos, a->quat, att_root, a->prefix)) {
        return false;
    }
    // att (deep-copied into robot_spec) is freed by MjSpecPtr at scope exit.

    // Register contact exclusions.
    for (const auto &ex : a->contact_exclusions) {
        mjsExclude *exc = mjs_addExclude(robot_spec);
        mjs_setString(exc->bodyname1, ex.first.c_str());
        mjs_setString(exc->bodyname2, ex.second.c_str());
    }
    return true;
}

static bool range_limited(mjtLimited flag, const double *range)
{
    return flag == mjLIMITED_TRUE || (flag == mjLIMITED_AUTO && range[0] < range[1]);
}

/* Give each listed joint one actuator per requested mode, each mode in its own group of this
 * robot; the joint's own actuator joins the group of the mode it gives natively. Collects the
 * groups to start disabled. */
static bool add_mode_actuators(mjSpec *arm, const RobotSpec &rs, int robot, int *disable_bits)
{
    if (rs.modes.empty()) return true;
    if (mode_group(robot, static_cast<int>(CtrlMode::VELOCITY)) > 30) {
        LOG_ERROR("robots[" << robot << "]: control-mode groups only reach robot index 9");
        return false;
    }

    std::map<std::string, std::vector<mjsActuator *>> by_joint;
    for (mjsElement *e = mjs_firstElement(arm, mjOBJ_ACTUATOR); e; e = mjs_nextElement(arm, e)) {
        mjsActuator *a = mjs_asActuator(e);
        if (a->trntype == mjTRN_JOINT) by_joint[mjs_getString(a->target)].push_back(a);
    }

    int enabled = 0, used = 0;
    for (const CtrlModeSpec &ms : rs.modes) {
        // Every actuated joint when none are named: those that cannot take modes are skipped.
        const bool               all    = ms.joints.empty();
        std::vector<std::string> joints = ms.joints;
        if (all)
            for (const auto &entry : by_joint) joints.push_back(entry.first);

        for (const std::string &joint : joints) {
            const auto   it     = by_joint.find(joint);
            mjsActuator *own    = (it != by_joint.end() && it->second.size() == 1) ? it->second.front() : nullptr;
            const int    native = own ? native_mode(
                                          own->gaintype, own->gainprm, own->biastype, own->biasprm, own->dyntype
                                        )
                                      : -1;
            if (native < 0) {
                if (all) {
                    LOG_INFO("robots[" << robot << "]: joint '" << joint << "' takes no control modes");
                    continue;
                }
                LOG_ERROR(
                  "robots[" << robot << "]: joint '" << joint
                            << "' needs exactly one position-servo or motor actuator"
                );
                return false;
            }
            own->group = mode_group(robot, native);
            enabled |= 1 << own->group;
            used |= 1 << own->group;

            const int mode = static_cast<int>(ms.mode);
            if (mode == native) continue;
            if (ms.mode == CtrlMode::POSITION) {
                LOG_ERROR(
                  "robots[" << robot << "]: joint '" << joint << "' is motor-driven; POSITION is not supported"
                );
                return false;
            }
            if (ms.mode == CtrlMode::VELOCITY && ms.kv <= 0.0) {
                LOG_ERROR("robots[" << robot << "]: VELOCITY needs kv > 0");
                return false;
            }

            // Both limits below are in actuator-force units: a servo's forcerange, a motor's ctrl.
            const bool    servo   = native == static_cast<int>(CtrlMode::POSITION);
            const double *range   = servo ? own->forcerange : own->ctrlrange;
            const bool    limited = range_limited(servo ? own->forcelimited : own->ctrllimited, range);

            mjsActuator *added = mjs_addActuator(arm, nullptr);
            added->trntype     = mjTRN_JOINT;
            mjs_setString(added->target, joint.c_str());
            added->gear[0] = own->gear[0];
            added->group   = mode_group(robot, mode);
            used |= 1 << added->group;
            if (ms.mode == CtrlMode::TORQUE) {
                mjs_setToMotor(added);
                added->ctrllimited  = limited ? mjLIMITED_TRUE : mjLIMITED_FALSE;
                added->ctrlrange[0] = range[0];
                added->ctrlrange[1] = range[1];
            } else {
                mjs_setToVelocity(added, ms.kv);
                added->forcelimited  = limited ? mjLIMITED_TRUE : mjLIMITED_FALSE;
                added->forcerange[0] = range[0];
                added->forcerange[1] = range[1];
            }
            const char *suffix = ms.mode == CtrlMode::TORQUE ? "_torque" : "_velocity";
            mjs_setName(added->element, (joint + suffix).c_str());
        }
    }
    *disable_bits |= used & ~enabled;

    // A keyframe that sets ctrl sets one per actuator; the added ones come last and start at 0.
    int nu = 0;
    for (mjsElement *e = mjs_firstElement(arm, mjOBJ_ACTUATOR); e; e = mjs_nextElement(arm, e))
        ++nu;
    for (mjsElement *e = mjs_firstElement(arm, mjOBJ_KEY); e; e = mjs_nextElement(arm, e)) {
        mjsKey       *key  = mjs_asKey(e);
        int           size = 0;
        const double *ctrl = mjs_getDouble(key->ctrl, &size);
        if (size == 0 || size >= nu) continue;
        std::vector<double> padded(ctrl, ctrl + size);
        padded.resize(nu, 0.0);
        mjs_setDouble(key->ctrl, padded.data(), nu);
    }
    return true;
}

bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *sc)
{
    if (!sc) return false;
    if (!(sc->timestep > 0.0)) {
        LOG_ERROR(
          "SceneSpec::timestep must be > 0 (got "
          << sc->timestep << "); the field has no default, set it explicitly (suggested 0.002 s)"
        );
        return false;
    }
    ensure_plugins_loaded();
    LOG_INFO(
      "build_scene: " << sc->robots.size() << " robot(s)" << ", objects=" << sc->objects.size()
    );

    MjSpecPtr scene = make_spec_ptr(mj_makeSpec());
    if (!scene) {
        LOG_ERROR("mj_makeSpec() failed");
        return false;
    }

    scene->compiler.balanceinertia = true; // mjsCompiler stores as int 0/1
    scene->compiler.discardvisual  = false;

    // Scene decorations go in before any object/robot so they exist as world
    // anchors regardless of declaration order.
    if (sc->add_skybox) add_skybox_to_spec(scene.get());
    if (sc->add_floor) add_floor_to_spec(scene.get(), sc->floor_z);

    // Objects come before robots so a robot can attach to a SceneObject (e.g.
    // {AttachKind::Site, "table_mount"}). A child object that references
    // another object must appear after its parent in SceneSpec::objects.
    if (!add_objects_to_spec(scene.get(), sc->objects)) return false;

    // Sites land as their bodies arrive, so a later attach can name one -- see add_ready_sites.
    std::vector<SiteSpec> pending_sites = sc->sites;
    add_ready_sites(scene.get(), pending_sites);

    bool first_arm      = true;
    int  disable_bits   = 0;
    char err[kMjErrBuf] = {};
    for (int ai = 0; ai < (int)sc->robots.size(); ++ai) {
        const RobotSpec &rs = sc->robots[ai];
        if (!rs.path) {
            LOG_ERROR("robots[" << ai << "].path is null");
            return false;
        }

        MjSpecPtr arm = make_spec_ptr(mj_parseXML(rs.path, nullptr, err, sizeof(err)));
        if (!arm) {
            LOG_ERROR("mj_parseXML failed for '" << rs.path << "': " << err);
            return false;
        }
        absolutize_asset_files(arm.get());
        // Before the attachments are merged in, so only the robot's own joints take modes.
        if (!add_mode_actuators(arm.get(), rs, ai, &disable_bits)) return false;

        // Inherit physics options (integrator, solver, etc.) from the first
        // arm, then apply the SceneSpec's user-controlled fields on top.
        if (first_arm) {
            scene->option            = arm->option;
            scene->option.timestep   = sc->timestep;
            scene->option.gravity[2] = sc->gravity_z;
            first_arm                = false;
        }

        // Apply attachment chain in order (mount, sensor, gripper, etc.).
        for (const auto &att : rs.attachments) {
            if (!attach_to_spec(arm.get(), &att)) return false;
        }

        mjsBody *arm_root = first_root_body(arm.get());
        if (!arm_root) {
            LOG_ERROR("no root body found in arm spec '" << rs.path << "'");
            return false;
        }

        if (!attach_child(scene.get(), rs.attach_to, rs.pos, rs.quat, arm_root, rs.prefix)) {
            LOG_ERROR("attach failed for arm " << ai);
            return false;
        }
        add_ready_sites(scene.get(), pending_sites);
        // arm (deep-copied into scene) is freed by MjSpecPtr at scope exit.
    }

    // Robot-less scenes skip the first_arm branch; apply timestep/gravity here.
    if (first_arm) {
        scene->option.timestep   = sc->timestep;
        scene->option.gravity[2] = sc->gravity_z;
    }
    // Every robot starts in its native mode: its other mode groups are off.
    scene->option.disableactuator |= disable_bits;

    if (!add_cameras_to_spec(scene.get(), sc->cameras)) return false;
    // Whatever is still pending: a site whose body no robot or object ever brought in, reported
    // here rather than dropped in silence.
    if (!pending_sites.empty()) add_sites_to_spec(scene.get(), pending_sites);
    // compile_and_make_data takes ownership of the raw spec and always deletes it.
    return compile_and_make_data(scene.release(), out_model, out_data);
}

static void reload_viewer(Env *env, mjModel *m, mjData *d);
static bool rebind_scene(Env *env);

// Every MuJoCo address a Robot holds is only valid for the model it was resolved on.
static bool rebind_robots(Env *env)
{
    const mjModel *m  = env->model;
    bool           ok = true;
    for (Robot *r : env->robots) {
        const std::vector<double> pos_cmd = r->jnt_pos_cmd;
        const std::vector<double> vel_cmd = r->jnt_vel_cmd;
        const std::vector<double> trq_cmd = r->jnt_trq_cmd;
        const CtrlMode            mode    = r->ctrl_mode;
        r->model                          = env->model;
        r->data                           = env->data;
        if (!build_index_map(r, r->_impl->mj_prefix)) {
            ok = false;
            continue;
        }
        // The rebuilt model starts in the native mode; carry on in the one the robot was in.
        if (mode != r->ctrl_mode && !set_control_mode(r, mode)) ok = false;
        r->jnt_pos_cmd = pos_cmd;
        r->jnt_vel_cmd = vel_cmd;
        r->jnt_trq_cmd = trq_cmd;
        for (ForceTorqueSensor &sensor : r->ft_sensors) {
            const int force_id  = mj_name2id(m, mjOBJ_SENSOR, sensor.force_sensor.c_str());
            const int torque_id = mj_name2id(m, mjOBJ_SENSOR, sensor.torque_sensor.c_str());
            const int site_id =
              sensor.frame_site.empty() ? -1 : mj_name2id(m, mjOBJ_SITE, sensor.frame_site.c_str());
            if (force_id < 0 || torque_id < 0 || (!sensor.frame_site.empty() && site_id < 0)) {
                LOG_ERROR("FT sensor '" << sensor.name << "' is gone from the rebuilt model");
                ok = false;
                continue;
            }
            sensor.force_adr     = m->sensor_adr[force_id];
            sensor.torque_adr    = m->sensor_adr[torque_id];
            sensor.frame_site_id = site_id;
        }
    }
    return ok;
}

// Swap in a rebuilt pair; the old one lives until the viewer has let go of it. Whatever no
// longer resolves is logged and skipped from then on.
static void swap_scene(Env *env, mjModel *m, mjData *d)
{
    reload_viewer(env, m, d);
    destroy_scene(env->model, env->data);
    env->model = m;
    env->data  = d;
    forget_model(env);
    rebind_robots(env);
    rebind_scene(env);
}

bool scene_add_object(Env *env, const SceneObject &obj)
{
    if (!env) return false;
    env->spec.objects.push_back(obj);
    mjModel *m = nullptr;
    mjData  *d = nullptr;
    if (!build_scene(&m, &d, &env->spec)) {
        env->spec.objects.pop_back();
        return false;
    }
    swap_scene(env, m, d);
    return true;
}

bool scene_remove_object(Env *env, const std::string &name)
{
    if (!env) return false;
    auto &objects = env->spec.objects;
    auto  it      = std::find_if(objects.begin(), objects.end(), [&](const SceneObject &o) {
        return o.name == name;
    });
    if (it == objects.end()) return false;
    SceneObject removed = std::move(*it);
    objects.erase(it);
    mjModel *m = nullptr;
    mjData  *d = nullptr;
    if (!build_scene(&m, &d, &env->spec)) {
        objects.push_back(removed);
        return false;
    }
    swap_scene(env, m, d);
    return true;
}

std::string scene_object_site_name(const SceneObject &obj, const char *site_name)
{
    if (!site_name) return {};
    return obj.name.empty() ? std::string(site_name) : obj.name + "_" + site_name;
}

bool get_site_frame(Env *env, const char *site_name, KDL::Frame *out)
{
    if (!env || !env->model || !site_name || !out) return false;

    const int sid = cached_name2id(env, mjOBJ_SITE, site_name);
    if (sid < 0) return false;

    const auto lock = lock_env(env);
    ensure_kinematics(env);
    const double *p = env->data->site_xpos + 3 * sid;
    const double *R = env->data->site_xmat + 9 * sid;
    *out            = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
    return true;
}

bool get_body_frame(Env *env, const char *body_name, KDL::Frame *out)
{
    if (!env || !env->model || !body_name || !out) return false;

    const int bid = cached_name2id(env, mjOBJ_BODY, body_name);
    if (bid < 0) return false;

    const auto lock = lock_env(env);
    ensure_kinematics(env);
    const double *p = env->data->xpos + 3 * bid;
    const double *R = env->data->xmat + 9 * bid;
    *out            = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
    return true;
}

// A joint by name, or the transmission joint of an actuator by that name; -1 when neither.
static int resolve_joint_id(Env *env, const char *name)
{
    const mjModel *model = env->model;
    int            jid   = cached_name2id(env, mjOBJ_JOINT, name);
    if (jid >= 0) return jid;
    const int aid = cached_name2id(env, mjOBJ_ACTUATOR, name);
    if (aid < 0) return -1;
    if (model->actuator_trntype[aid] == mjTRN_JOINT) {
        jid = model->actuator_trnid[2 * aid];
    } else if (model->actuator_trntype[aid] == mjTRN_TENDON) {
        const int tid = model->actuator_trnid[2 * aid];
        jid           = model->wrap_objid[model->tendon_adr[tid]];
    }
    return jid;
}

bool get_joint_position(Env *env, const char *name, double *out)
{
    if (!env || !env->model || !name || !out) return false;

    const int jid = resolve_joint_id(env, name);
    if (jid < 0) return false;

    const auto lock = lock_env(env);
    *out            = env->data->qpos[env->model->jnt_qposadr[jid]];
    return true;
}

bool get_joint_velocity(Env *env, const char *name, double *out)
{
    if (!env || !env->model || !name || !out) return false;

    const int jid = resolve_joint_id(env, name);
    if (jid < 0) return false;

    const auto lock = lock_env(env);
    *out            = env->data->qvel[env->model->jnt_dofadr[jid]];
    return true;
}

std::vector<std::string> get_camera_names(const mjModel *model)
{
    std::vector<std::string> names;
    if (!model) return names;
    for (int i = 0; i < model->ncam; ++i) {
        const char *name = mj_id2name(model, mjOBJ_CAMERA, i);
        if (name) names.push_back(name);
    }
    return names;
}

static bool use_camera_impl(mjvCamera *cam, const mjModel *model, const char *name)
{
    if (!name || name[0] == '\0') {
        mjv_defaultFreeCamera(model, cam);
        return true;
    }
    int id = mj_name2id(model, mjOBJ_CAMERA, name);
    if (id < 0) return false;
    cam->type       = mjCAMERA_FIXED;
    cam->fixedcamid = id;
    return true;
}

bool use_camera(VideoRecorder *vr, const mjModel *model, const char *name)
{
    if (!vr || !model) return false;
    return use_camera_impl(&vr->cam, model, name);
}

// Robot API

static bool resolve_ft_sensors(Robot *r, const ToolFrameSpec *tool)
{
    r->ft_sensors.clear();
    if (!tool) return true;

    for (const ForceTorqueSensorSpec &spec : tool->ft_sensors) {
        if (!spec.name || spec.name[0] == '\0') {
            LOG_ERROR("ForceTorqueSensorSpec.name is required");
            return false;
        }

        const std::string name = spec.name;
        const std::string force_name =
          (spec.force_sensor && spec.force_sensor[0] != '\0') ? spec.force_sensor : name + "_force";
        const std::string torque_name = (spec.torque_sensor && spec.torque_sensor[0] != '\0')
                                          ? spec.torque_sensor
                                          : name + "_torque";

        const int force_id = mj_name2id(r->model, mjOBJ_SENSOR, force_name.c_str());
        if (force_id < 0) {
            LOG_ERROR(
              "force sensor '" << force_name << "' not found for FT sensor '" << name << "'"
            );
            return false;
        }
        const int torque_id = mj_name2id(r->model, mjOBJ_SENSOR, torque_name.c_str());
        if (torque_id < 0) {
            LOG_ERROR(
              "torque sensor '" << torque_name << "' not found for FT sensor '" << name << "'"
            );
            return false;
        }
        if (r->model->sensor_type[force_id] != mjSENS_FORCE
            || r->model->sensor_dim[force_id] != 3) {
            LOG_ERROR("sensor '" << force_name << "' must be a 3D MuJoCo force sensor");
            return false;
        }
        if (r->model->sensor_type[torque_id] != mjSENS_TORQUE
            || r->model->sensor_dim[torque_id] != 3) {
            LOG_ERROR("sensor '" << torque_name << "' must be a 3D MuJoCo torque sensor");
            return false;
        }

        ForceTorqueSensor sensor;
        sensor.name          = name;
        sensor.force_sensor  = force_name;
        sensor.torque_sensor = torque_name;
        sensor.force_adr     = r->model->sensor_adr[force_id];
        sensor.torque_adr    = r->model->sensor_adr[torque_id];
        if (spec.frame_site && spec.frame_site[0] != '\0') {
            sensor.frame_site    = spec.frame_site;
            sensor.frame_site_id = mj_name2id(r->model, mjOBJ_SITE, spec.frame_site);
            if (sensor.frame_site_id < 0) {
                LOG_ERROR(
                  "frame_site '" << spec.frame_site << "' not found for FT sensor '" << name << "'"
                );
                return false;
            }
        }
        r->ft_sensors.push_back(std::move(sensor));
    }
    return true;
}

// Ports that hold the robot where it is, in the mode its model is in.
static RobotPorts seeded_ports(const Robot &r)
{
    const RobotInternals &in = *r._impl;
    const mjData         *d  = r.data;
    const int             n  = r.n_joints;
    RobotPorts            p;
    p.ctrl_mode = r.ctrl_mode; // a requested switch still happens at the next update()
    p.jnt_pos_msr.resize(n);
    p.jnt_vel_msr.resize(n);
    p.jnt_trq_msr.resize(n);
    p.jnt_pos_cmd.resize(n);
    p.jnt_vel_cmd.assign(n, 0.0);
    p.jnt_trq_cmd.assign(n, 0.0);
    p.jnt_saturated.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        const int dof    = in.kdl_to_mj_dof[i];
        p.jnt_pos_msr[i] = d->qpos[in.kdl_to_mj_qpos[i]];
        p.jnt_vel_msr[i] = d->qvel[dof];
        p.jnt_trq_msr[i] = d->qfrc_actuator[dof];
        p.jnt_pos_cmd[i] = p.jnt_pos_msr[i];
    }
    return p;
}

static void register_robot(Robot *r, Env *env)
{
    if (std::find(env->robots.begin(), env->robots.end(), r) == env->robots.end())
        env->robots.push_back(r);
    r->_impl->env                 = env;
    static_cast<RobotPorts &>(*r) = seeded_ports(*r);
}

bool init_robot_from_mjcf(
  Robot               *r,
  Env                 *env,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix,
  const ToolFrameSpec *tool
)
{
    if (!r || !env || !env->model) return false;
    LOG_INFO(
      "init_robot_from_mjcf: '"
      << base_body << "' -> '" << tip_body << "' prefix='" << (prefix ? prefix : "") << "'"
      << (tool && tool->tool_body ? std::string(" tool='") + tool->tool_body + "'" : "")
      << (tool && tool->tcp_site ? std::string(" tcp='") + tool->tcp_site + "'" : "")
    );
    const auto lock  = lock_env(env);
    mjModel   *model = env->model;
    mjData    *data  = env->data;
    r->model         = model;
    r->data          = data;
    r->tip_T_tcp     = KDL::Frame::Identity();
    r->has_tcp_frame = false;
    r->tcp_site.clear();
    r->ft_sensors.clear();
    if (!build_kdl_from_model(r, model, base_body, tip_body)) return false;
    if (!build_index_map(r, prefix ? prefix : "")) return false;
    if (!resolve_ft_sensors(r, tool)) return false;

    KDL::Frame tip_T_tcp = KDL::Frame::Identity();
    bool       has_tcp   = false;
    if (tool && tool->tcp_site) {
        if (!get_site_frame_in_body(env, tip_body, tool->tcp_site, &tip_T_tcp)) {
            LOG_ERROR(
              "tcp_site '" << tool->tcp_site << "' or tip body '" << tip_body
                           << "' not found in model"
            );
            return false;
        }
        has_tcp          = true;
        r->tip_T_tcp     = tip_T_tcp;
        r->has_tcp_frame = true;
        r->tcp_site      = tool->tcp_site;
    } else if (tool && !Equal(tool->tcp_frame, KDL::Frame::Identity(), kIdentityTol)) {
        tip_T_tcp        = tool->tcp_frame;
        has_tcp          = true;
        r->tip_T_tcp     = tip_T_tcp;
        r->has_tcp_frame = true;
    }

    if (tool && tool->tool_body) {
        int tool_bid = mj_name2id(model, mjOBJ_BODY, tool->tool_body);
        if (tool_bid < 0) {
            LOG_ERROR("tool_body '" << tool->tool_body << "' not found in model");
            return false;
        }
        int tip_bid = mj_name2id(model, mjOBJ_BODY, tip_body);
        ensure_kinematics(env);
        std::vector<int>      subtree      = collect_subtree(model, tool_bid);
        KDL::RigidBodyInertia tool_inertia = compute_tool_inertia(model, data, tip_bid, subtree);
        LOG_INFO(
          "appending lumped tool inertia: " << subtree.size() << " bodies under '"
                                            << tool->tool_body << "'"
        );
        r->chain.addSegment(KDL::Segment(
          tool->tool_body, KDL::Joint(KDL::Joint::None), KDL::Frame::Identity(), tool_inertia
        ));
        // Fixed joints do not count: n_joints remains the same after addSegment.
    }

    if (has_tcp) {
        const std::string seg_name = (tool && tool->tcp_site) ? tool->tcp_site : "tcp";
        r->chain.addSegment(KDL::Segment(seg_name, KDL::Joint(KDL::Joint::None), tip_T_tcp));
    }

    LOG_INFO(
      "chain ready: " << r->n_joints << " joints [" << base_body << " -> " << tip_body << "]"
                      << (tool && tool->tool_body ? std::string(" + tool '") + tool->tool_body + "'"
                                                  : "")
                      << (has_tcp ? (tool && tool->tcp_site
                                       ? std::string(" tcp site '") + tool->tcp_site + "'"
                                       : " tcp frame (manual)")
                                  : "")
    );
    register_robot(r, env);
    return true;
}

bool init_robot_from_chain(
  Robot                          *r,
  Env                            *env,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix,
  const ToolFrameSpec            *tool
)
{
    if (!r || !env || !env->model) return false;
    const auto lock  = lock_env(env);
    mjModel   *model = env->model;
    LOG_INFO(
      "init_robot_from_chain: " << chain.getNrOfSegments() << " segments, " << chain.getNrOfJoints()
                                << " joints, prefix='" << (prefix ? prefix : "") << "'"
    );
    if (joint_names.size() != chain.getNrOfJoints()) {
        LOG_ERROR(
          "joint_names has " << joint_names.size() << " entries but the chain has "
                             << chain.getNrOfJoints() << " joints"
        );
        return false;
    }

    r->model         = model;
    r->data          = env->data;
    r->tip_T_tcp     = KDL::Frame::Identity();
    r->has_tcp_frame = false;
    r->tcp_site.clear();
    r->ft_sensors.clear();

    // The chain is authored, not derived: take it as given, tool segments included.
    r->chain       = chain;
    r->n_joints    = (int)chain.getNrOfJoints();
    r->joint_names = joint_names;

    // Limits stay a property of the simulated model, as they are for a derived chain.
    const std::string pfx = prefix ? prefix : "";
    r->joint_limits.clear();
    for (const auto &name : joint_names) {
        r->joint_limits.push_back(
          joint_range(model, mj_name2id(model, mjOBJ_JOINT, (pfx + name).c_str()))
        );
    }

    if (!build_index_map(r, pfx)) return false;
    if (!resolve_ft_sensors(r, tool)) return false;

    LOG_INFO(
      "chain adopted: " << r->n_joints << " joints, " << r->chain.getNrOfSegments() << " segments"
    );
    register_robot(r, env);
    return true;
}

const ForceTorqueSensor *find_ft_sensor(const Robot *r, const char *name)
{
    if (!r || !name) return nullptr;
    for (const auto &sensor : r->ft_sensors) {
        if (sensor.name == name) return &sensor;
    }
    return nullptr;
}

std::vector<double> joint_force_limits(const Robot *r, double fallback)
{
    std::vector<double> limits(r->n_joints, fallback);
    if (!r->model) return limits;
    const mjModel *m     = r->model;
    const bool     trq   = r->ctrl_mode == CtrlMode::TORQUE;
    const auto     bound = [](const mjtNum *range) {
        return std::max(std::abs(range[0]), std::abs(range[1]));
    };
    for (int i = 0; i < r->n_joints; ++i) {
        const int a = r->_impl->mode_ctrl[static_cast<int>(r->ctrl_mode)][i];
        if (a < 0) continue;
        const double gear = std::abs(m->actuator_gear[6 * a]);
        double       lim  = fallback;
        if (m->actuator_forcelimited[a]) lim = bound(m->actuator_forcerange + 2 * a) * gear;
        // TORQUE commands ctrl = tau / gear, clamped to ctrlrange.
        if (trq && m->actuator_ctrllimited[a])
            lim = std::min(lim, bound(m->actuator_ctrlrange + 2 * a) * gear);
        limits[i] = lim;
    }
    return limits;
}

void cleanup(Robot *r)
{
    if (!r) return;
    if (Env *env = r->_impl->env) {
        auto &robots = env->robots;
        robots.erase(std::remove(robots.begin(), robots.end(), r), robots.end());
    }
    r->model         = nullptr;
    r->data          = nullptr;
    r->chain         = KDL::Chain();
    r->tip_T_tcp     = KDL::Frame::Identity();
    r->has_tcp_frame = false;
    r->tcp_site.clear();
    r->n_joints = 0;
    r->joint_names.clear();
    r->joint_limits.clear();
    r->ft_sensors.clear();
    r->paused                     = false;
    static_cast<RobotPorts &>(*r) = RobotPorts{};
    *r->_impl                     = RobotInternals{};
}

void set_joint_pos(Robot *r, const KDL::JntArray &q)
{
    if (!r || !r->_impl->env) return;
    const auto lock = lock_env(r->_impl->env);
    const int  n    = std::min((int)q.rows(), r->n_joints);
    for (int i = 0; i < n; ++i) r->data->qpos[r->_impl->kdl_to_mj_qpos[i]] = q(i);
}

void set_body_pose(Env *env, const char *body_name, const double pos[3], const double *quat)
{
    if (!env || !env->model || !body_name) return;
    const mjModel *model = env->model;
    mjData        *data  = env->data;
    int            bid   = mj_name2id(model, mjOBJ_BODY, body_name);
    if (bid < 0) return;
    int jnt_start = model->body_jntadr[bid];
    int jnt_count = model->body_jntnum[bid];
    int jid       = -1;
    for (int k = 0; k < jnt_count; ++k) {
        if (model->jnt_type[jnt_start + k] == mjJNT_FREE) {
            jid = jnt_start + k;
            break;
        }
    }
    if (jid < 0) return;
    const auto lock      = lock_env(env);
    int        qadr      = model->jnt_qposadr[jid];
    int        dadr      = model->jnt_dofadr[jid];
    data->qpos[qadr]     = pos[0];
    data->qpos[qadr + 1] = pos[1];
    data->qpos[qadr + 2] = pos[2];
    data->qpos[qadr + 3] = quat ? quat[0] : 1.0;
    data->qpos[qadr + 4] = quat ? quat[1] : 0.0;
    data->qpos[qadr + 5] = quat ? quat[2] : 0.0;
    data->qpos[qadr + 6] = quat ? quat[3] : 0.0;
    for (int k = 0; k < 6; ++k) data->qvel[dadr + k] = 0.0;
}

// Simulation API

static bool step_viewer(Env *env); // step() with the simulate UI open, defined with it

static bool all_paused(const Env *env)
{
    return !env->robots.empty()
           && std::all_of(env->robots.begin(), env->robots.end(), [](const Robot *r) {
                  return r->paused;
              });
}

bool step(Env *env)
{
    if (!env || !env->model) return true;
    if (env->viewer._sim_ui) return step_viewer(env);
    if (all_paused(env)) return true;
    end_step(env);
    begin_step(env);
    return true;
}

/* Pacing is the caller's job, not step()'s: a physics call that sleeps spends a time budget it
 * does not own, and does so invisibly at the call site. A loop with no timing of its own calls
 * this to track wall time; a loop that paces itself reads realtime_factor_of() and scales its
 * own period instead. */
void pace_realtime(Env *env)
{
    using Clock = std::chrono::steady_clock;
    using Dur   = std::chrono::duration<double>;
    if (!env || !env->model || !env->viewer._sim_ui) return;
    Viewer        *v = &env->viewer;
    const mjModel *m = env->model;
    const double   wall_per_step =
      (v->realtime_factor > 0.0) ? m->opt.timestep / v->realtime_factor : 0.0;
    const auto now = Clock::now();
    if (wall_per_step > 0.0 && v->_tick_t.time_since_epoch().count() != 0) {
        /* In the clock's own duration, so that the deadline can be carried
         * forward below without a lossy conversion on every step. */
        const auto period = std::chrono::duration_cast<Clock::duration>(Dur(wall_per_step));
        const auto next   = v->_tick_t + period;
        if (now < next) {
            std::this_thread::sleep_until(next);
            /* Carry the deadline rather than restarting from the wake time:
             * sleep_until overshoots by tens of microseconds, and measuring the
             * next period from the moment we woke would fold that into every
             * step. The loss is per step, so it grows as the timestep shrinks
             * -- roughly 1% at 2 ms and 2% at 1 ms. */
            v->_tick_t = next;
            return;
        }
    }
    /* Already past the deadline: resynchronise instead of carrying it, so that
     * a loop coming back from a stall does not burst to catch up. */
    v->_tick_t = now;
}

/* The user's current speed setting; 0.0 means uncapped. Read without a lock because the render
 * thread only ever pushes into the rtf_step atomic -- realtime_factor itself is written on the
 * control thread, inside step(), where that atomic is drained. */
double realtime_factor_of(const Viewer *v) { return v ? v->realtime_factor : 1.0; }

static mjtNum clamp_ctrlrange(const mjModel *m, int ci, mjtNum u)
{
    if (m->actuator_ctrllimited[ci])
        u = std::clamp(u, m->actuator_ctrlrange[2 * ci], m->actuator_ctrlrange[2 * ci + 1]);
    return u;
}

static void read_robot(Robot *r)
{
    const RobotInternals &in = *r->_impl;
    const mjData         *d  = r->data;
    for (int i = 0; i < r->n_joints; ++i) {
        r->jnt_pos_msr[i] = d->qpos[in.kdl_to_mj_qpos[i]];
        r->jnt_vel_msr[i] = d->qvel[in.kdl_to_mj_dof[i]];
        r->jnt_trq_msr[i] = d->qfrc_actuator[in.kdl_to_mj_dof[i]];
    }
    for (auto &sensor : r->ft_sensors) {
        const double *f = d->sensordata + sensor.force_adr;
        const double *t = d->sensordata + sensor.torque_adr;
        sensor.wrench   = KDL::Wrench(KDL::Vector(f[0], f[1], f[2]), KDL::Vector(t[0], t[1], t[2]));
    }
}

static void read_scene(Env *env)
{
    const mjData *data = env->data;
    for (auto &slot : env->scene.joints) {
        if (slot.qpos_adr < 0) continue;
        slot.position = data->qpos[slot.qpos_adr];
        slot.velocity = data->qvel[slot.dof_adr];
        ++slot.seq;
    }
    for (auto &slot : env->scene.free_bodies) {
        if (slot.qpos_adr < 0) continue;
        const double *p = data->qpos + slot.qpos_adr;
        // MuJoCo stores the freejoint quaternion as [w x y z]; KDL takes [x y z w].
        slot.pose = KDL::Frame(
          KDL::Rotation::Quaternion(p[4], p[5], p[6], p[3]), KDL::Vector(p[0], p[1], p[2])
        );
        ++slot.seq;
    }
}

/* What reset() restores. Each part's runtime state lives in one struct that is assigned afresh,
 * so a field added to it is reset without a line here; a part handed to reset_parts() without a
 * reset_part() overload does not compile. */
static void reset_part(std::vector<Robot *> &robots, Env *env)
{
    for (Robot *r : robots) {
        static_cast<RobotPorts &>(*r) = seeded_ports(*r);
        for (auto &sensor : r->ft_sensors)
            static_cast<ForceTorqueReading &>(sensor) = ForceTorqueReading{};
        // Hold the reset pose in whatever mode the robot is in.
        const RobotInternals &in = *r->_impl;
        for (int i = 0; i < r->n_joints; ++i) {
            for (int mode = 0; mode < 3; ++mode) {
                const int a = in.mode_ctrl[mode][i];
                if (a < 0) continue;
                env->data->ctrl[a] = mode == static_cast<int>(CtrlMode::POSITION)
                                       ? env->model->actuator_gear[6 * a] * r->jnt_pos_msr[i]
                                       : 0.0;
            }
        }
    }
}

static void reset_part(SceneState &scene, Env *env)
{
    for (auto &slot : scene.joints) static_cast<SceneJointReading &>(slot) = SceneJointReading{};
    for (auto &slot : scene.free_bodies)
        static_cast<SceneFreeBodyReading &>(slot) = SceneFreeBodyReading{};
    for (auto &slot : scene.wrenches)
        static_cast<SceneWrenchCommand &>(slot) = SceneWrenchCommand{};
    for (auto &slot : scene.actuators) {
        const double held = slot.ctrl_id >= 0 ? env->data->ctrl[slot.ctrl_id] : 0.0;
        static_cast<SceneActuatorCommand &>(slot) = SceneActuatorCommand{ .command = held };
    }
}

static void reset_part(Viewer &viewer, Env *) { viewer._tick_t = {}; }

template<class Part>
concept Resettable = requires(Part &part, Env *env) { reset_part(part, env); };

template<Resettable... Parts> static void reset_parts(Env *env, Parts &...parts)
{
    (reset_part(parts, env), ...);
}

// reset_mujoco is false when the simulate UI has already reset the data itself.
static ResetInfo reset_env(Env *env, const ResetOptions *options, bool reset_mujoco)
{
    ResetInfo    info{};
    ResetOptions default_options;
    if (!options) options = &default_options;
    mjModel *model = env->model;
    mjData  *data  = env->data;

    if (reset_mujoco) {
        if (options->use_keyframe && options->keyframe >= 0 && options->keyframe < model->nkey) {
            mj_resetDataKeyframe(model, data, options->keyframe);
            info.used_keyframe = true;
            info.keyframe      = options->keyframe;
        } else {
            mj_resetData(model, data);
        }
    }

    mj_forward(model, data);
    record_kinematics(env);
    reset_parts(env, env->robots, env->scene, env->viewer);

    // After the re-seed, so the hook can prime commands; what it moves is read back below.
    ResetContext ctx;
    ctx.env     = env;
    ctx.model   = model;
    ctx.data    = data;
    ctx.options = options;
    ctx.info    = &info;
    if (env->on_reset) env->on_reset(&ctx);

    ensure_kinematics(env);
    for (Robot *r : env->robots) read_robot(r);
    read_scene(env);
    return info;
}

ResetInfo reset(Env *env, const ResetOptions *options)
{
    if (!env || !env->model) return {};
    const auto lock = lock_env(env);
    return reset_env(env, options, true);
}

static bool switch_group(Env *env, int robot, CtrlMode mode)
{
    mjModel  *m      = env->model;
    mjData   *d      = env->data;
    const int target = mode_group(robot, static_cast<int>(mode));
    bool      found  = false;
    for (int a = 0; a < m->nu; ++a) {
        if (m->actuator_group[a] != target) continue;
        found = true;
        // Seed the new actuator where the joint already is, so switching does not jump.
        switch (mode) {
        case CtrlMode::POSITION: d->ctrl[a] = d->actuator_length[a]; break;
        case CtrlMode::VELOCITY: d->ctrl[a] = d->actuator_velocity[a]; break;
        case CtrlMode::TORQUE: d->ctrl[a] = 0.0; break;
        }
    }
    if (!found) {
        LOG_ERROR("robot " << robot << " has no actuators for this control mode");
        return false;
    }
    for (int k = 0; k < 3; ++k) m->opt.disableactuator |= 1 << mode_group(robot, k);
    m->opt.disableactuator &= ~(1 << target);
    return true;
}

bool set_control_mode(Env *env, int robot, CtrlMode mode)
{
    if (!env || !env->model || robot < 0 || robot > 9) return false;
    const auto lock = lock_env(env);
    return switch_group(env, robot, mode);
}

// seed_ports: an explicit switch starts the new mode from where the joints are; a switch the
// caller asked for by setting ctrl_mode keeps the commands it set alongside.
static bool switch_control_mode(Robot *r, CtrlMode mode, bool seed_ports)
{
    RobotInternals &in  = *r->_impl;
    const int       idx = static_cast<int>(mode);
    for (int i = 0; i < r->n_joints; ++i) {
        if (in.mode_ctrl[idx][i] < 0) {
            LOG_ERROR(
              "joint '" << r->joint_names[i] << "' has no actuator for this control mode"
              << " (add it to RobotSpec::modes)"
            );
            return false;
        }
    }
    if (in.robot_index >= 0 && !switch_group(in.env, in.robot_index, mode)) return false;
    const mjData *d = r->data;
    for (int i = 0; seed_ports && i < r->n_joints; ++i) {
        r->jnt_pos_cmd[i] = d->qpos[in.kdl_to_mj_qpos[i]];
        r->jnt_vel_cmd[i] = d->qvel[in.kdl_to_mj_dof[i]];
        r->jnt_trq_cmd[i] = 0.0;
    }
    r->ctrl_mode    = mode;
    in.applied_mode = mode;
    in.mode_applied = true;
    return true;
}

bool set_control_mode(Robot *r, CtrlMode mode)
{
    if (!r || !r->_impl->env) return false;
    const auto lock = lock_env(r->_impl->env);
    return switch_control_mode(r, mode, true);
}

static void apply_robot(Robot *r)
{
    RobotInternals &in = *r->_impl;
    // ctrl_mode may have been set directly; switch before commanding the new mode's actuators.
    if (!in.mode_applied) return;
    if (r->ctrl_mode != in.applied_mode && !switch_control_mode(r, r->ctrl_mode, false))
        r->ctrl_mode = in.applied_mode;

    mjModel  *m    = r->model;
    mjData   *d    = r->data;
    const int mode = static_cast<int>(r->ctrl_mode);
    for (int i = 0; i < r->n_joints; ++i) {
        const int a = in.mode_ctrl[mode][i];
        if (a < 0) continue;
        const double gear = m->actuator_gear[6 * a];
        double       u    = 0.0;
        switch (r->ctrl_mode) {
        case CtrlMode::POSITION: u = gear * r->jnt_pos_cmd[i]; break;
        case CtrlMode::VELOCITY: u = gear * r->jnt_vel_cmd[i]; break;
        case CtrlMode::TORQUE: u = r->jnt_trq_cmd[i] / gear; break;
        }
        d->ctrl[a]          = clamp_ctrlrange(m, a, u);
        r->jnt_saturated[i] = d->ctrl[a] != u;
    }
}

static void apply_scene(Env *env)
{
    mjData *data = env->data;
    for (const auto &slot : env->scene.wrenches) {
        if (slot.body_id < 0) continue;
        double *target = data->xfrc_applied + 6 * slot.body_id;
        target[0]      = slot.wrench.force.x();
        target[1]      = slot.wrench.force.y();
        target[2]      = slot.wrench.force.z();
        target[3]      = slot.wrench.torque.x();
        target[4]      = slot.wrench.torque.y();
        target[5]      = slot.wrench.torque.z();
    }
    for (auto &slot : env->scene.actuators) {
        if (slot.ctrl_id < 0) continue;
        data->ctrl[slot.ctrl_id] = clamp_ctrlrange(env->model, slot.ctrl_id, slot.command);
        slot.saturated           = data->ctrl[slot.ctrl_id] != slot.command;
    }
}

void update(Env *env)
{
    if (!env || !env->model) return;
    const auto lock = lock_env(env);
    for (Robot *r : env->robots) read_robot(r);
    read_scene(env);
    for (Robot *r : env->robots) apply_robot(r);
    apply_scene(env);
}

// The free joint a body owns, or -1 when it owns none.
static int free_joint_of_body(const mjModel *model, int bid)
{
    const int start = model->body_jntadr[bid];
    const int count = model->body_jntnum[bid];
    for (int k = 0; k < count; ++k) {
        if (model->jnt_type[start + k] == mjJNT_FREE) return start + k;
    }
    return -1;
}

SceneJointSlot *bind_scene_joint(SceneState *s, const char *joint_name)
{
    if (!s || !s->model || !joint_name) {
        LOG_ERROR("bind_scene_joint: null scene state or name");
        return nullptr;
    }
    const int jid = mj_name2id(s->model, mjOBJ_JOINT, joint_name);
    if (jid < 0) {
        LOG_ERROR("bind_scene_joint: no joint named '" << joint_name << "'");
        return nullptr;
    }
    const int type = s->model->jnt_type[jid];
    if (type == mjJNT_FREE || type == mjJNT_BALL) {
        LOG_ERROR("bind_scene_joint: joint '" << joint_name << "' is not a scalar joint");
        return nullptr;
    }
    for (const auto &slot : s->joints) {
        if (slot.name == joint_name) {
            LOG_ERROR("bind_scene_joint: joint '" << joint_name << "' is already bound");
            return nullptr;
        }
    }
    s->joints.push_back(SceneJointSlot{
      {}, joint_name, s->model->jnt_qposadr[jid], s->model->jnt_dofadr[jid] });
    return &s->joints.back();
}

SceneFreeBodySlot *bind_scene_free_body(SceneState *s, const char *body_name)
{
    if (!s || !s->model || !body_name) {
        LOG_ERROR("bind_scene_free_body: null scene state or name");
        return nullptr;
    }
    const int bid = mj_name2id(s->model, mjOBJ_BODY, body_name);
    if (bid < 0) {
        LOG_ERROR("bind_scene_free_body: no body named '" << body_name << "'");
        return nullptr;
    }
    const int jid = free_joint_of_body(s->model, bid);
    if (jid < 0) {
        LOG_ERROR("bind_scene_free_body: body '" << body_name << "' owns no free joint");
        return nullptr;
    }
    for (const auto &slot : s->free_bodies) {
        if (slot.name == body_name) {
            LOG_ERROR("bind_scene_free_body: body '" << body_name << "' is already bound");
            return nullptr;
        }
    }
    s->free_bodies.push_back(SceneFreeBodySlot{ {}, body_name, s->model->jnt_qposadr[jid] });
    return &s->free_bodies.back();
}

SceneWrenchSlot *bind_scene_wrench(SceneState *s, const char *body_name)
{
    if (!s || !s->model || !body_name) {
        LOG_ERROR("bind_scene_wrench: null scene state or name");
        return nullptr;
    }
    const int bid = mj_name2id(s->model, mjOBJ_BODY, body_name);
    if (bid < 0) {
        LOG_ERROR("bind_scene_wrench: no body named '" << body_name << "'");
        return nullptr;
    }
    for (const auto &slot : s->wrenches) {
        if (slot.name == body_name) {
            LOG_ERROR("bind_scene_wrench: body '" << body_name << "' is already bound");
            return nullptr;
        }
    }
    s->wrenches.push_back(SceneWrenchSlot{ {}, body_name, bid });
    return &s->wrenches.back();
}

// The actuator of that name, or the one driving the joint of that name; -1 if neither.
static int actuator_for_name(const mjModel *m, const char *name)
{
    int aid = mj_name2id(m, mjOBJ_ACTUATOR, name);
    if (aid < 0) {
        // A model commands the joint it means; the actuator driving it carries its own name.
        const int jid = mj_name2id(m, mjOBJ_JOINT, name);
        for (int i = 0; aid < 0 && jid >= 0 && i < m->nu; ++i) {
            if (m->actuator_trntype[i] == mjTRN_JOINT && m->actuator_trnid[2 * i] == jid) aid = i;
        }
    }
    return aid;
}

SceneActuatorSlot *bind_scene_actuator(SceneState *s, const char *name)
{
    if (!s || !s->model || !name) {
        LOG_ERROR("bind_scene_actuator: null scene state or name");
        return nullptr;
    }
    const int aid = actuator_for_name(s->model, name);
    if (aid < 0) {
        LOG_ERROR("bind_scene_actuator: nothing actuates '" << name << "'");
        return nullptr;
    }
    for (const auto &slot : s->actuators) {
        if (slot.name == name) {
            LOG_ERROR("bind_scene_actuator: '" << name << "' is already bound");
            return nullptr;
        }
    }
    s->actuators.push_back(SceneActuatorSlot{ {}, name, aid });
    return &s->actuators.back();
}

// A slot whose name is gone from the rebuilt model is unbound, and skipped from then on.
static bool rebind_scene(Env *env)
{
    SceneState    *s     = &env->scene;
    const mjModel *model = env->model;
    s->model             = model;
    std::string unbound;
    for (auto &slot : s->joints) {
        const int  jid = mj_name2id(model, mjOBJ_JOINT, slot.name.c_str());
        const bool scalar =
          jid >= 0 && model->jnt_type[jid] != mjJNT_FREE && model->jnt_type[jid] != mjJNT_BALL;
        slot.qpos_adr = scalar ? model->jnt_qposadr[jid] : -1;
        slot.dof_adr  = scalar ? model->jnt_dofadr[jid] : -1;
        if (!scalar) unbound += " joint '" + slot.name + "'";
    }
    for (auto &slot : s->free_bodies) {
        const int bid = mj_name2id(model, mjOBJ_BODY, slot.name.c_str());
        const int jid = bid >= 0 ? free_joint_of_body(model, bid) : -1;
        slot.qpos_adr = jid >= 0 ? model->jnt_qposadr[jid] : -1;
        if (jid < 0) unbound += " free body '" + slot.name + "'";
    }
    for (auto &slot : s->wrenches) {
        slot.body_id = mj_name2id(model, mjOBJ_BODY, slot.name.c_str());
        if (slot.body_id < 0) unbound += " body '" + slot.name + "'";
    }
    for (auto &slot : s->actuators) {
        slot.ctrl_id = actuator_for_name(model, slot.name.c_str());
        if (slot.ctrl_id < 0) unbound += " actuator '" + slot.name + "'";
    }
    if (!unbound.empty()) LOG_ERROR("scene slots gone from the rebuilt model, unbound:" << unbound);
    return unbound.empty();
}

// GLFW/UI

static std::string realtime_factor_label(double realtime_factor)
{
    if (realtime_factor == 0.0) return "RTF: MAX";

    char buf[32] = {};
    std::snprintf(buf, sizeof(buf), "RTF: %.2fx", realtime_factor);
    return buf;
}

static void adjust_realtime_factor(Viewer *v, int direction)
{
    if (!v || direction == 0) return;

    constexpr double kStep   = 1.41421356237;
    constexpr double kMinRtf = 0.05;
    constexpr double kMaxRtf = 10.0;

    if (direction > 0) {
        if (v->realtime_factor == 0.0) return; // already uncapped/max-speed
        double next        = v->realtime_factor * kStep;
        v->realtime_factor = (next > kMaxRtf) ? 0.0 : next;
    } else {
        if (v->realtime_factor == 0.0) {
            v->realtime_factor = kMaxRtf;
        } else {
            v->realtime_factor = std::max(kMinRtf, v->realtime_factor / kStep);
        }
    }

    v->_tick_t = {};
    LOG_INFO(realtime_factor_label(v->realtime_factor));
}

/* Hint GLFW to use the Wayland backend on pure Wayland sessions.
 * On GLFW < 3.4 the platform select API does not exist; GLFW 3.3 auto-detects
 * via WAYLAND_DISPLAY, so this is a no-op for older installs.
 * Must be called before the first glfwInit(). */
static void apply_glfw_platform_hints()
{
#if defined(__linux__) && GLFW_VERSION_MAJOR * 100 + GLFW_VERSION_MINOR >= 304
    if (!getenv("DISPLAY") && getenv("WAYLAND_DISPLAY"))
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    else
        glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
#endif
}

/* Internal state for open_viewer(): bundles the mj::Simulate object so it
 * can be stored behind a void* in Viewer._sim_ui.
 *
 * Threading: GlfwAdapter (and therefore the GL context) is created INSIDE
 * render_thread so that glfwMakeContextCurrent() is called on the thread that
 * will own the context.  RenderLoop() runs on that same thread and processes
 * Load() requests from the main thread.  step() does physics only -- the
 * render thread handles all calls to Render(). */
struct SimUiState
{
    GLFWkeyfun                        prev_key_cb = nullptr; // Simulate's own, chained
    mjvCamera                         cam{};
    mjvOption                         opt{};
    mjvPerturb                        pert{};
    std::unique_ptr<mujoco::Simulate> sim;
    std::thread                       render_thread;
    bool                              sim_ready = false;
    std::mutex                        sim_ready_mtx;
    std::condition_variable           sim_ready_cv;
    double                            prev_sim_time = 0.0;
    int                               pert_body     = -1; // body step() last pushed; -1 none
    std::atomic<int>                  rtf_step{ 0 };      // + faster, - slower (render thread)
    GLFWwindow                       *glfw_window = nullptr;
    VideoRecorder                     recorder;
    bool                              recorder_active = false;
    int record_camera        = 0; // 0=current, 1=free, 2=tracking, 3+=fixed cam
    int record_frame_stride  = 1;
    int record_frame_counter = 0;
    /* User scene merged into each render frame by Simulate (overlay polylines,
     * e.g. the EE trajectory trace). Guarded by user_scn_mtx because the control
     * thread appends to it while the render thread reads it. */
    mjvScene   user_scn{};
    std::mutex user_scn_mtx;
    /* Key state as the render thread's key callback sees it, so that a caller
     * driving physics on another thread can read the keyboard without touching
     * GLFW, which requires its window calls on the owning thread. */
    std::atomic<bool> keys[GLFW_KEY_LAST + 1]{};
    /* Keys the caller has claimed: recorded above, but withheld from the UI's
     * own handler so that its bindings do not fight the caller's. */
    std::atomic<bool> captured[GLFW_KEY_LAST + 1]{};
};

static std::unique_lock<std::recursive_mutex> lock_env(const Env *env)
{
    const auto *ss = env ? static_cast<SimUiState *>(env->viewer._sim_ui) : nullptr;
    if (!ss || !ss->sim) return {};
    return std::unique_lock<std::recursive_mutex>(ss->sim->mtx);
}

// GLFW key callbacks carry only the window; this finds the viewer that owns it.
static std::mutex                                     g_windows_mtx;
static std::unordered_map<GLFWwindow *, SimUiState *> g_windows;

static SimUiState *viewer_of(GLFWwindow *w)
{
    std::lock_guard<std::mutex> lk(g_windows_mtx);
    const auto                  found = g_windows.find(w);
    return found == g_windows.end() ? nullptr : found->second;
}

static VideoResolution recorder_resolution_from_index(int index)
{
    switch (index) {
    case 0:
        return VideoResolution::R360p;
    case 1:
        return VideoResolution::R480p;
    case 3:
        return VideoResolution::R1080p;
    case 2:
    default:
        return VideoResolution::R720p;
    }
}

static void handle_recorder_request(SimUiState *ss, mjModel *m)
{
    if (!ss || !ss->sim || !m) return;

    char path[mujoco::Simulate::kMaxFilenameLength] = {};
    int  camera                                     = 0;
    int  resolution                                 = kRecorderDefaultResIndex;
    int  fps                                        = kRecorderDefaultFps;
    int  request =
      ss->sim->ConsumeWrapperRecordRequest(path, sizeof(path), &camera, &resolution, &fps);
    if (!request) return;

    if (request == 2) {
        if (ss->recorder_active) cleanup(&ss->recorder);
        ss->recorder_active      = false;
        ss->record_frame_counter = 0;
        ss->sim->SetWrapperRecorderState(0);
        return;
    }

    if (ss->recorder_active) cleanup(&ss->recorder);

    const char *out_path = path[0] ? path : "recording.mp4";

    /* Every camera choice records offscreen, "Current" included: record_sim_ui_frame copies the
       live GUI camera into the recorder each frame, so the GUI never stalls on a readback. */
    if (!init_video_recorder(
          &ss->recorder, m, out_path, recorder_resolution_from_index(resolution), std::max(1, fps)
        )) {
        ss->recorder_active      = false;
        ss->record_frame_counter = 0;
        ss->sim->SetWrapperRecorderState(2);
        return;
    }

    ss->recorder_active      = true;
    ss->record_camera        = camera;
    ss->record_frame_counter = 0;
    ss->record_frame_stride =
      std::max(1, (int)std::lround(1.0 / (std::max(1, fps) * m->opt.timestep)));
    ss->sim->SetWrapperRecorderState(1);
}

static void record_sim_ui_frame(SimUiState *ss, Env *env)
{
    const mjModel *m = env->model;
    if (!ss || !ss->recorder_active) return;
    if (++ss->record_frame_counter < ss->record_frame_stride) return;
    ss->record_frame_counter = 0;

    {
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        ss->recorder.opt = ss->opt;
        ss->recorder.cam = ss->cam;
        if (ss->record_camera == 1) {
            ss->recorder.cam.type       = mjCAMERA_FREE;
            ss->recorder.cam.fixedcamid = -1;
        } else if (ss->record_camera == 2) {
            if (ss->sim->pert.select > 0) {
                ss->recorder.cam.type        = mjCAMERA_TRACKING;
                ss->recorder.cam.trackbodyid = ss->sim->pert.select;
                ss->recorder.cam.fixedcamid  = -1;
            } else {
                ss->recorder.cam.type       = mjCAMERA_FREE;
                ss->recorder.cam.fixedcamid = -1;
            }
        } else if (ss->record_camera >= 3 && ss->record_camera - 3 < m->ncam) {
            ss->recorder.cam.type       = mjCAMERA_FIXED;
            ss->recorder.cam.fixedcamid = ss->record_camera - 3;
        }
    }

    if (!record_frame(&ss->recorder, env)) {
        cleanup(&ss->recorder);
        ss->recorder_active = false;
        ss->sim->SetWrapperRecorderState(2);
    }
}

static void sim_ui_key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    SimUiState *ss = viewer_of(w);
    if (!ss) return;
    /* Record the state for key_pressed() before anything consumes the event, so
     * that a caller on the physics thread sees every key the window receives.
     * A key the caller has claimed stops here and never reaches the UI. */
    if (key >= 0 && key <= GLFW_KEY_LAST) {
        if (action == GLFW_PRESS)
            ss->keys[key].store(true, std::memory_order_relaxed);
        else if (action == GLFW_RELEASE)
            ss->keys[key].store(false, std::memory_order_relaxed);
        if (ss->captured[key].load(std::memory_order_relaxed)) return;
    }

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_PERIOD) {
            ss->rtf_step.fetch_add(+1);
            return;
        }
        if (key == GLFW_KEY_COMMA) {
            ss->rtf_step.fetch_add(-1);
            return;
        }
    }
    if (ss->prev_key_cb) ss->prev_key_cb(w, key, scancode, action, mods);
}

bool use_camera(Viewer *v, const mjModel *model, const char *name)
{
    if (!v || !model) return false;
    if (v->_sim_ui) {
        auto                                  *ss = static_cast<SimUiState *>(v->_sim_ui);
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        return use_camera_impl(&ss->cam, model, name);
    }
    return use_camera_impl(&v->cam, model, name);
}

static void set_free_camera_impl(
  mjvCamera                   *cam,
  double                       distance,
  double                       azimuth,
  double                       elevation,
  const std::array<double, 3> &lookat
)
{
    cam->type       = mjCAMERA_FREE;
    cam->fixedcamid = -1;
    cam->distance   = distance;
    cam->azimuth    = azimuth;
    cam->elevation  = elevation;
    cam->lookat[0]  = lookat[0];
    cam->lookat[1]  = lookat[1];
    cam->lookat[2]  = lookat[2];
}

void set_free_camera(
  Viewer                      *v,
  double                       distance,
  double                       azimuth,
  double                       elevation,
  const std::array<double, 3> &lookat
)
{
    if (!v) return;
    if (v->_sim_ui) {
        auto                                  *ss = static_cast<SimUiState *>(v->_sim_ui);
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        set_free_camera_impl(&ss->cam, distance, azimuth, elevation, lookat);
        ss->sim->camera = 0;
        return;
    }
    set_free_camera_impl(&v->cam, distance, azimuth, elevation, lookat);
}

static void close_viewer(Env *env)
{
    Viewer *v = &env->viewer;
    if (!v->_sim_ui) return;
    auto *ss             = static_cast<SimUiState *>(v->_sim_ui);
    ss->sim->exitrequest = 1;
    if (ss->render_thread.joinable()) ss->render_thread.join();
    {
        std::lock_guard<std::mutex> lk(g_windows_mtx);
        g_windows.erase(ss->glfw_window);
    }
    if (ss->recorder_active) cleanup(&ss->recorder);
    /* Render thread has stopped, so no one is reading user_scn now. */
    mjv_freeScene(&ss->user_scn);
    delete ss;
    v->_sim_ui = nullptr;
}

void clear_trace(Viewer *v)
{
    if (!v || !v->_sim_ui) return; // headless / no window
    auto                       *ss = static_cast<SimUiState *>(v->_sim_ui);
    std::lock_guard<std::mutex> lk(ss->user_scn_mtx);
    ss->user_scn.ngeom = 0;
}

void add_trace_segment(Viewer *v, const KDL::Vector &a, const KDL::Vector &b, const float rgba[4])
{
    if (!v || !v->_sim_ui) return; // headless / no window
    auto                       *ss = static_cast<SimUiState *>(v->_sim_ui);
    std::lock_guard<std::mutex> lk(ss->user_scn_mtx);
    if (ss->user_scn.ngeom >= ss->user_scn.maxgeom) return;
    mjvGeom *g = &ss->user_scn.geoms[ss->user_scn.ngeom++];

    static constexpr float kDefault[4] = { 1.0f, 0.5f, 0.1f, 1.0f }; // warm orange
    const float           *col         = rgba ? rgba : kDefault;
    mjv_initGeom(g, mjGEOM_LINE, /*size=*/nullptr, /*pos=*/nullptr, /*mat=*/nullptr, col);

    const mjtNum from[3] = { a.x(), a.y(), a.z() };
    const mjtNum to[3]   = { b.x(), b.y(), b.z() };
    mjv_connector(g, mjGEOM_LINE, /*width=*/3.0, from, to);
}

void add_overlay_arrow(
  Viewer            *v,
  const KDL::Vector &from,
  const KDL::Vector &dir,
  double             length,
  const float        rgba[4]
)
{
    if (!v || !v->_sim_ui) return; // headless / no window
    KDL::Vector unit = dir;
    if (unit.Normalize() < 1e-9 || length <= 0.0) return; // nothing to point at
    auto                       *ss = static_cast<SimUiState *>(v->_sim_ui);
    std::lock_guard<std::mutex> lk(ss->user_scn_mtx);
    if (ss->user_scn.ngeom >= ss->user_scn.maxgeom) return;
    mjvGeom *g = &ss->user_scn.geoms[ss->user_scn.ngeom++];

    static constexpr float kDefault[4] = { 1.0f, 0.5f, 0.1f, 1.0f }; // warm orange
    const float           *col         = rgba ? rgba : kDefault;
    mjv_initGeom(g, mjGEOM_ARROW, /*size=*/nullptr, /*pos=*/nullptr, /*mat=*/nullptr, col);

    const KDL::Vector tip  = from + unit * length;
    const mjtNum      a[3] = { from.x(), from.y(), from.z() };
    const mjtNum      b[3] = { tip.x(), tip.y(), tip.z() };
    mjv_connector(g, mjGEOM_ARROW, /*width=*/0.02, a, b);
}

bool is_running(const Viewer *v)
{
    if (!v || !v->_sim_ui) return false;
    auto *ss = static_cast<SimUiState *>(v->_sim_ui);
    return ss->sim && !ss->sim->exitrequest.load();
}

void capture_key(Viewer *v, int glfw_key, bool capture)
{
    if (!v || !v->_sim_ui || glfw_key < 0 || glfw_key > GLFW_KEY_LAST) return;
    auto *ss = static_cast<SimUiState *>(v->_sim_ui);
    if (ss) ss->captured[glfw_key].store(capture, std::memory_order_relaxed);
}

bool key_pressed(const Viewer *v, int glfw_key)
{
    if (!v || !v->_sim_ui || glfw_key < 0 || glfw_key > GLFW_KEY_LAST) return false;
    auto *ss = static_cast<SimUiState *>(v->_sim_ui);
    return ss->keys[glfw_key].load(std::memory_order_relaxed);
}

bool open_viewer(Env *env, const char *title)
{
    if (!env || !env->model || env->viewer._sim_ui) return false;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;
    Viewer  *v = &env->viewer;
    mjModel *m = env->model;
    mjData  *d = env->data;

    // glfwInitHint() is "main thread only" per GLFW docs -- call before spawning.
    apply_glfw_platform_hints();

    auto *ss = new SimUiState();
    mjv_defaultCamera(&ss->cam);
    mjv_defaultOption(&ss->opt);
    mjv_defaultPerturb(&ss->pert);
    /* If the caller configured Viewer.cam before open_viewer (e.g. set a
     * named camera via use_camera() or a free-camera distance > 0), apply it. */
    if (v->cam.type != mjCAMERA_FREE || v->cam.distance > 0.0) ss->cam = v->cam;

    /* Create GlfwAdapter INSIDE the render thread so glfwMakeContextCurrent() is
     * called there and the GL context is owned by that thread.  RenderLoop() then
     * runs correctly (gladLoadGL, mjr_makeContext, and all Render() calls stay on
     * the same thread as the context). */
    ss->render_thread = std::thread([ss]() {
        namespace mj = mujoco;
        ss->sim      = std::make_unique<mj::Simulate>(
          std::make_unique<mj::GlfwAdapter>(), &ss->cam, &ss->opt, &ss->pert, false
        );
        ss->sim->font = 2; // preferred 150%; overwritten by RenderLoop HiDPI detection
        /* GlfwAdapter constructor calls glfwMakeContextCurrent, so
         * glfwGetCurrentContext() returns the SimUI window on this thread.
         * Install a chained key callback so ,/. speed keys reach sim_ui_key_cb. */
        ss->glfw_window = glfwGetCurrentContext();
        if (ss->glfw_window) {
            {
                std::lock_guard<std::mutex> lk(g_windows_mtx);
                g_windows[ss->glfw_window] = ss;
            }
            ss->prev_key_cb = glfwSetKeyCallback(ss->glfw_window, sim_ui_key_cb);
        }
        {
            std::lock_guard<std::mutex> lk(ss->sim_ready_mtx);
            ss->sim_ready = true;
        }
        ss->sim_ready_cv.notify_one();
        ss->sim->RenderLoop(); // blocks; GL context lives here
    });

    // Wait for render thread to construct the Simulate object.
    {
        std::unique_lock<std::mutex> lk(ss->sim_ready_mtx);
        ss->sim_ready_cv.wait(lk, [ss] { return ss->sim_ready; });
    }

    /* Send load request from this thread; the render loop processes it.
     * RenderLoop() runs ComputeFontScale() on HiDPI displays (200% on 2x),
     * overriding the font we set above.  Load() calls RefreshMjrContext with
     * the current font value, so we correct it here after the first load. */
    ss->sim->LoadMessage(title);
    ss->sim->Load(m, d, title);
    ss->sim->SetWrapperRealtimeFactor(v->realtime_factor);
    if (ss->sim->font != 2) {
        ss->sim->font = 2; // 150%: 0=50% 1=100% 2=150% 3=200%
        ss->sim->Load(m, d, title);
        ss->sim->SetWrapperRealtimeFactor(v->realtime_factor);
    }
    {
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        ensure_kinematics(env);
    }

    /* Allocate the overlay user scene and hand it to Simulate, which merges it
     * into every rendered frame (simulate.h: Simulate::user_scn). add_trace_segment()
     * appends geoms here from the control thread. */
    mjv_defaultScene(&ss->user_scn);
    // Overlay geom budget for add_trace_segment(); caps the DSL trace-length
    // (mj:trace-length maxInclusive in simulation/mujoco.shacl.ttl).
    mjv_makeScene(m, &ss->user_scn, /*maxgeom=*/8192);
    ss->user_scn.ngeom = 0;
    ss->sim->user_scn  = &ss->user_scn;

    v->_sim_ui = ss;
    return true;
}

// Show a rebuilt pair; call before the old one is freed.
static void reload_viewer(Env *env, mjModel *m, mjData *d)
{
    Viewer *v = &env->viewer;
    if (!v->_sim_ui) return;
    auto *ss = static_cast<SimUiState *>(v->_sim_ui);
    // Camera, geom and site ids move with a recompile, so a recording cannot carry on.
    if (ss->recorder_active) {
        LOG_WARN("scene rebuilt, recording stopped");
        cleanup(&ss->recorder);
        ss->recorder_active      = false;
        ss->record_frame_counter = 0;
        ss->sim->SetWrapperRecorderState(0);
    }
    ss->sim->Load(m, d, ss->sim->filename);
    {
        // Sync() copies user_scn under sim->mtx; the control thread appends under user_scn_mtx.
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        std::lock_guard<std::mutex>            lk(ss->user_scn_mtx);
        mjv_freeScene(&ss->user_scn);
        mjv_makeScene(m, &ss->user_scn, /*maxgeom=*/8192);
        ss->user_scn.ngeom = 0;
        mj_forward(m, d);
    }
}

static bool step_viewer(Env *env)
{
    Viewer  *v = &env->viewer;
    mjModel *m = env->model;
    mjData  *d = env->data;
    {
        auto *ss  = static_cast<SimUiState *>(v->_sim_ui);
        auto *sim = ss->sim.get();

        if (sim->exitrequest.load()) return false;

        // Speed control: ,/. keys are intercepted by sim_ui_key_cb.
        int  rtf_step    = ss->rtf_step.exchange(0);
        bool rtf_changed = false;
        while (rtf_step > 0) {
            adjust_realtime_factor(v, +1);
            rtf_changed = true;
            --rtf_step;
        }
        while (rtf_step < 0) {
            adjust_realtime_factor(v, -1);
            rtf_changed = true;
            ++rtf_step;
        }
        if (rtf_changed) sim->SetWrapperRealtimeFactor(v->realtime_factor);
        handle_recorder_request(ss, m);


        {
            /* step() is the sole physics driver; the render thread only renders.
             * Honour sim->run as the pause flag so the Simulate UI Space-bar
             * and Pause/Run radio button work correctly. */
            std::unique_lock<std::recursive_mutex> lock(sim->mtx);

            // Detect a UI-driven reset: time jumped back (or to zero) while the
            // simulation had already advanced.  Guard with prev_sim_time > timestep
            // to avoid false triggers on the very first tick or when the user
            // scrubs to t=0 from a paused state before any physics has run.
            const bool time_jumped_back =
              ss->prev_sim_time > m->opt.timestep && d->time < ss->prev_sim_time - kSimTimeEps;
            if (time_jumped_back) (void)reset_env(env, nullptr, false);
            ss->prev_sim_time = d->time;

            if (sim->run && !all_paused(env)) {
                // Only the dragged body's xfrc_applied is ours: user wrenches elsewhere survive.
                const int dragged = (sim->pert.active | sim->pert.active2) ? sim->pert.select : -1;
                if (ss->pert_body >= 0 && ss->pert_body != dragged)
                    mju_zero(d->xfrc_applied + 6 * ss->pert_body, 6);
                ss->pert_body = dragged;
                mjv_applyPerturbForce(m, d, &sim->pert);
                end_step(env);
                begin_step(env);
                sim->AddToHistory();
            } else {
                ensure_kinematics(env);
            }
        }

        record_sim_ui_frame(ss, env);

        // Render is handled by the render thread inside RenderLoop().
        return !sim->exitrequest.load();
    }
}

// VideoRecorder -- EGL headless offscreen recording via ffmpeg pipe

#ifdef MJ_KDL_HAS_EGL

struct VideoRecorderImpl
{
    EGLDisplay           egl_dpy = EGL_NO_DISPLAY;
    EGLContext           egl_ctx = EGL_NO_CONTEXT;
    mjvScene             scn{};
    mjrContext           con{};
    FfmpegSink           sink;
    int                  width  = 0;
    int                  height = 0;
    std::vector<uint8_t> rgb_buf;
};

static constexpr EGLint kEglMaxDevices = 8;

/* An initialized EGL display for offscreen rendering, or EGL_NO_DISPLAY.

   EGL_DEFAULT_DISPLAY names no native display, so the loader guesses a platform from the
   process's window-system state; in a viewer process that already holds a Wayland context it
   can answer EGL_NO_DISPLAY with no error set. Name a GPU device instead, which is what
   offscreen rendering wants anyway, and keep the default as the fallback for loaders without
   EGL_EXT_platform_device. */
static EGLDisplay vr_egl_display(EGLint *major, EGLint *minor)
{
    auto query_devices =
      reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto platform_display =
      reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"
      ));

    EGLDeviceEXT devices[kEglMaxDevices];
    EGLint       count = 0;
    if (query_devices && platform_display && query_devices(kEglMaxDevices, devices, &count)) {
        for (EGLint i = 0; i < count; ++i) {
            EGLDisplay dpy = platform_display(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
            if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, major, minor)) return dpy;
        }
    }

    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, major, minor)) return dpy;
    return EGL_NO_DISPLAY;
}

static bool vr_egl_init(VideoRecorderImpl *impl)
{
    const EGLint attrs[] = {
        EGL_RED_SIZE,          kEglChannelBits, EGL_GREEN_SIZE,   kEglChannelBits,
        EGL_BLUE_SIZE,         kEglChannelBits, EGL_ALPHA_SIZE,   kEglChannelBits,
        EGL_DEPTH_SIZE,        kEglDepthBits,   EGL_STENCIL_SIZE, kEglChannelBits,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,  EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,   EGL_OPENGL_BIT,  EGL_NONE
    };

    EGLint major = 0, minor = 0;
    impl->egl_dpy = vr_egl_display(&major, &minor);
    if (impl->egl_dpy == EGL_NO_DISPLAY) {
        LOG_ERROR("EGL: no device display and no default display could be initialized");
        return false;
    }

    EGLConfig cfg;
    EGLint    n = 0;
    if (!eglChooseConfig(impl->egl_dpy, attrs, &cfg, 1, &n) || n == 0) {
        LOG_ERROR("EGL: no suitable config");
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        LOG_ERROR("EGL: bind OpenGL API failed");
        return false;
    }

    impl->egl_ctx = eglCreateContext(impl->egl_dpy, cfg, EGL_NO_CONTEXT, nullptr);
    if (impl->egl_ctx == EGL_NO_CONTEXT) {
        LOG_ERROR("EGL: context creation failed (error 0x" << std::hex << eglGetError() << ")");
        return false;
    }

    if (!eglMakeCurrent(impl->egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, impl->egl_ctx)) {
        LOG_ERROR("EGL: make current failed (error 0x" << std::hex << eglGetError() << ")");
        return false;
    }

    LOG_INFO("EGL " << major << "." << minor << " headless context ready");
    return true;
}

static void vr_egl_done(VideoRecorderImpl *impl)
{
    if (impl->egl_dpy == EGL_NO_DISPLAY) return;
    eglMakeCurrent(impl->egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (impl->egl_ctx != EGL_NO_CONTEXT) {
        eglDestroyContext(impl->egl_dpy, impl->egl_ctx);
        impl->egl_ctx = EGL_NO_CONTEXT;
    }
    eglTerminate(impl->egl_dpy);
    impl->egl_dpy = EGL_NO_DISPLAY;
}

bool init_offscreen(VideoRecorder *vr, mjModel *model, int width, int height)
{
    if (!vr || !model) return false;

    // Set up default camera and options on the user-visible struct.
    mjv_defaultCamera(&vr->cam);
    mjv_defaultFreeCamera(model, &vr->cam);
    mjv_defaultOption(&vr->opt);

    auto *impl   = new VideoRecorderImpl();
    impl->width  = width;
    impl->height = height;

    if (!vr_egl_init(impl)) {
        delete impl;
        return false;
    }

    // MuJoCo rendering contexts.
    mjv_defaultScene(&impl->scn);
    mjr_defaultContext(&impl->con);
    mjv_makeScene(model, &impl->scn, 4000);
    mjr_makeContext(model, &impl->con, mjFONTSCALE_150);
    mjr_setBuffer(mjFB_OFFSCREEN, &impl->con);
    mjr_resizeOffscreen(width, height, &impl->con);

    vr->_impl = impl;
    return true;
}

bool init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width,
  int            height,
  int            fps
)
{
    if (!out_path) return false;
    if (!init_offscreen(vr, model, width, height)) return false;

    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    impl->rgb_buf.resize(
      static_cast<size_t>(width) * static_cast<size_t>(height) * kRgbBytesPerPixel
    );

    if (!sink_open(&impl->sink, out_path, width, height, width, height, fps, true)) {
        cleanup(vr);
        return false;
    }

    LOG_INFO("VideoRecorder: " << width << "x" << height << " @ " << fps << " fps -> " << out_path);
    return true;
}

// Renders into `out` the way MuJoCo fills it: bottom row first.
static bool render_bottom_up(VideoRecorder *vr, Env *env, std::uint8_t *out)
{
    if (!vr || !vr->_impl || !env || !env->model || !out) return false;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);

    // One EGL context per recorder: make this one current before rendering.
    eglMakeCurrent(impl->egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, impl->egl_ctx);

    {
        const auto lock = lock_env(env);
        mjv_updateScene(env->model, env->data, &vr->opt, nullptr, &vr->cam, mjCAT_ALL, &impl->scn);
    }

    // The overlay geoms a window shows live on the viewer's user scene, and this offscreen
    // scene is rebuilt from the model every frame -- so append them the same way the UI thread
    // does (simulate.cc), or a recording loses every trace segment and every arrow.
    if (env->viewer._sim_ui) {
        auto                       *ss = static_cast<SimUiState *>(env->viewer._sim_ui);
        std::lock_guard<std::mutex> lk(ss->user_scn_mtx);
        const int ngeom = std::min(ss->user_scn.ngeom, impl->scn.maxgeom - impl->scn.ngeom);
        if (ngeom > 0) {
            std::memcpy(
              impl->scn.geoms + impl->scn.ngeom, ss->user_scn.geoms, ngeom * sizeof(mjvGeom)
            );
            impl->scn.ngeom += ngeom;
        }
    }

    mjrRect vp = { 0, 0, impl->width, impl->height };
    mjr_render(vp, &impl->scn, &impl->con);
    mjr_readPixels(out, nullptr, vp, &impl->con);
    return true;
}

bool render_rgb(VideoRecorder *vr, Env *env, std::uint8_t *out)
{
    if (!render_bottom_up(vr, env, out)) return false;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);

    // Callers of the public API get the image top-down, as every image format wants it.
    const int            row_bytes = kRgbBytesPerPixel * impl->width;
    std::vector<uint8_t> tmp(static_cast<size_t>(row_bytes));
    for (int top = 0, bot = impl->height - 1; top < bot; ++top, --bot) {
        memcpy(tmp.data(), out + top * row_bytes, row_bytes);
        memcpy(out + top * row_bytes, out + bot * row_bytes, row_bytes);
        memcpy(out + bot * row_bytes, tmp.data(), row_bytes);
    }
    return true;
}

bool record_frame(VideoRecorder *vr, Env *env)
{
    if (!vr || !vr->_impl) return false;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    if (!impl->sink.pipe) return false;
    // No flip here: the sink's filter chain turns the frame over on its way into the encoder.
    if (!render_bottom_up(vr, env, impl->rgb_buf.data())) return false;
    return sink_write(&impl->sink, impl->rgb_buf.data(), impl->rgb_buf.size());
}

bool init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps
)
{
    /* 16:9 width for each named height, rounded up to even: 480p is 853.3 wide, and the encoder
       takes no odd dimension. */
    int h = static_cast<int>(resolution);
    int w = ((h * 16 / 9) + 1) / 2 * 2;
    return init_video_recorder(vr, model, out_path, w, h, fps);
}

void cleanup(VideoRecorder *vr)
{
    if (!vr || !vr->_impl) return;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    sink_close(&impl->sink);
    mjr_freeContext(&impl->con);
    mjv_freeScene(&impl->scn);
    vr_egl_done(impl);
    delete impl;
    vr->_impl = nullptr;
}

#else // MJ_KDL_HAS_EGL not defined

bool init_video_recorder(VideoRecorder *, mjModel *, const char *, int, int, int)
{
    LOG_ERROR("VideoRecorder requires EGL; rebuild with -DBUILD_RECORDER=ON");
    return false;
}
bool record_frame(VideoRecorder *, Env *) { return false; }
bool init_offscreen(VideoRecorder *, mjModel *, int, int)
{
    LOG_ERROR("offscreen rendering requires EGL; rebuild with -DBUILD_RECORDER=ON");
    return false;
}
bool render_rgb(VideoRecorder *, Env *, std::uint8_t *) { return false; }
void cleanup(VideoRecorder *vr)
{
    if (vr) vr->_impl = nullptr;
}

#endif // MJ_KDL_HAS_EGL

} // namespace mj_kdl
