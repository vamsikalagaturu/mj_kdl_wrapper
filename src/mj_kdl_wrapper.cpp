/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <dlfcn.h>

#include <kdl/frames.hpp>

#include <algorithm>
#include <iterator>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <span>
#include <string_view>
#include <unordered_map>


namespace mj_kdl {

// Next to this library first, so an installed tree keeps its plugins after it is moved.
static std::string default_mujoco_plugin_dir()
{
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&default_mujoco_plugin_dir), &info) && info.dli_fname) {
        const auto dir = std::filesystem::path(info.dli_fname).parent_path() / "mujoco_plugin";
        if (std::filesystem::is_directory(dir)) return dir.string();
    }
    return MUJOCO_PLUGIN_DIR;
}

static void ensure_plugins_loaded()
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
    std::vector<int> mode_ctrl[3]; // per CtrlMode: the joint's actuator, -1 if none
    std::vector<int> mode_owners;  // owners of the joints' mode groups (1 + 3 * owner + mode)
    CtrlMode         applied_mode = CtrlMode::POSITION; // mode whose group is enabled
    bool             mode_applied = false;              // applied_mode is live in the model
    std::string      mj_prefix;                         // re-resolves the joints after a rebuild
};

struct NameHash
{
    using is_transparent = void;
    std::size_t operator()(std::string_view s) const { return std::hash<std::string_view>{}(s); }
};

using NameMap = std::unordered_map<std::string, int, NameHash, std::equal_to<>>;

struct EnvInternals
{
    // What the last mj_step1/mj_forward read; its results stay current while the state matches.
    std::vector<mjtNum>  computed_from;
    std::vector<mjtNum>  state_now; // kinematics_current's buffer
    bool                 computed = false;
    NameMap              names[mjNOBJECT];      // mj_name2id per object type
    bool                 adopted       = false; // model/data came from Env::adopt
    std::uint64_t        model_gen     = 0; // unique per compiled model, so recorders see a rebuild
    bool                 in_reset_hook = false; // a rebuild would swap the model under the hook
    mutable std::mutex   overlay_mtx;
    std::vector<mjvGeom> overlays; // world-space, model independent
};

// Source of EnvInternals::model_gen: process-wide, so no two models ever share a value.
static std::atomic<std::uint64_t> g_model_gen{ 0 };

Robot::Robot() : _impl(std::make_unique<RobotInternals>()) {}
Robot::~Robot() { cleanup(this); }
Env::Env() : _impl(std::make_unique<EnvInternals>()) {}
Env::~Env() { cleanup(this); }

/* Poses and position/velocity sensors (what mj_step1 computes, and mj_forward with it) are
 * recomputed only when the state they came from has changed. Compared, not trusted: a caller
 * may write qpos, qvel or mocap between two calls. */
static constexpr int kKinematicsInputs =
  mjSTATE_FULLPHYSICS | mjSTATE_EQ_ACTIVE | mjSTATE_MOCAP_POS | mjSTATE_MOCAP_QUAT;

static bool kinematics_current(const Env *env)
{
    EnvInternals &in = *env->_impl;
    if (!in.computed) return false;
    in.state_now.resize(mj_stateSize(env->model, kKinematicsInputs));
    mj_getState(env->model, env->data, in.state_now.data(), kKinematicsInputs);
    return in.state_now == in.computed_from;
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

static int cached_name2id(Env *env, mjtObj type, const char *name)
{
    auto      &names = env->_impl->names[type];
    const auto found = names.find(std::string_view(name));
    if (found != names.end()) return found->second;
    const int id = mj_name2id(env->model, type, name);
    names.emplace(name, id);
    return id;
}


// Spec-API helpers

// Buffer size for MuJoCo error strings returned by mj_parseXML / mj_saveLastXML.
static constexpr size_t kMjErrBuf = 2048;

// assets/stage/default.xml, compiled in by CMake so a scene needs no file at run time.
extern const char *const kDefaultStageXml;

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
    KDL::Rotation r;
    std::ranges::copy(std::span(m, 9), r.data);
    return r;
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
    const auto                  resolve = [&base](mjString *file, const mjString *dir) {
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
static Status resolve_parent(mjSpec *spec, const AttachTarget &t, mjsElement **out)
{
    *out = nullptr;
    switch (t.kind) {
    case AttachKind::World:
        if (mjsBody *wb = world_body(spec)) *out = wb->element;
        break;
    case AttachKind::Body:
        if (t.name.empty()) MJ_FAIL("AttachTarget kind=Body has no name");
        if (mjsBody *b = mjs_findBody(spec, t.name.c_str())) *out = b->element;
        if (!*out) MJ_FAIL("attach parent body '" << t.name << "' not found");
        break;
    case AttachKind::Site:
        if (t.name.empty()) MJ_FAIL("AttachTarget kind=Site has no name");
        *out = mjs_findElement(spec, mjOBJ_SITE, t.name.c_str());
        if (!*out) MJ_FAIL("attach parent site '" << t.name << "' not found");
        break;
    case AttachKind::Frame:
        if (t.name.empty()) MJ_FAIL("AttachTarget kind=Frame has no name");
        if (mjsFrame *f = mjs_findFrame(spec, t.name.c_str())) *out = f->element;
        if (!*out) MJ_FAIL("attach parent frame '" << t.name << "' not found");
        break;
    }
    if (!*out) MJ_FAIL("the scene has no worldbody to attach to");
    return {};
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
// On success *attached is the attached body in the scene spec (so callers can rename or inspect
// it).
static Status attach_child(
  mjSpec             *spec,
  const AttachTarget &target,
  const double        pos[3],
  const double        quat[4],
  mjsBody            *child_root,
  const std::string  &prefix,
  mjsBody           **attached_out = nullptr
)
{
    if (!spec || !child_root) MJ_FAIL("attach: null spec or child body");
    mjsElement *parent = nullptr;
    if (Status s = resolve_parent(spec, target, &parent); !s) return s;

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

    // The child's unnamed top-level default class is written back nameless by mj_saveXML, which
    // mj_loadXML then rejects; give it the child model's name.
    mjSpec     *child     = mjs_getSpec(child_root->element);
    mjsDefault *child_def = child ? mjs_getSpecDefault(child) : nullptr;
    if (child_def && child->modelname) {
        mjs_setName(child_def->element, (prefix + mjs_getString(child->modelname)).c_str());
    }
    mjsElement *attached = mjs_attach(attach_parent, child_root->element, prefix.c_str(), "");
    if (!attached) MJ_FAIL("mjs_attach failed: " << mjs_getError(spec));
    if (attached_out) *attached_out = mjs_asBody(attached);
    return {};
}

// Extract the first root body from a freshly parsed or built mjSpec
// (i.e. the first body child of its worldbody).
static mjsBody *first_root_body(mjSpec *spec)
{
    mjsBody    *wb    = world_body(spec);
    mjsElement *first = wb ? mjs_firstChild(wb, mjOBJ_BODY, 0) : nullptr;
    return first ? mjs_asBody(first) : nullptr;
}

// The spec a scene is built on: the stage, its planes raised by floor_z, or an empty world.
static Status make_stage(const SceneSpec *sc, MjSpecPtr *out)
{
    if (!sc->use_stage) {
        *out = make_spec_ptr(mj_makeSpec());
        if (!*out) MJ_FAIL("mj_makeSpec() failed");
        return {};
    }
    char err[kMjErrBuf] = {};
    if (sc->stage.empty()) {
        *out = make_spec_ptr(mj_parseXMLString(kDefaultStageXml, nullptr, err, sizeof(err)));
        if (!*out) MJ_FAIL("the bundled default stage does not parse: " << err);
    } else {
        *out = make_spec_ptr(mj_parseXML(sc->stage.c_str(), nullptr, err, sizeof(err)));
        if (!*out) MJ_FAIL("SceneSpec::stage '" << sc->stage << "': " << err);
        absolutize_asset_files(out->get());
    }
    mjsBody *wb = world_body(out->get());
    for (mjsElement *e = mjs_firstChild(wb, mjOBJ_GEOM, 0); e; e = mjs_nextChild(wb, e, 0)) {
        mjsGeom *g = mjs_asGeom(e);
        if (g->type == mjGEOM_PLANE) g->pos[2] += sc->floor_z;
    }
    return {};
}

static Status add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects)
{
    for (const auto &obj : objects) {
        if (!obj.mjcf_path.empty()) {
            if (obj.has_rgba
                && std::ranges::any_of(obj.rgba, [](float v) { return std::isnan(v); }))
                MJ_FAIL("SceneObject '" << obj.name << "' sets has_rgba but not .rgba");
            char      err[kMjErrBuf] = {};
            MjSpecPtr asset =
              make_spec_ptr(mj_parseXML(obj.mjcf_path.c_str(), nullptr, err, sizeof(err)));
            if (!asset)
                MJ_FAIL("mj_parseXML failed for object asset '" << obj.mjcf_path << "': " << err);
            absolutize_asset_files(asset.get());

            // Compiled alone first: the scene's compile error would not name the asset.
            if (mjModel *compiled = mj_compile(asset.get(), nullptr)) {
                mj_deleteModel(compiled);
            } else {
                MJ_FAIL(
                  "failed to compile object asset '" << obj.mjcf_path
                                                     << "': " << mjs_getError(asset.get())
                );
            }

            mjsBody *root = first_root_body(asset.get());
            if (!root) MJ_FAIL("no root body found in object asset '" << obj.mjcf_path << "'");

            mjsBody *attached = nullptr;
            if (Status s =
                  attach_child(spec, obj.attach_to, obj.pos, obj.quat, root, obj.prefix, &attached);
                !s)
                return s;
            // attach_to = { Body, obj.name } then works without the asset's root body name.
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
            MJ_FAIL(
              "primitive SceneObject '" << obj.name
                                        << "' has Shape::Unspecified; set .shape explicitly "
                                           "(BOX/SPHERE/CYLINDER/ELLIPSOID/MESH)"
            );
        }
        if (obj.shape == Shape::MESH) {
            if (obj.mesh_path.empty())
                MJ_FAIL("MESH SceneObject '" << obj.name << "' leaves .mesh_path unset");
            // Checked here: MuJoCo's compile error for a missing file would not name the object.
            if (std::error_code ec; !std::filesystem::is_regular_file(obj.mesh_path, ec))
                MJ_FAIL(
                  "MESH SceneObject '" << obj.name << "': .mesh_path '" << obj.mesh_path
                                       << "' is not a file"
                );
        }
        // Validate fields the user must set on a primitive (unset is NaN, which fails every > 0).
        // Free-jointed bodies also need mass > 0; a fixed body may leave it unset.
        const int n_size = obj.shape == Shape::SPHERE ? 1 : obj.shape == Shape::CYLINDER ? 2 : 3;
        if (!std::ranges::all_of(std::span(obj.size).first(n_size), [](double s) {
                return s > 0.0;
            })) {
            MJ_FAIL(
              "primitive SceneObject '"
              << obj.name
              << "' has an unset or non-positive .size for its shape; set its dimensions"
            );
        }
        if (!obj.fixed && !(obj.mass > 0.0)) {
            MJ_FAIL(
              "primitive SceneObject '" << obj.name << "' has .mass=" << obj.mass
                                        << "; non-fixed bodies require mass > 0"
            );
        }
        // Unset leaves MuJoCo's default; a partly set vector is a mistake, not a choice.
        const bool rgba_set = std::ranges::none_of(obj.rgba, [](float v) { return std::isnan(v); });
        const bool friction_set =
          std::ranges::none_of(obj.friction, [](double v) { return std::isnan(v); });
        if (!rgba_set && !std::ranges::all_of(obj.rgba, [](float v) { return std::isnan(v); }))
            MJ_FAIL("primitive SceneObject '" << obj.name << "' sets only part of .rgba");
        if (obj.has_rgba && !rgba_set)
            MJ_FAIL("SceneObject '" << obj.name << "' sets has_rgba but not .rgba");
        if (!friction_set && (!obj.fixed || !std::ranges::all_of(obj.friction, [](double v) {
                return std::isnan(v);
            }))) {
            MJ_FAIL(
              "primitive SceneObject '"
              << obj.name
              << "' leaves .friction unset or partly set; only a fixed primitive may leave it unset"
            );
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
        case Shape::ELLIPSOID:
            g->type = mjGEOM_ELLIPSOID;
            break;
        case Shape::MESH: {
            // The mesh asset travels with the body when attach_child copies it into the scene.
            const std::string mesh_name = obj.name + "_mesh";
            mjsMesh          *mesh      = mjs_addMesh(tmp.get(), nullptr);
            mjs_setString(mjs_getName(mesh->element), mesh_name.c_str());
            mjs_setString(
              mesh->file, std::filesystem::absolute(obj.mesh_path).lexically_normal().c_str()
            );
            for (int k = 0; k < 3; ++k) mesh->scale[k] = obj.size[k];
            g->type = mjGEOM_MESH;
            mjs_setString(g->meshname, mesh_name.c_str());
            break;
        }
        case Shape::Unspecified:
            break; // unreachable, guarded above
        }
        // A mesh is sized by its asset's scale; the geom's own size is unused.
        if (obj.shape != Shape::MESH)
            for (int k = 0; k < 3; ++k) g->size[k] = k < n_size ? obj.size[k] : 0.0;
        if (!std::isnan(obj.mass)) g->mass = obj.mass;
        if (rgba_set) std::ranges::copy(obj.rgba, g->rgba);
        if (friction_set) std::ranges::copy(obj.friction, g->friction);
        g->contype     = obj.contype;
        g->conaffinity = obj.conaffinity;
        g->condim      = static_cast<int>(obj.condim);

        if (Status s = attach_child(spec, obj.attach_to, obj.pos, obj.quat, ob, ""); !s) return s;
    }
    return {};
}

static Status add_cameras_to_spec(mjSpec *spec, const std::vector<CameraSpec> &cameras)
{
    mjsBody *wb = world_body(spec);
    for (const auto &cs : cameras) {
        if (std::ranges::any_of(cs.pos, [](double v) { return std::isnan(v); })
            || std::isnan(cs.fovy))
            MJ_FAIL("camera '" << cs.name << "' leaves .pos or .fovy unset");
        // The asset's own camera wins: it is the one its author placed, and re-adding the
        // name would fail the compile on a duplicate.
        if (mjs_findElement(spec, mjOBJ_CAMERA, cs.name.c_str())) continue;
        mjsBody *anchor = cs.body.empty() ? wb : mjs_findBody(spec, cs.body.c_str());
        if (!anchor) {
            MJ_LOG_WARN("camera '" << cs.name << "': no body '" << cs.body << "' in the scene");
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
    return {};
}

static void add_site_to_spec(mjSpec *spec, mjsBody *body, const SiteSpec &ss)
{
    // The asset's own site wins: it is the one its author placed, and re-adding the
    // name would fail the compile on a duplicate.
    if (mjs_findElement(spec, mjOBJ_SITE, ss.name.c_str())) return;

    mjsSite *site = mjs_addSite(body, nullptr);
    mjs_setString(mjs_getName(site->element), ss.name.c_str());
    site->type    = mjGEOM_SPHERE;
    site->size[0] = ss.size;
    site->size[1] = ss.size;
    site->size[2] = ss.size;
    site->group   = ss.group;
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
            MJ_LOG_WARN("site '" << ss.name << "': no body '" << ss.body << "' in the scene");
            continue;
        }
        add_site_to_spec(spec, body, ss);
    }
}

// Compile spec into a model and data; spec is always deleted.
// A compiled model cannot be written back to XML on its own; mj_saveXML needs its spec.
static std::unordered_map<const mjModel *, MjSpecPtr> g_model_specs;
static std::mutex                                     g_model_specs_mtx; // Envs on many threads

static Status compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data)
{
    MjSpecPtr owned = make_spec_ptr(spec);
    *out_model      = mj_compile(spec, nullptr);
    if (!*out_model) {
        const std::string error = mjs_getError(spec);
        MJ_FAIL(
          "mj_compile failed: " << error
                                << (error.find("repeated name") != std::string::npos
                                      ? " (an asset used twice needs a distinct prefix: "
                                        "SceneObject::prefix, RobotSpec::prefix or "
                                        "AttachmentSpec::prefix)"
                                      : "")
        );
    }
    MJ_LOG_INFO(
      "scene compiled: nq=" << (*out_model)->nq << " nv=" << (*out_model)->nv
                            << " nbody=" << (*out_model)->nbody
    );
    *out_data = mj_makeData(*out_model);
    if (!*out_data) {
        mj_deleteModel(*out_model);
        *out_model = nullptr;
        MJ_FAIL("mj_makeData failed: out of memory for the compiled model");
    }
    const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
    g_model_specs.insert_or_assign(*out_model, std::move(owned));
    return {};
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
static int
  native_mode(int gaintype, const double *gainprm, int biastype, const double *biasprm, int dyntype)
{
    if (dyntype != mjDYN_NONE || gaintype != mjGAIN_FIXED || gainprm[0] == 0.0) return -1;
    if (biastype == mjBIAS_NONE) return static_cast<int>(CtrlMode::TORQUE);
    if (biastype == mjBIAS_AFFINE && biasprm[0] == 0.0 && biasprm[1] == -gainprm[0])
        return static_cast<int>(CtrlMode::POSITION);
    return -1;
}

static int mode_group(int robot, int mode) { return 1 + 3 * robot + mode; }

// joint_names' MuJoCo addresses and the mode m is in; *out (env aside) changes only on success.
static Status resolve_joints(
  const mjModel                  *m,
  const std::vector<std::string> &joint_names,
  const std::string              &pfx,
  RobotInternals                 *out
)
{
    RobotInternals in;
    in.env       = out->env;
    in.mj_prefix = pfx;
    for (auto &ctrl : in.mode_ctrl) ctrl.assign(joint_names.size(), -1);
    for (size_t j = 0; j < joint_names.size(); ++j) {
        const std::string &name = joint_names[j];
        int                id   = mj_name2id(m, mjOBJ_JOINT, (pfx + name).c_str());
        if (id < 0) {
            MJ_FAIL(
              "joint '" << pfx << name
                        << "' not found in MuJoCo model - check robot prefix or URDF joint names"
            );
        }
        in.kdl_to_mj_qpos.push_back(m->jnt_qposadr[id]);
        in.kdl_to_mj_dof.push_back(m->jnt_dofadr[id]);
        // A mode group says which mode an actuator serves; ungrouped ones serve their native mode.
        for (int ai = 0; ai < m->nu; ++ai) {
            if (m->actuator_trntype[ai] != mjTRN_JOINT || m->actuator_trnid[2 * ai] != id) continue;
            const int group = m->actuator_group[ai];
            int       mode  = -1;
            if (group >= 1 && group <= 30) {
                mode            = (group - 1) % 3;
                const int owner = (group - 1) / 3;
                if (std::ranges::find(in.mode_owners, owner) == in.mode_owners.end())
                    in.mode_owners.push_back(owner);
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
    }

    // The mode the model is in now: every joint has its actuator and its group is enabled.
    for (int mode = 0; mode < 3 && !joint_names.empty(); ++mode) {
        const bool driven  = std::ranges::all_of(in.mode_ctrl[mode], [](int a) { return a >= 0; });
        const bool enabled = std::ranges::none_of(in.mode_owners, [&](int owner) {
            return m->opt.disableactuator & (1 << mode_group(owner, mode));
        });
        if (driven && enabled) {
            in.applied_mode = static_cast<CtrlMode>(mode);
            in.mode_applied = true;
            break;
        }
    }
    *out = std::move(in);
    return {};
}

// A joint the model leaves unlimited (a continuous wrist) is unlimited here too, not +-pi.
static std::pair<double, double> joint_range(const mjModel *model, int jid)
{
    if (jid >= 0 && model->jnt_limited[jid])
        return { model->jnt_range[2 * jid], model->jnt_range[2 * jid + 1] };
    MJ_LOG_INFO(
      "joint '" << (jid >= 0 ? mj_id2name(model, mjOBJ_JOINT, jid) : "?") << "' is unlimited"
    );
    const double inf = std::numeric_limits<double>::infinity();
    return { -inf, inf };
}

// Build KDL chain from compiled mjModel (no URDF needed)

static Status
  build_kdl_from_model(Robot *s, mjModel *model, const char *base_body, const char *tip_body)
{
    int base_bid = mj_name2id(model, mjOBJ_BODY, base_body);
    int tip_bid  = mj_name2id(model, mjOBJ_BODY, tip_body);
    if (base_bid < 0) MJ_FAIL("base body '" << base_body << "' not found in compiled model");
    if (tip_bid < 0) MJ_FAIL("tip body '" << tip_body << "' not found in compiled model");

    std::vector<int> bids;
    for (int b = tip_bid; b != base_bid; b = model->body_parentid[b]) {
        if (b == 0) {
            MJ_FAIL(
              "'" << tip_body << "' is not a descendant of '" << base_body
                  << "'  - check body hierarchy in the model"
            );
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

        // A KDL segment carries at most one scalar joint; anything else would be dropped silently.
        const char *body = bname ? bname : "(unnamed)";
        if (model->body_jntnum[bid] > 1) {
            MJ_FAIL(
              "body '" << body << "' has " << model->body_jntnum[bid]
                       << " joints; a chain body carries at most one"
            );
        }
        KDL::Joint jnt(KDL::Joint::None);
        if (model->body_jntnum[bid] == 1) {
            const int jid = model->body_jntadr[bid];
            if (model->jnt_type[jid] != mjJNT_HINGE && model->jnt_type[jid] != mjJNT_SLIDE) {
                MJ_FAIL(
                  "body '" << body << "' has a ball or free joint; a chain takes hinges and slides"
                );
            }
            const char *jname = mj_id2name(model, mjOBJ_JOINT, jid);
            if (!jname)
                MJ_FAIL("the joint of body '" << body << "' is unnamed, so it cannot be mapped");
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
            jnt                   = KDL::Joint(jname, origin, axis, jtype, 1.0, 0.0, armature);
            s->joint_names.push_back(jname);
            s->joint_limits.push_back(joint_range(model, jid));
        }

        KDL::RigidBodyInertia inertia = mj_body_inertia(model, bid);

        s->chain.addSegment(KDL::Segment(bname ? bname : "", jnt, F, inertia));
    }
    s->n_joints = (int)s->chain.getNrOfJoints();
    return {};
}

// Scene API

Status save_model_xml(const mjModel *model, const char *path)
{
    char                              err[kMjErrBuf] = {};
    int                               ok             = 0;
    const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
    const auto                        spec = g_model_specs.find(model);
    if (spec != g_model_specs.end()) {
        ok = mj_copyBack(spec->second.get(), model)
             && mj_saveXML(spec->second.get(), path, err, sizeof(err)) == 0;
    } else {
        ok = mj_saveLastXML(path, model, err, sizeof(err));
    }
    if (!ok) MJ_FAIL("saving the model to '" << path << "' failed: " << err);
    MJ_LOG_INFO("model saved to '" << path << "'");
    return {};
}

// The body the robot's first joint moves against, -1 if unknown; the viewer's label for an
// unnamed robot. Declared there, not in a header: no part of the public API.
int robot_base_body(const Robot *r)
{
    if (!r || !r->model || r->_impl->kdl_to_mj_qpos.empty()) return -1;
    const mjModel *m   = r->model;
    const int      adr = r->_impl->kdl_to_mj_qpos.front();
    for (int j = 0; j < m->njnt; ++j)
        if (m->jnt_qposadr[j] == adr) return m->body_parentid[m->jnt_bodyid[j]];
    return -1;
}

// The viewer's Explorer reads it; declared there, not in a header: no part of the public API.
mjSpec *copy_model_spec(const mjModel *model)
{
    const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
    const auto                        spec = g_model_specs.find(model);
    return spec == g_model_specs.end() ? nullptr : mj_copySpec(spec->second.get());
}

static void forget_spec(const mjModel *model)
{
    const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
    g_model_specs.erase(model);
}

void destroy_scene(mjModel *model, mjData *data)
{
    if (model) forget_spec(model);
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
}

// Hand a freshly compiled pair to env->adopt, if set; the spec save_model_xml needs follows it.
static Status adopt_pair(Env *env, mjModel **m, mjData **d, bool *adopted)
{
    *adopted = static_cast<bool>(env->adopt);
    if (!*adopted) return {};
    decltype(g_model_specs)::node_type spec;
    {
        const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
        spec = g_model_specs.extract(*m);
    }
    auto [am, ad] = env->adopt(*m, *d);
    if (!am || !ad) MJ_FAIL("Env::adopt returned no model or data");
    if (spec) {
        spec.key() = am;
        const std::lock_guard<std::mutex> lock(g_model_specs_mtx);
        g_model_specs.insert(std::move(spec));
    }
    *m = am;
    *d = ad;
    return {};
}

// Free a pair, unless adopt gave it: then its owner frees it.
static void release_pair(mjModel *m, mjData *d, bool adopted)
{
    if (adopted)
        forget_spec(m);
    else
        destroy_scene(m, d);
}

/* A hook may add or erase entries while it runs (a viewer starting or stopping its recorder), so
 * each one is looked up again, and called on a copy the map cannot free under it. */
template<class Fn, class Call> static void for_each_hook(Env *env, Fn EnvHooks::*hook, Call call)
{
    std::vector<const void *> owners;
    owners.reserve(env->hooks.size());
    for (const auto &entry : env->hooks) owners.push_back(entry.first);
    for (const void *owner : owners) {
        const auto it = env->hooks.find(owner);
        if (it == env->hooks.end() || !(it->second.*hook)) continue;
        const Fn fn = it->second.*hook;
        call(fn);
    }
}

// Every cache is tied to the model it was taken from.
static void forget_model(Env *env)
{
    env->_impl->computed  = false;
    env->_impl->model_gen = ++g_model_gen;
    for (auto &names : env->_impl->names) names.clear();
}

Status init_env(Env *env, const SceneSpec *spec)
{
    if (!env || !spec) MJ_FAIL("init_env: null env or spec");

    ResetHook on_reset = std::move(env->on_reset);
    cleanup(env);
    env->on_reset = std::move(on_reset);
    env->spec     = *spec;
    mjModel *m    = nullptr;
    mjData  *d    = nullptr;
    if (Status built = build_scene(&m, &d, &env->spec); !built) return built;
    bool adopted = false;
    if (Status s = adopt_pair(env, &m, &d, &adopted); !s) return s;
    env->model          = m;
    env->data           = d;
    env->_impl->adopted = adopted;
    env->scene          = SceneState{};
    env->scene.model    = env->model;
    forget_model(env);
    return {};
}

void cleanup(Env *env)
{
    if (!env) return;
    for_each_hook(env, &EnvHooks::on_close, [env](const auto &fn) { fn(env); });
    env->hooks.clear();
    for (Robot *r : env->robots) {
        r->_impl->env = nullptr;
        r->model      = nullptr;
        r->data       = nullptr;
    }
    env->robots.clear();
    release_pair(env->model, env->data, env->_impl->adopted);
    env->_impl->adopted = false;
    env->model          = nullptr;
    env->data           = nullptr;
    env->scene          = SceneState{};
    forget_model(env);
    env->on_reset = nullptr;
}

static Status apply_ctrl_modes(
  mjSpec                          *arm,
  const std::vector<CtrlModeSpec> &modes,
  int                              owner,
  const std::string               &who,
  int                             *disable_bits
);

// owner is the attachment's mode-group owner, used only when it has modes.
static Status attach_to_spec(
  mjSpec               *robot_spec,
  const AttachmentSpec *a,
  int                   owner,
  const std::string    &who,
  int                  *disable_bits
)
{
    if (!robot_spec || !a || a->mjcf_path.empty())
        MJ_FAIL("attach_to_spec: null spec or empty mjcf_path");
    // Only the robot's first root body reaches the scene, so a second root would vanish.
    if (a->attach_to.kind == AttachKind::World)
        MJ_FAIL(who << ": attach_to is the world; name a body, site or frame of the robot");
    ensure_plugins_loaded();
    MJ_LOG_INFO(
      "attach_to_spec: parent='" << (a->attach_to.name.empty() ? "(world)" : a->attach_to.name)
                                 << "' prefix='" << a->prefix << "'"
    );

    char      err[kMjErrBuf] = {};
    MjSpecPtr att = make_spec_ptr(mj_parseXML(a->mjcf_path.c_str(), nullptr, err, sizeof(err)));
    if (!att) MJ_FAIL("mj_parseXML failed for attachment '" << a->mjcf_path << "': " << err);
    absolutize_asset_files(att.get());
    // On the attachment's own spec, before the prefix: only its own joints take its modes.
    if (Status s = apply_ctrl_modes(att.get(), a->modes, owner, who, disable_bits); !s) return s;

    mjsBody *att_root = first_root_body(att.get());
    if (!att_root) MJ_FAIL("no root body found in attachment spec '" << a->mjcf_path << "'");

    if (Status s = attach_child(robot_spec, a->attach_to, a->pos, a->quat, att_root, a->prefix); !s)
        return s;
    // att (deep-copied into robot_spec) is freed by MjSpecPtr at scope exit.

    // Register contact exclusions.
    for (const auto &ex : a->contact_exclusions) {
        mjsExclude *exc = mjs_addExclude(robot_spec);
        mjs_setString(exc->bodyname1, ex.first.c_str());
        mjs_setString(exc->bodyname2, ex.second.c_str());
    }
    return {};
}

/* Give each listed joint one actuator per requested mode, each mode in its own group of this
 * owner; the joint's own actuator joins the group of the mode it gives natively. Collects the
 * groups to start disabled. */
static Status apply_ctrl_modes(
  mjSpec                          *arm,
  const std::vector<CtrlModeSpec> &modes,
  int                              owner,
  const std::string               &who,
  int                             *disable_bits
)
{
    if (modes.empty()) return {};
    if (mode_group(owner, static_cast<int>(CtrlMode::VELOCITY)) > 30)
        MJ_FAIL(who << ": control-mode groups run out after ten robots and moded attachments");

    std::map<std::string, std::vector<mjsActuator *>> by_joint;
    for (mjsElement *e = mjs_firstElement(arm, mjOBJ_ACTUATOR); e; e = mjs_nextElement(arm, e)) {
        mjsActuator *a = mjs_asActuator(e);
        if (a->trntype == mjTRN_JOINT) by_joint[mjs_getString(a->target)].push_back(a);
    }

    int enabled = 0, used = 0;
    for (const CtrlModeSpec &ms : modes) {
        // Every actuated joint when none are named: those that cannot take modes are skipped.
        const bool               all    = ms.joints.empty();
        std::vector<std::string> joints = ms.joints;
        if (all)
            for (const auto &entry : by_joint) joints.push_back(entry.first);

        for (const std::string &joint : joints) {
            const auto   it = by_joint.find(joint);
            mjsActuator *own =
              (it != by_joint.end() && it->second.size() == 1) ? it->second.front() : nullptr;
            const int native =
              own ? native_mode(
                      own->gaintype, own->gainprm, own->biastype, own->biasprm, own->dyntype
                    )
                  : -1;
            if (native < 0) {
                if (all) {
                    MJ_LOG_INFO(who << ": joint '" << joint << "' takes no control modes");
                    continue;
                }
                MJ_FAIL(
                  who << ": joint '" << joint
                      << "' needs exactly one position-servo or motor actuator"
                );
            }
            own->group = mode_group(owner, native);
            enabled |= 1 << own->group;
            used |= 1 << own->group;

            const int mode = static_cast<int>(ms.mode);
            if (mode == native) continue;
            if (ms.mode == CtrlMode::POSITION) {
                MJ_FAIL(
                  who << ": joint '" << joint << "' is motor-driven; POSITION is not supported"
                );
            }
            if (ms.mode == CtrlMode::VELOCITY && ms.kv <= 0.0)
                MJ_FAIL(who << ": VELOCITY needs kv > 0");

            // Both limits below are in actuator-force units: a servo's forcerange, a motor's ctrl.
            const bool    servo = native == static_cast<int>(CtrlMode::POSITION);
            const double *range = servo ? own->forcerange : own->ctrlrange;
            const int     flag  = servo ? own->forcelimited : own->ctrllimited;
            const bool    limited =
              flag == mjLIMITED_TRUE || (flag == mjLIMITED_AUTO && range[0] < range[1]);

            mjsActuator *added = mjs_addActuator(arm, nullptr);
            added->trntype     = mjTRN_JOINT;
            mjs_setString(added->target, joint.c_str());
            added->gear[0] = own->gear[0];
            added->group   = mode_group(owner, mode);
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
    return {};
}

Status build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *sc)
{
    if (!sc) MJ_FAIL("build_scene: null spec");
    if (!(sc->timestep > 0.0)) {
        MJ_FAIL(
          "SceneSpec::timestep must be > 0 (got "
          << sc->timestep << "); the field has no default, set it explicitly (suggested 0.001 s)"
        );
    }
    ensure_plugins_loaded();
    MJ_LOG_INFO(
      "build_scene: " << sc->robots.size() << " robot(s)" << ", objects=" << sc->objects.size()
    );

    MjSpecPtr scene = make_spec_ptr(nullptr);
    if (Status s = make_stage(sc, &scene); !s) return s;

    scene->compiler.balanceinertia = true; // mjsCompiler stores as int 0/1
    scene->compiler.discardvisual  = false;

    // Objects come before robots so a robot can attach to a SceneObject (e.g.
    // {AttachKind::Site, "table_mount"}). A child object that references
    // another object must appear after its parent in SceneSpec::objects.
    if (Status s = add_objects_to_spec(scene.get(), sc->objects); !s) return s;

    // Sites land as their bodies arrive, so a later attach can name one -- see add_ready_sites.
    std::vector<SiteSpec> pending_sites = sc->sites;
    add_ready_sites(scene.get(), pending_sites);

    bool first_arm      = true;
    int  disable_bits   = 0;
    char err[kMjErrBuf] = {};
    // Robots own mode groups by their index; attachments with modes take the owners after them.
    int next_owner = static_cast<int>(sc->robots.size());
    for (int ai = 0; ai < (int)sc->robots.size(); ++ai) {
        const RobotSpec &rs = sc->robots[ai];
        if (rs.path.empty()) MJ_FAIL("robots[" << ai << "].path is empty");

        MjSpecPtr arm = make_spec_ptr(mj_parseXML(rs.path.c_str(), nullptr, err, sizeof(err)));
        if (!arm) MJ_FAIL("mj_parseXML failed for '" << rs.path << "': " << err);
        absolutize_asset_files(arm.get());
        const std::string who = "robots[" + std::to_string(ai) + "]";
        // Before the attachments are merged in, so only the robot's own joints take modes.
        if (Status s = apply_ctrl_modes(arm.get(), rs.modes, ai, who, &disable_bits); !s) return s;

        // Inherit physics options (integrator, solver, etc.) from the first
        // arm, then apply the SceneSpec's user-controlled fields on top.
        if (first_arm) {
            scene->option            = arm->option;
            scene->option.timestep   = sc->timestep;
            scene->option.gravity[2] = sc->gravity_z;
            first_arm                = false;
        }

        // Apply attachment chain in order (mount, sensor, gripper, etc.).
        for (size_t k = 0; k < rs.attachments.size(); ++k) {
            const AttachmentSpec &att     = rs.attachments[k];
            const int             owner   = att.modes.empty() ? -1 : next_owner++;
            const std::string     att_who = who + ".attachments[" + std::to_string(k) + "]";
            if (Status s = attach_to_spec(arm.get(), &att, owner, att_who, &disable_bits); !s)
                return s;
        }

        mjsBody *arm_root = first_root_body(arm.get());
        if (!arm_root) MJ_FAIL("no root body found in arm spec '" << rs.path << "'");

        if (Status s =
              attach_child(scene.get(), rs.attach_to, rs.pos, rs.quat, arm_root, rs.prefix);
            !s)
            return s;
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
    // Set only: a scene MJCF that enables it keeps it (doc/computation/speculative.rst).
    if (sc->speculative_contacts) scene->option.enableflags |= mjENBL_SPECULATIVE;
    switch (sc->integrator) {
    case Integrator::Robot:
        break;
    case Integrator::Euler:
        scene->option.integrator = mjINT_EULER;
        break;
    case Integrator::RK4:
        scene->option.integrator = mjINT_RK4;
        break;
    case Integrator::Implicit:
        scene->option.integrator = mjINT_IMPLICIT;
        break;
    case Integrator::ImplicitFast:
        scene->option.integrator = mjINT_IMPLICITFAST;
        break;
    case Integrator::Discrete:
        scene->option.integrator = mjINT_DISCRETE;
        break;
    }
    switch (sc->solver) {
    case Solver::Robot:
        break;
    case Solver::PGS:
        scene->option.solver = mjSOL_PGS;
        break;
    case Solver::CG:
        scene->option.solver = mjSOL_CG;
        break;
    case Solver::Newton:
        scene->option.solver = mjSOL_NEWTON;
        break;
    }

    if (Status s = add_cameras_to_spec(scene.get(), sc->cameras); !s) return s;
    // Whatever is still pending: a site whose body no robot or object ever brought in, reported
    // here rather than dropped in silence.
    if (!pending_sites.empty()) add_sites_to_spec(scene.get(), pending_sites);
    // compile_and_make_data takes ownership of the raw spec and always deletes it.
    return compile_and_make_data(scene.release(), out_model, out_data);
}

static bool rebind_scene(Env *env);

static Status resolve_ft(const mjModel *m, ForceTorqueSensor *sensor);
static Status switch_control_mode(Robot *r, CtrlMode mode, bool seed_ports);

// What a registered robot becomes on a rebuilt model.
struct RobotRebind
{
    RobotInternals                 in;
    std::vector<ForceTorqueSensor> ft_sensors;
};

// Every MuJoCo address a Robot holds is only valid for the model it was resolved on.
static Status resolve_robots(const Env *env, const mjModel *m, std::vector<RobotRebind> *out)
{
    out->clear();
    for (const Robot *r : env->robots) {
        const RobotInternals &was = *r->_impl;
        RobotRebind           rb;
        rb.in.env = was.env;
        if (Status s = resolve_joints(m, r->joint_names, was.mj_prefix, &rb.in); !s) return s;
        const auto &ctrl = rb.in.mode_ctrl[static_cast<int>(was.applied_mode)];
        if (was.mode_applied && std::ranges::any_of(ctrl, [](int a) { return a < 0; }))
            MJ_FAIL("a robot's control mode lost its actuators in the rebuilt model");
        rb.ft_sensors = r->ft_sensors;
        for (ForceTorqueSensor &sensor : rb.ft_sensors)
            if (Status s = resolve_ft(m, &sensor); !s) return s;
        out->push_back(std::move(rb));
    }
    return {};
}

static void rebind_robots(Env *env, std::vector<RobotRebind> *rebinds)
{
    for (size_t k = 0; k < env->robots.size(); ++k) {
        Robot           *r           = env->robots[k];
        const RobotPorts ports       = *r;
        const bool       was_applied = r->_impl->mode_applied;
        const CtrlMode   was         = r->_impl->applied_mode;
        r->model                     = env->model;
        r->data                      = env->data;
        *r->_impl                    = std::move((*rebinds)[k].in);
        r->ft_sensors                = std::move((*rebinds)[k].ft_sensors);
        // The rebuilt model starts in the native mode; carry on in the one the robot was in.
        if (was_applied && was != r->_impl->applied_mode) (void)switch_control_mode(r, was, false);
        static_cast<RobotPorts &>(*r) = ports;
    }
}

// Time, and qpos/qvel/act of every joint and actuator the old model names too.
static void carry_state(const mjModel *om, const mjData *od, const mjModel *nm, mjData *nd)
{
    nd->time = od->time;
    for (int j = 0; j < nm->njnt; ++j) {
        const char *name = mj_id2name(nm, mjOBJ_JOINT, j);
        const int   oj   = name ? mj_name2id(om, mjOBJ_JOINT, name) : -1;
        if (oj < 0 || om->jnt_type[oj] != nm->jnt_type[j]) continue;
        const int type = nm->jnt_type[j];
        const int nq   = type == mjJNT_FREE ? 7 : type == mjJNT_BALL ? 4 : 1;
        const int nv   = type == mjJNT_FREE ? 6 : type == mjJNT_BALL ? 3 : 1;
        mju_copy(nd->qpos + nm->jnt_qposadr[j], od->qpos + om->jnt_qposadr[oj], nq);
        mju_copy(nd->qvel + nm->jnt_dofadr[j], od->qvel + om->jnt_dofadr[oj], nv);
    }
    for (int a = 0; a < nm->nu; ++a) {
        const char *name = mj_id2name(nm, mjOBJ_ACTUATOR, a);
        const int   oa   = name ? mj_name2id(om, mjOBJ_ACTUATOR, name) : -1;
        const int   n    = nm->actuator_actnum[a];
        if (oa < 0 || n == 0 || om->actuator_actnum[oa] != n) continue;
        mju_copy(nd->act + nm->actuator_actadr[a], od->act + om->actuator_actadr[oa], n);
    }
}

// Swap in a rebuilt pair, or leave env as it was when a robot no longer resolves.
static Status swap_scene(Env *env, mjModel *m, mjData *d)
{
    // Before adopt, whose copy keeps the structure, so a failure has nothing to undo.
    std::vector<RobotRebind> rebinds;
    if (Status s = resolve_robots(env, m, &rebinds); !s) {
        destroy_scene(m, d);
        return s;
    }
    bool adopted = false;
    if (Status s = adopt_pair(env, &m, &d, &adopted); !s) return s;
    carry_state(env->model, env->data, m, d);
    mj_forward(m, d);
    // The old pair lives until its ctrl is carried over.
    mjModel   *old_m       = env->model;
    mjData    *old_d       = env->data;
    const bool old_adopted = env->_impl->adopted;
    env->model             = m;
    env->data              = d;
    env->_impl->adopted    = adopted;
    forget_model(env);
    rebind_robots(env, &rebinds);
    // After the mode switches, which seed ctrl from the joints: the commands as they were.
    for (int a = 0; a < m->nu; ++a) {
        const char *name = mj_id2name(m, mjOBJ_ACTUATOR, a);
        const int   oa   = name ? mj_name2id(old_m, mjOBJ_ACTUATOR, name) : -1;
        if (oa >= 0) d->ctrl[a] = old_d->ctrl[oa];
    }
    rebind_scene(env);
    release_pair(old_m, old_d, old_adopted);
    for_each_hook(env, &EnvHooks::on_rebuild, [env](const auto &fn) { fn(env); });
    return {};
}

Status scene_add_object(Env *env, const SceneObject &obj)
{
    if (!env) MJ_FAIL("scene_add_object: null env");
    if (env->_impl->in_reset_hook) MJ_FAIL("scene_add_object: not allowed inside on_reset");
    env->spec.objects.push_back(obj);
    mjModel *m = nullptr;
    mjData  *d = nullptr;
    Status   s = build_scene(&m, &d, &env->spec);
    if (s) s = swap_scene(env, m, d);
    if (!s) env->spec.objects.pop_back();
    return s;
}

// The viewer's Reload: the MJCF files read again, the state carried over. Declared there.
Status reload_scene(Env *env)
{
    if (!env) MJ_FAIL("reload_scene: null env");
    if (env->_impl->in_reset_hook) MJ_FAIL("reload_scene: not allowed inside on_reset");
    mjModel *m = nullptr;
    mjData  *d = nullptr;
    if (Status s = build_scene(&m, &d, &env->spec); !s) return s;
    return swap_scene(env, m, d);
}

Status scene_remove_object(Env *env, const std::string &name)
{
    if (!env) MJ_FAIL("scene_remove_object: null env");
    if (env->_impl->in_reset_hook) MJ_FAIL("scene_remove_object: not allowed inside on_reset");
    auto &objects = env->spec.objects;
    auto  it      = std::ranges::find(objects, std::string_view(name), &SceneObject::name);
    if (it == objects.end()) MJ_FAIL("scene_remove_object: no object named '" << name << "'");
    const auto  index   = it - objects.begin();
    SceneObject removed = std::move(*it);
    objects.erase(it);
    mjModel *m = nullptr;
    mjData  *d = nullptr;
    Status   s = build_scene(&m, &d, &env->spec);
    if (s) s = swap_scene(env, m, d);
    // Back where it was: a later object may attach to it.
    if (!s) objects.insert(objects.begin() + index, std::move(removed));
    return s;
}

bool get_site_frame(Env *env, const char *site_name, KDL::Frame *out)
{
    if (!env || !env->model || !site_name || !out) return false;

    const int sid = cached_name2id(env, mjOBJ_SITE, site_name);
    if (sid < 0) return false;

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

    ensure_kinematics(env);
    const double *p = env->data->xpos + 3 * bid;
    const double *R = env->data->xmat + 9 * bid;
    *out            = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
    return true;
}

// Robot API

// The sensor's addresses in m, from its MuJoCo names.
static Status resolve_ft(const mjModel *m, ForceTorqueSensor *sensor)
{
    const std::string &name     = sensor->name;
    const int          force_id = mj_name2id(m, mjOBJ_SENSOR, sensor->force_sensor.c_str());
    if (force_id < 0) {
        MJ_FAIL(
          "force sensor '" << sensor->force_sensor << "' not found for FT sensor '" << name << "'"
        );
    }
    const int torque_id = mj_name2id(m, mjOBJ_SENSOR, sensor->torque_sensor.c_str());
    if (torque_id < 0) {
        MJ_FAIL(
          "torque sensor '" << sensor->torque_sensor << "' not found for FT sensor '" << name << "'"
        );
    }
    if (m->sensor_type[force_id] != mjSENS_FORCE || m->sensor_dim[force_id] != 3)
        MJ_FAIL("sensor '" << sensor->force_sensor << "' must be a 3D MuJoCo force sensor");
    if (m->sensor_type[torque_id] != mjSENS_TORQUE || m->sensor_dim[torque_id] != 3)
        MJ_FAIL("sensor '" << sensor->torque_sensor << "' must be a 3D MuJoCo torque sensor");
    int site_id = -1;
    if (!sensor->frame_site.empty()) {
        site_id = mj_name2id(m, mjOBJ_SITE, sensor->frame_site.c_str());
        if (site_id < 0) {
            MJ_FAIL(
              "frame_site '" << sensor->frame_site << "' not found for FT sensor '" << name << "'"
            );
        }
    }
    sensor->force_adr     = m->sensor_adr[force_id];
    sensor->torque_adr    = m->sensor_adr[torque_id];
    sensor->frame_site_id = site_id;
    return {};
}

static Status resolve_ft_sensors(Robot *r, const ToolFrameSpec *tool, const std::string &pfx)
{
    r->ft_sensors.clear();
    if (!tool) return {};

    for (const ForceTorqueSensorSpec &spec : tool->ft_sensors) {
        if (spec.name.empty()) MJ_FAIL("ForceTorqueSensorSpec.name is required");
        ForceTorqueSensor sensor;
        sensor.name = spec.name;
        sensor.force_sensor =
          pfx + (spec.force_sensor.empty() ? spec.name + "_force" : spec.force_sensor);
        sensor.torque_sensor =
          pfx + (spec.torque_sensor.empty() ? spec.name + "_torque" : spec.torque_sensor);
        if (!spec.frame_site.empty()) sensor.frame_site = pfx + spec.frame_site;
        if (Status s = resolve_ft(r->model, &sensor); !s) return s;
        r->ft_sensors.push_back(std::move(sensor));
    }
    return {};
}

// The only place the ports are allocated: callers keep pointers into them across reset().
static void size_ports(RobotPorts &p, int n)
{
    p.jnt_pos_msr.assign(n, 0.0);
    p.jnt_vel_msr.assign(n, 0.0);
    p.jnt_trq_msr.assign(n, 0.0);
    p.jnt_pos_cmd.assign(n, 0.0);
    p.jnt_vel_cmd.assign(n, 0.0);
    p.jnt_trq_cmd.assign(n, 0.0);
    p.jnt_saturated.assign(n, 0);
}

// Ports that hold the robot where it is, written into their storage; ctrl_mode is left as set,
// so a requested switch still happens at the next update().
static void seed_ports(Robot &r)
{
    const RobotInternals &in = *r._impl;
    const mjData         *d  = r.data;
    for (int i = 0; i < r.n_joints; ++i) {
        const int dof      = in.kdl_to_mj_dof[i];
        r.jnt_pos_msr[i]   = d->qpos[in.kdl_to_mj_qpos[i]];
        r.jnt_vel_msr[i]   = d->qvel[dof];
        r.jnt_trq_msr[i]   = d->qfrc_actuator[dof];
        r.jnt_pos_cmd[i]   = r.jnt_pos_msr[i];
        r.jnt_vel_cmd[i]   = 0.0;
        r.jnt_trq_cmd[i]   = 0.0;
        r.jnt_saturated[i] = 0;
    }
}

static void unregister_robot(Robot *r)
{
    if (Env *env = r->_impl->env) {
        auto &robots = env->robots;
        robots.erase(std::remove(robots.begin(), robots.end(), r), robots.end());
    }
}

// r takes the configuration built into `built` and registers with env; r is untouched before.
static void take_robot(Robot *r, Robot *built, Env *env)
{
    if (r->_impl->env != env) unregister_robot(r);
    r->model          = built->model;
    r->data           = built->data;
    r->chain          = built->chain;
    r->tip_T_tcp      = built->tip_T_tcp;
    r->has_tcp_frame  = built->has_tcp_frame;
    r->tcp_site       = std::move(built->tcp_site);
    r->n_joints       = built->n_joints;
    r->joint_names    = std::move(built->joint_names);
    r->joint_limits   = std::move(built->joint_limits);
    r->ft_sensors     = std::move(built->ft_sensors);
    r->ctrl_mode      = built->ctrl_mode;
    built->_impl->env = nullptr;
    std::swap(r->_impl, built->_impl);
    if (std::ranges::find(env->robots, r) == env->robots.end()) env->robots.push_back(r);
    r->_impl->env = env;
    size_ports(*r, r->n_joints);
    seed_ports(*r);
}

static Status resolve_robot_joints(Robot *r, const std::string &pfx)
{
    if (Status s = resolve_joints(r->model, r->joint_names, pfx, r->_impl.get()); !s) return s;
    if (r->_impl->mode_applied) r->ctrl_mode = r->_impl->applied_mode;
    return {};
}

static Status configure_from_mjcf(
  Robot               *r,
  Env                 *env,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix,
  const ToolFrameSpec *tool
)
{
    if (!base_body || !tip_body) MJ_FAIL("init_robot_from_mjcf: null base or tip body");
    const std::string pfx           = prefix ? prefix : "";
    const std::string base          = pfx + base_body;
    const std::string tip           = pfx + tip_body;
    const bool        has_tool_body = tool && !tool->tool_body.empty();
    const bool        has_tcp_site  = tool && !tool->tcp_site.empty();
    const std::string tool_body     = has_tool_body ? pfx + tool->tool_body : "";
    const std::string tcp_site      = has_tcp_site ? pfx + tool->tcp_site : "";
    MJ_LOG_INFO(
      "init_robot_from_mjcf: '" << base << "' -> '" << tip << "'"
                                << (has_tool_body ? " tool='" + tool_body + "'" : "")
                                << (has_tcp_site ? " tcp='" + tcp_site + "'" : "")
    );
    mjModel *model = env->model;
    mjData  *data  = env->data;
    r->model       = model;
    r->data        = data;
    if (Status s = build_kdl_from_model(r, model, base.c_str(), tip.c_str()); !s) return s;
    // The chain's joint names come from the model, so they already carry the prefix.
    if (Status s = resolve_robot_joints(r, ""); !s) return s;
    if (Status s = resolve_ft_sensors(r, tool, pfx); !s) return s;

    KDL::Frame tip_T_tcp = KDL::Frame::Identity();
    bool       has_tcp   = false;
    if (has_tcp_site) {
        if (!get_site_frame_in_body(env, tip.c_str(), tcp_site.c_str(), &tip_T_tcp)) {
            MJ_FAIL("tcp_site '" << tcp_site << "' or tip body '" << tip << "' not found in model");
        }
        has_tcp          = true;
        r->tip_T_tcp     = tip_T_tcp;
        r->has_tcp_frame = true;
        r->tcp_site      = tcp_site;
    } else if (tool && !Equal(tool->tcp_frame, KDL::Frame::Identity(), tool->tcp_identity_tol)) {
        tip_T_tcp        = tool->tcp_frame;
        has_tcp          = true;
        r->tip_T_tcp     = tip_T_tcp;
        r->has_tcp_frame = true;
    }

    if (has_tool_body) {
        int tool_bid = mj_name2id(model, mjOBJ_BODY, tool_body.c_str());
        if (tool_bid < 0) MJ_FAIL("tool_body '" << tool_body << "' not found in model");
        int tip_bid = mj_name2id(model, mjOBJ_BODY, tip.c_str());
        ensure_kinematics(env);
        std::vector<int>      subtree      = collect_subtree(model, tool_bid);
        KDL::RigidBodyInertia tool_inertia = compute_tool_inertia(model, data, tip_bid, subtree);
        MJ_LOG_INFO(
          "appending lumped tool inertia: " << subtree.size() << " bodies under '" << tool_body
                                            << "'"
        );
        r->chain.addSegment(KDL::Segment(
          tool_body, KDL::Joint(KDL::Joint::None), KDL::Frame::Identity(), tool_inertia
        ));
        // Fixed joints do not count: n_joints remains the same after addSegment.
    }

    if (has_tcp) {
        const std::string seg_name = has_tcp_site ? tcp_site : "tcp";
        r->chain.addSegment(KDL::Segment(seg_name, KDL::Joint(KDL::Joint::None), tip_T_tcp));
    }

    MJ_LOG_INFO(
      "chain ready: " << r->n_joints << " joints [" << base << " -> " << tip << "]"
                      << (has_tool_body ? " + tool '" + tool_body + "'" : "")
                      << (has_tcp ? (has_tcp_site ? " tcp site '" + tcp_site + "'"
                                                  : std::string(" tcp frame (manual)"))
                                  : "")
    );
    return {};
}

static Status configure_from_chain(
  Robot                          *r,
  Env                            *env,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix,
  const ToolFrameSpec            *tool
)
{
    mjModel *model = env->model;
    MJ_LOG_INFO(
      "init_robot_from_chain: " << chain.getNrOfSegments() << " segments, " << chain.getNrOfJoints()
                                << " joints, prefix='" << (prefix ? prefix : "") << "'"
    );
    if (joint_names.size() != chain.getNrOfJoints()) {
        MJ_FAIL(
          "joint_names has " << joint_names.size() << " entries but the chain has "
                             << chain.getNrOfJoints() << " joints"
        );
    }

    r->model = model;
    r->data  = env->data;

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

    if (Status s = resolve_robot_joints(r, pfx); !s) return s;
    if (Status s = resolve_ft_sensors(r, tool, pfx); !s) return s;

    MJ_LOG_INFO(
      "chain adopted: " << r->n_joints << " joints, " << r->chain.getNrOfSegments() << " segments"
    );
    return {};
}

Status init_robot_from_mjcf(
  Robot               *r,
  Env                 *env,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix,
  const ToolFrameSpec *tool
)
{
    if (!r || !env || !env->model) MJ_FAIL("init_robot_from_mjcf: null robot or empty env");
    Robot built;
    if (Status s = configure_from_mjcf(&built, env, base_body, tip_body, prefix, tool); !s)
        return s;
    take_robot(r, &built, env);
    return {};
}

Status init_robot_from_chain(
  Robot                          *r,
  Env                            *env,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix,
  const ToolFrameSpec            *tool
)
{
    if (!r || !env || !env->model) MJ_FAIL("init_robot_from_chain: null robot or empty env");
    Robot built;
    if (Status s = configure_from_chain(&built, env, chain, joint_names, prefix, tool); !s)
        return s;
    take_robot(r, &built, env);
    return {};
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
    unregister_robot(r);
    r->name.clear();
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
    const int n = std::min((int)q.rows(), r->n_joints);
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
    int qadr             = model->jnt_qposadr[jid];
    int dadr             = model->jnt_dofadr[jid];
    data->qpos[qadr]     = pos[0];
    data->qpos[qadr + 1] = pos[1];
    data->qpos[qadr + 2] = pos[2];
    data->qpos[qadr + 3] = quat ? quat[3] : 1.0; // xyzw in, MuJoCo's qpos is wxyz
    data->qpos[qadr + 4] = quat ? quat[0] : 0.0;
    data->qpos[qadr + 5] = quat ? quat[1] : 0.0;
    data->qpos[qadr + 6] = quat ? quat[2] : 0.0;
    for (int k = 0; k < 6; ++k) data->qvel[dadr + k] = 0.0;
}

// Simulation API

bool step(Env *env)
{
    if (!env || !env->model) return true;
    bool running   = true;
    bool integrate = true;
    for_each_hook(env, &EnvHooks::on_step, [env, &running, &integrate](const auto &fn) {
        if (!fn(env, &integrate)) running = false;
    });
    if (!running) return false;
    const bool all_paused =
      !env->robots.empty() && std::ranges::all_of(env->robots, &Robot::paused);
    if (integrate && !all_paused) {
        begin_step(env);
        mj_step2(env->model, env->data);
        env->_impl->computed = false;
        begin_step(env);
    } else {
        ensure_kinematics(env);
    }
    return true;
}

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
        // MuJoCo gives the parent's force on the child; a physical sensor reports the load on it.
        sensor.wrench = -KDL::Wrench(KDL::Vector(f[0], f[1], f[2]), KDL::Vector(t[0], t[1], t[2]));
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
 * reset_part() overload does not compile. A robot's ports are the exception: callers hold
 * pointers into them, so seed_ports() rewrites them in place and a new field needs a line.
 */
static void reset_part(std::vector<Robot *> &robots, Env *env)
{
    for (Robot *r : robots) {
        seed_ports(*r);
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

template<class Part>
concept Resettable = requires(Part &part, Env *env) { reset_part(part, env); };

template<Resettable... Parts> static void reset_parts(Env *env, Parts &...parts)
{
    (reset_part(parts, env), ...);
}

ResetInfo reset(Env *env, const ResetOptions *options)
{
    if (!env || !env->model) return {};
    ResetInfo    info{};
    ResetOptions default_options;
    if (!options) options = &default_options;
    mjModel *model = env->model;
    mjData  *data  = env->data;

    if (options->use_keyframe && options->keyframe >= 0 && options->keyframe < model->nkey) {
        mj_resetDataKeyframe(model, data, options->keyframe);
        info.used_keyframe = true;
        info.keyframe      = options->keyframe;
    } else {
        mj_resetData(model, data);
    }

    mj_forward(model, data);
    record_kinematics(env);
    reset_parts(env, env->robots, env->scene);
    for_each_hook(env, &EnvHooks::on_reset, [env](const auto &fn) { fn(env); });

    // After the re-seed, so the hook can prime commands; what it moves is read back below.
    ResetContext ctx;
    ctx.env     = env;
    ctx.model   = model;
    ctx.data    = data;
    ctx.options = options;
    ctx.info    = &info;
    if (env->on_reset) {
        struct Leave
        {
            bool &flag;
            ~Leave() { flag = false; }
        } leave{ env->_impl->in_reset_hook = true };
        env->on_reset(&ctx);
    }

    ensure_kinematics(env);
    for (Robot *r : env->robots) read_robot(r);
    read_scene(env);
    return info;
}

static Status switch_group(Env *env, int robot, CtrlMode mode)
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
        case CtrlMode::POSITION:
            d->ctrl[a] = d->actuator_length[a];
            break;
        case CtrlMode::VELOCITY:
            d->ctrl[a] = d->actuator_velocity[a];
            break;
        case CtrlMode::TORQUE:
            d->ctrl[a] = 0.0;
            break;
        }
    }
    if (!found) MJ_FAIL("robot " << robot << " has no actuators for this control mode");
    for (int k = 0; k < 3; ++k) m->opt.disableactuator |= 1 << mode_group(robot, k);
    m->opt.disableactuator &= ~(1 << target);
    return {};
}

Status set_control_mode(Env *env, int robot, CtrlMode mode)
{
    if (!env || !env->model) MJ_FAIL("set_control_mode: empty env");
    if (robot < 0 || robot > 9) MJ_FAIL("set_control_mode: robot index " << robot << " not 0..9");
    return switch_group(env, robot, mode);
}

// seed_ports: an explicit switch starts the new mode from where the joints are; a switch the
// caller asked for by setting ctrl_mode keeps the commands it set alongside.
static Status switch_control_mode(Robot *r, CtrlMode mode, bool seed_ports)
{
    RobotInternals &in  = *r->_impl;
    const int       idx = static_cast<int>(mode);
    for (int i = 0; i < r->n_joints; ++i) {
        if (in.mode_ctrl[idx][i] < 0) {
            MJ_FAIL(
              "joint '" << r->joint_names[i] << "' has no actuator for this control mode"
                        << " (add it to RobotSpec::modes)"
            );
        }
    }
    for (const int owner : in.mode_owners) {
        if (Status s = switch_group(in.env, owner, mode); !s) return s;
    }
    const mjData *d = r->data;
    for (int i = 0; seed_ports && i < r->n_joints; ++i) {
        r->jnt_pos_cmd[i] = d->qpos[in.kdl_to_mj_qpos[i]];
        r->jnt_vel_cmd[i] = d->qvel[in.kdl_to_mj_dof[i]];
        r->jnt_trq_cmd[i] = 0.0;
    }
    r->ctrl_mode    = mode;
    in.applied_mode = mode;
    in.mode_applied = true;
    return {};
}

Status set_control_mode(Robot *r, CtrlMode mode)
{
    if (!r || !r->_impl->env) MJ_FAIL("set_control_mode: robot is not registered with an Env");
    return switch_control_mode(r, mode, true);
}

// Gravity compensation on every body the robot's joints carry, its attachments included.
static void set_gravity_compensation(Robot *r, bool on)
{
    mjModel *m = r->model;
    if (r->n_joints == 0) return;
    const int    root  = m->jnt_bodyid[m->dof_jntid[r->_impl->kdl_to_mj_dof[0]]];
    const mjtNum value = on ? 1.0 : 0.0;
    // Set once per switch: a rebuilt model starts without it, so the root's value tells.
    if (m->body_gravcomp[root] == value) return;
    // A parent's id is below its children's, so the subtree starts at its root.
    for (int b = root; b < m->nbody; ++b) {
        int a = b;
        while (a > root) a = m->body_parentid[a];
        if (a == root) m->body_gravcomp[b] = value;
    }
    m->ngravcomp = 0;
    for (int b = 0; b < m->nbody; ++b) m->ngravcomp += m->body_gravcomp[b] != 0.0;
    m->flg_gravcomp = m->ngravcomp > 0;
}

static void apply_robot(Robot *r)
{
    RobotInternals &in = *r->_impl;
    // ctrl_mode may have been set directly; switch before commanding the new mode's actuators.
    if (!in.mode_applied) return;
    if (r->ctrl_mode != in.applied_mode && !switch_control_mode(r, r->ctrl_mode, false))
        r->ctrl_mode = in.applied_mode;
    // A servo holds its setpoint against gravity; a torque command carries gravity itself.
    set_gravity_compensation(r, r->ctrl_mode != CtrlMode::TORQUE);

    mjModel  *m    = r->model;
    mjData   *d    = r->data;
    const int mode = static_cast<int>(r->ctrl_mode);
    for (int i = 0; i < r->n_joints; ++i) {
        const int a = in.mode_ctrl[mode][i];
        if (a < 0) continue;
        const double gear = m->actuator_gear[6 * a];
        double       u    = 0.0;
        switch (r->ctrl_mode) {
        case CtrlMode::POSITION:
            u = gear * r->jnt_pos_cmd[i];
            break;
        case CtrlMode::VELOCITY:
            u = gear * r->jnt_vel_cmd[i];
            break;
        case CtrlMode::TORQUE:
            u = r->jnt_trq_cmd[i] / gear;
            break;
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
        MJ_LOG_ERROR("bind_scene_joint: null scene state or name");
        return nullptr;
    }
    const int jid = mj_name2id(s->model, mjOBJ_JOINT, joint_name);
    if (jid < 0) {
        MJ_LOG_ERROR("bind_scene_joint: no joint named '" << joint_name << "'");
        return nullptr;
    }
    const int type = s->model->jnt_type[jid];
    if (type == mjJNT_FREE || type == mjJNT_BALL) {
        MJ_LOG_ERROR("bind_scene_joint: joint '" << joint_name << "' is not a scalar joint");
        return nullptr;
    }
    for (const auto &slot : s->joints) {
        if (slot.name == joint_name) {
            MJ_LOG_ERROR("bind_scene_joint: joint '" << joint_name << "' is already bound");
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
        MJ_LOG_ERROR("bind_scene_free_body: null scene state or name");
        return nullptr;
    }
    const int bid = mj_name2id(s->model, mjOBJ_BODY, body_name);
    if (bid < 0) {
        MJ_LOG_ERROR("bind_scene_free_body: no body named '" << body_name << "'");
        return nullptr;
    }
    const int jid = free_joint_of_body(s->model, bid);
    if (jid < 0) {
        MJ_LOG_ERROR("bind_scene_free_body: body '" << body_name << "' owns no free joint");
        return nullptr;
    }
    for (const auto &slot : s->free_bodies) {
        if (slot.name == body_name) {
            MJ_LOG_ERROR("bind_scene_free_body: body '" << body_name << "' is already bound");
            return nullptr;
        }
    }
    s->free_bodies.push_back(SceneFreeBodySlot{ {}, body_name, s->model->jnt_qposadr[jid] });
    return &s->free_bodies.back();
}

SceneWrenchSlot *bind_scene_wrench(SceneState *s, const char *body_name)
{
    if (!s || !s->model || !body_name) {
        MJ_LOG_ERROR("bind_scene_wrench: null scene state or name");
        return nullptr;
    }
    const int bid = mj_name2id(s->model, mjOBJ_BODY, body_name);
    if (bid < 0) {
        MJ_LOG_ERROR("bind_scene_wrench: no body named '" << body_name << "'");
        return nullptr;
    }
    for (const auto &slot : s->wrenches) {
        if (slot.name == body_name) {
            MJ_LOG_ERROR("bind_scene_wrench: body '" << body_name << "' is already bound");
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
        MJ_LOG_ERROR("bind_scene_actuator: null scene state or name");
        return nullptr;
    }
    const int aid = actuator_for_name(s->model, name);
    if (aid < 0) {
        MJ_LOG_ERROR("bind_scene_actuator: nothing actuates '" << name << "'");
        return nullptr;
    }
    for (const auto &slot : s->actuators) {
        if (slot.name == name) {
            MJ_LOG_ERROR("bind_scene_actuator: '" << name << "' is already bound");
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
    if (!unbound.empty())
        MJ_LOG_ERROR("scene slots gone from the rebuilt model, unbound:" << unbound);
    return unbound.empty();
}

// Overlay

void clear_trace(Env *env)
{
    if (!env) return;
    const std::lock_guard<std::mutex> lock(env->_impl->overlay_mtx);
    env->_impl->overlays.clear();
}

static void add_overlay_connector(
  Env               *env,
  int                type,
  double             width,
  const KDL::Vector &from,
  const KDL::Vector &to,
  const float        rgba[4]
)
{
    const std::lock_guard<std::mutex> lock(env->_impl->overlay_mtx);
    auto                             &overlays = env->_impl->overlays;
    if (overlays.size() >= env->overlay_max_geoms) return;
    mjvGeom &g = overlays.emplace_back();
    mjv_initGeom(&g, type, nullptr, nullptr, nullptr, rgba ? rgba : env->overlay_rgba.data());
    const mjtNum a[3] = { from.x(), from.y(), from.z() };
    const mjtNum b[3] = { to.x(), to.y(), to.z() };
    mjv_connector(&g, type, width, a, b);
}

void add_trace_segment(Env *env, const KDL::Vector &a, const KDL::Vector &b, const float rgba[4])
{
    if (!env) return;
    add_overlay_connector(env, mjGEOM_LINE, 3.0, a, b, rgba);
}

void add_overlay_arrow(
  Env               *env,
  const KDL::Vector &from,
  const KDL::Vector &dir,
  double             length,
  const float        rgba[4]
)
{
    if (!env) return;
    KDL::Vector unit = dir;
    if (unit.Normalize() < 1e-9 || length <= 0.0) return;
    add_overlay_connector(env, mjGEOM_ARROW, 0.02, from, from + unit * length, rgba);
}

std::vector<mjvGeom> overlay_geoms(const Env *env)
{
    if (!env) return {};
    const std::lock_guard<std::mutex> lock(env->_impl->overlay_mtx);
    return env->_impl->overlays;
}
} // namespace mj_kdl
