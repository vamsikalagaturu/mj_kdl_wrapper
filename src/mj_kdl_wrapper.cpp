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
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

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

// Global viewer/robot pointers - written by init_window / cleanup.
static Robot  *g_robot  = nullptr;
static Viewer *g_viewer = nullptr;


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

// Numerical tolerances. Frame-equality test treats relative deviations below
// kIdentityTol as zero; sim-time guard rejects steps shorter than kSimTimeEps
// caused by floating-point round-trip through MuJoCo.
static constexpr double kIdentityTol = 1e-12;
static constexpr double kSimTimeEps  = 1e-9;

// Default MuJoCo collision category bitmask for wrapper-authored geoms (sets
// category bit 0, matching MuJoCo's MJCF compiler defaults for contype and
// conaffinity). This is a bitmask, not a boolean.
static constexpr int kContactCategoryAll = 1;

// GL/MuJoCo viewer defaults.
static constexpr int    kMsaaSamples    = 4;    // GLFW_SAMPLES (4x MSAA)
static constexpr int    kMaxSceneGeoms  = 2000; // mjv_makeScene buffer capacity
static constexpr double kCamDefaultDist = 2.5;
static constexpr double kCamDefaultAzim = 135.0;
static constexpr double kCamDefaultElev = -20.0;

// Pixel layout for the offscreen recorder (RGB24 / ffmpeg rgb24 input).
static constexpr int kRgbBytesPerPixel = 3;

// Defaults applied when the Simulate UI omits recorder fields.
static constexpr int kRecorderDefaultResIndex = 2; // 720p (see recorder_resolution_from_index)
static constexpr int kRecorderDefaultFps      = 30;

#ifdef MJ_KDL_HAS_EGL
// EGL framebuffer attribute values for the headless recorder.
static constexpr EGLint kEglChannelBits = 8;
static constexpr EGLint kEglDepthBits   = 24;
#endif

// Sole owner of the MuJoCo worldbody name. Every other site that needs the
// worldbody (skybox light, floor geom, resolve_parent with kind == World,
// cameras) calls this so the literal "world" exists in one place.
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

    const char *pfx      = prefix ? prefix : "";
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

void add_floor_to_spec(mjSpec *spec)
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

    mjsGeom *floor = mjs_addGeom(wb, nullptr);
    mjs_setString(mjs_getName(floor->element), "floor");
    mjs_setString(floor->material, "groundplane");
    floor->type        = mjGEOM_PLANE;
    floor->size[0]     = kFloorHalfSize;
    floor->size[1]     = kFloorHalfSize;
    floor->size[2]     = kFloorThickness;
    floor->contype     = kContactCategoryAll;
    floor->conaffinity = kContactCategoryAll;
    floor->condim      = static_cast<int>(Condim::Tangential);
}

void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects)
{
    for (const auto &obj : objects) {
        if (!obj.mjcf_path.empty()) {
            char      err[kMjErrBuf] = {};
            MjSpecPtr asset =
              make_spec_ptr(mj_parseXML(obj.mjcf_path.c_str(), nullptr, err, sizeof(err)));
            if (!asset) {
                LOG_ERROR("mj_parseXML failed for object asset '" << obj.mjcf_path << "': " << err);
                continue;
            }

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
                continue;
            }

            mjsBody *root = first_root_body(asset.get());
            if (!root) {
                LOG_ERROR("no root body found in object asset '" << obj.mjcf_path << "'");
                continue;
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
            continue;
        }

        if (obj.shape == Shape::Unspecified) {
            LOG_ERROR(
              "primitive SceneObject '"
              << obj.name << "' has Shape::Unspecified; set .shape explicitly (BOX/SPHERE/CYLINDER)"
            );
            continue;
        }
        // Validate fields the user must set on a primitive. Free-jointed bodies
        // also need mass > 0; fixed bodies tolerate any non-negative mass.
        const bool need_size2 = (obj.shape == Shape::BOX);
        const bool need_size1 = (obj.shape == Shape::CYLINDER);
        if (obj.size[0] <= 0.0 || (need_size1 && obj.size[1] <= 0.0)
            || (need_size2 && (obj.size[1] <= 0.0 || obj.size[2] <= 0.0))) {
            LOG_ERROR(
              "primitive SceneObject '"
              << obj.name << "' has zero or negative .size for its shape; set explicit dimensions"
            );
            continue;
        }
        if (!obj.fixed && obj.mass <= 0.0) {
            LOG_ERROR(
              "primitive SceneObject '" << obj.name << "' has .mass=" << obj.mass
                                        << "; non-fixed bodies require mass > 0"
            );
            continue;
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
        g->size[0] = obj.size[0];
        g->size[1] = obj.size[1];
        g->size[2] = obj.size[2];
        g->mass    = obj.mass;
        for (int k = 0; k < 4; ++k) g->rgba[k] = obj.rgba[k];
        for (int k = 0; k < 3; ++k) g->friction[k] = obj.friction[k];
        g->contype     = kContactCategoryAll;
        g->conaffinity = kContactCategoryAll;
        g->condim      = static_cast<int>(obj.condim);

        attach_child(spec, obj.attach_to, obj.pos, obj.quat, ob, "");
    }
}

static void add_cameras_to_spec(mjSpec *spec, const std::vector<CameraSpec> &cameras)
{
    mjsBody *wb = world_body(spec);
    for (const auto &cs : cameras) {
        mjsCamera *cam = mjs_addCamera(wb, nullptr);
        mjs_setString(mjs_getName(cam->element), cs.name.c_str());
        cam->pos[0] = cs.pos[0];
        cam->pos[1] = cs.pos[1];
        cam->pos[2] = cs.pos[2];
        cam->fovy   = cs.fovy;
        quat_xyzw_to_mj_quat(cs.quat, cam->quat);
    }
}

// Compile spec into a model and data; spec is always deleted.
bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data)
{
    *out_model = mj_compile(spec, nullptr);
    if (!*out_model) {
        LOG_ERROR("mj_compile failed: " << mjs_getError(spec));
        mj_deleteSpec(spec);
        return false;
    }
    LOG_INFO(
      "scene compiled: nq=" << (*out_model)->nq << " nv=" << (*out_model)->nv
                            << " nbody=" << (*out_model)->nbody
    );
    mj_deleteSpec(spec);
    *out_data = mj_makeData(*out_model);
    if (!*out_data) {
        mj_deleteModel(*out_model);
        *out_model = nullptr;
        return false;
    }
    return true;
}

// KDL helpers

static bool get_site_frame_in_body(
  const mjModel *model,
  mjData        *data,
  const char    *body_name,
  const char    *site_name,
  KDL::Frame    *out
)
{
    if (!model || !data || !body_name || !site_name || !out) return false;

    mj_forward(model, data);

    int body_id = mj_name2id(model, mjOBJ_BODY, body_name);
    int site_id = mj_name2id(model, mjOBJ_SITE, site_name);
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


static bool build_index_map(Robot *s, const std::string &pfx = "")
{
    s->kdl_to_mj_qpos.clear();
    s->kdl_to_mj_dof.clear();
    s->kdl_to_mj_ctrl.clear();
    if (!s->model) return false;
    for (const auto &name : s->joint_names) {
        int id = mj_name2id(s->model, mjOBJ_JOINT, (pfx + name).c_str());
        if (id < 0) {
            LOG_ERROR(
              "joint '" << pfx << name
                        << "' not found in MuJoCo model - check robot prefix or URDF joint names"
            );
            return false;
        }
        s->kdl_to_mj_qpos.push_back(s->model->jnt_qposadr[id]);
        int dof = s->model->jnt_dofadr[id];
        s->kdl_to_mj_dof.push_back(dof);
        // Find the actuator that drives this joint (mjTRN_JOINT type).
        int ctrl_idx = -1;
        for (int ai = 0; ai < s->model->nu; ++ai) {
            if (s->model->actuator_trntype[ai] == mjTRN_JOINT
                && s->model->actuator_trnid[2 * ai] == id) {
                ctrl_idx = ai;
                break;
            }
        }
        s->kdl_to_mj_ctrl.push_back(ctrl_idx);
    }
    int n = s->n_joints;
    s->jnt_pos_msr.assign(n, 0.0);
    s->jnt_vel_msr.assign(n, 0.0);
    s->jnt_trq_msr.assign(n, 0.0);
    s->jnt_pos_cmd.assign(n, 0.0);
    s->jnt_trq_cmd.assign(n, 0.0);
    return true;
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
            jnt = KDL::Joint(jname ? jname : "", origin, axis, jtype);
            if (jname) {
                s->joint_names.push_back(jname);
                double lo = -M_PI, hi = M_PI;
                if (model->jnt_limited[jid]) {
                    lo = model->jnt_range[2 * jid];
                    hi = model->jnt_range[2 * jid + 1];
                }
                s->joint_limits.emplace_back(lo, hi);
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
    char err[kMjErrBuf] = {};
    int  ok             = mj_saveLastXML(path, model, err, sizeof(err));
    if (!ok) {
        LOG_ERROR("mj_saveLastXML failed for '" << path << "': " << err);
    } else {
        LOG_INFO("model saved to '" << path << "'");
    }
    return ok != 0;
}

void destroy_scene(mjModel *model, mjData *data)
{
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
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
    return true;
}

void env_add_robot(Env *env, Robot *robot)
{
    if (!env || !robot) return;
    if (std::find(env->robots.begin(), env->robots.end(), robot) == env->robots.end())
        env->robots.push_back(robot);
}

void cleanup(Env *env)
{
    if (!env) return;
    destroy_scene(env->model, env->data);
    env->model = nullptr;
    env->data  = nullptr;
    env->robots.clear();
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

bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *sc)
{
    if (!sc) return false;
    if (sc->timestep <= 0.0) {
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
    if (sc->add_floor) add_floor_to_spec(scene.get());

    // Objects come before robots so a robot can attach to a SceneObject (e.g.
    // {AttachKind::Site, "table_mount"}). A child object that references
    // another object must appear after its parent in SceneSpec::objects.
    if (!sc->objects.empty()) add_objects_to_spec(scene.get(), sc->objects);

    bool first_arm      = true;
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
        // arm (deep-copied into scene) is freed by MjSpecPtr at scope exit.
    }

    // Robot-less scenes skip the first_arm branch; apply timestep/gravity here.
    if (first_arm) {
        scene->option.timestep   = sc->timestep;
        scene->option.gravity[2] = sc->gravity_z;
    }

    if (!sc->cameras.empty()) add_cameras_to_spec(scene.get(), sc->cameras);
    // compile_and_make_data takes ownership of the raw spec and always deletes it.
    return compile_and_make_data(scene.release(), out_model, out_data);
}

bool scene_add_object(mjModel **model, mjData **data, SceneSpec *spec, const SceneObject &obj)
{
    spec->objects.push_back(obj);
    mjModel *nm = nullptr;
    mjData  *nd = nullptr;
    if (!build_scene(&nm, &nd, spec)) {
        spec->objects.pop_back();
        return false;
    }
    destroy_scene(*model, *data);
    *model = nm;
    *data  = nd;
    return true;
}

bool scene_remove_object(mjModel **model, mjData **data, SceneSpec *spec, const std::string &name)
{
    auto it = std::find_if(spec->objects.begin(), spec->objects.end(), [&](const SceneObject &o) {
        return o.name == name;
    });
    if (it == spec->objects.end()) return false;
    SceneObject removed = std::move(*it);
    spec->objects.erase(it);
    mjModel *nm = nullptr;
    mjData  *nd = nullptr;
    if (!build_scene(&nm, &nd, spec)) {
        spec->objects.push_back(removed);
        return false;
    }
    destroy_scene(*model, *data);
    *model = nm;
    *data  = nd;
    return true;
}

static void reinit_robot(Robot *r, mjModel *model, mjData *data)
{
    /* Objects are always appended after robot bodies in build_scene(), so MuJoCo's
     * compilation preserves all robot joint indices.  Only the pointers change. */
    r->model = model;
    r->data  = data;
}

bool scene_add_object(Env *env, const SceneObject &obj)
{
    if (!env) return false;
    if (!scene_add_object(&env->model, &env->data, &env->spec, obj)) return false;
    for (Robot *r : env->robots) reinit_robot(r, env->model, env->data);
    return true;
}

bool scene_remove_object(Env *env, const std::string &name)
{
    if (!env) return false;
    if (!scene_remove_object(&env->model, &env->data, &env->spec, name)) return false;
    for (Robot *r : env->robots) reinit_robot(r, env->model, env->data);
    return true;
}

std::string scene_object_site_name(const SceneObject &obj, const char *site_name)
{
    if (!site_name) return {};
    return obj.name.empty() ? std::string(site_name) : obj.name + "_" + site_name;
}

bool get_site_frame(const mjModel *model, mjData *data, const char *site_name, KDL::Frame *out)
{
    if (!model || !data || !site_name || !out) return false;

    int sid = mj_name2id(model, mjOBJ_SITE, site_name);
    if (sid < 0) return false;

    mj_forward(model, data);
    const double *p = data->site_xpos + 3 * sid;
    const double *R = data->site_xmat + 9 * sid;
    *out            = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
    return true;
}

bool get_body_frame(const mjModel *model, mjData *data, const char *body_name, KDL::Frame *out)
{
    if (!model || !data || !body_name || !out) return false;

    int bid = mj_name2id(model, mjOBJ_BODY, body_name);
    if (bid < 0) return false;

    mj_forward(model, data);
    const double *p = data->xpos + 3 * bid;
    const double *R = data->xmat + 9 * bid;
    *out            = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
    return true;
}

bool get_joint_position(const mjModel *model, mjData *data, const char *name, double *out)
{
    if (!model || !data || !name || !out) return false;

    int jid = mj_name2id(model, mjOBJ_JOINT, name);
    if (jid < 0) {
        // Not a joint name: accept an actuator name and resolve its transmission joint.
        const int aid = mj_name2id(model, mjOBJ_ACTUATOR, name);
        if (aid < 0) return false;
        if (model->actuator_trntype[aid] == mjTRN_JOINT) {
            jid = model->actuator_trnid[2 * aid];
        } else if (model->actuator_trntype[aid] == mjTRN_TENDON) {
            const int tid = model->actuator_trnid[2 * aid];
            jid           = model->wrap_objid[model->tendon_adr[tid]];
        }
        if (jid < 0) return false;
    }

    *out = data->qpos[model->jnt_qposadr[jid]];
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

bool init_robot_from_mjcf(
  Robot               *r,
  mjModel             *model,
  mjData              *data,
  const char          *base_body,
  const char          *tip_body,
  const char          *prefix,
  const ToolFrameSpec *tool
)
{
    LOG_INFO(
      "init_robot_from_mjcf: '"
      << base_body << "' -> '" << tip_body << "' prefix='" << (prefix ? prefix : "") << "'"
      << (tool && tool->tool_body ? std::string(" tool='") + tool->tool_body + "'" : "")
      << (tool && tool->tcp_site ? std::string(" tcp='") + tool->tcp_site + "'" : "")
    );
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
        if (!get_site_frame_in_body(model, data, tip_body, tool->tcp_site, &tip_T_tcp)) {
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
        // Ensure xpos/xmat are valid for inertia computation.
        mj_forward(model, data);
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
    return true;
}

bool init_robot_from_chain(
  Robot                          *r,
  mjModel                        *model,
  mjData                         *data,
  const KDL::Chain               &chain,
  const std::vector<std::string> &joint_names,
  const char                     *prefix,
  const ToolFrameSpec            *tool
)
{
    LOG_INFO(
      "init_robot_from_chain: " << chain.getNrOfSegments() << " segments, "
                                << chain.getNrOfJoints() << " joints, prefix='"
                                << (prefix ? prefix : "") << "'"
    );
    if (joint_names.size() != chain.getNrOfJoints()) {
        LOG_ERROR(
          "joint_names has " << joint_names.size() << " entries but the chain has "
                             << chain.getNrOfJoints() << " joints"
        );
        return false;
    }

    r->model         = model;
    r->data          = data;
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
        double lo = -M_PI, hi = M_PI;
        int    jid = mj_name2id(model, mjOBJ_JOINT, (pfx + name).c_str());
        if (jid >= 0 && model->jnt_limited[jid]) {
            lo = model->jnt_range[2 * jid];
            hi = model->jnt_range[2 * jid + 1];
        }
        r->joint_limits.emplace_back(lo, hi);
    }

    if (!build_index_map(r, pfx)) return false;
    if (!resolve_ft_sensors(r, tool)) return false;

    LOG_INFO(
      "chain adopted: " << r->n_joints << " joints, " << r->chain.getNrOfSegments() << " segments"
    );
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
    for (int i = 0; i < r->n_joints; ++i) {
        const int ctrl_id = r->kdl_to_mj_ctrl[i];
        if (ctrl_id < 0 || !r->model->actuator_forcelimited[ctrl_id]) continue;
        const double lo = r->model->actuator_forcerange[2 * ctrl_id];
        const double hi = r->model->actuator_forcerange[2 * ctrl_id + 1];
        limits[i]       = std::max(std::abs(lo), std::abs(hi));
    }
    return limits;
}

void cleanup(Robot *r)
{
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
    r->ctrl_mode = CtrlMode::POSITION;
    r->paused    = false;
    r->jnt_pos_msr.clear();
    r->jnt_vel_msr.clear();
    r->jnt_trq_msr.clear();
    r->jnt_pos_cmd.clear();
    r->jnt_trq_cmd.clear();
    r->kdl_to_mj_qpos.clear();
    r->kdl_to_mj_dof.clear();
    r->kdl_to_mj_ctrl.clear();
    if (g_robot == r) g_robot = nullptr;
}

void set_joint_pos(Robot *r, const KDL::JntArray &q, bool call_forward)
{
    if (!r->model || !r->data) return;
    int n = std::min((int)q.rows(), r->n_joints);
    for (int i = 0; i < n; ++i) r->data->qpos[r->kdl_to_mj_qpos[i]] = q(i);
    if (call_forward) mj_forward(r->model, r->data);
}

void set_body_pose(
  mjModel      *model,
  mjData       *data,
  const char   *body_name,
  const double  pos[3],
  const double *quat
)
{
    if (!model || !data || !body_name) return;
    int bid = mj_name2id(model, mjOBJ_BODY, body_name);
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
    data->qpos[qadr + 3] = quat ? quat[0] : 1.0;
    data->qpos[qadr + 4] = quat ? quat[1] : 0.0;
    data->qpos[qadr + 5] = quat ? quat[2] : 0.0;
    data->qpos[qadr + 6] = quat ? quat[3] : 0.0;
    for (int k = 0; k < 6; ++k) data->qvel[dadr + k] = 0.0;
}

// Simulation API

static bool tick_impl(Viewer *v, mjModel *m, mjData *d, bool paused); // defined below

bool step(Robot *s)
{
    if (!s->model || !s->data) return true;
    if (g_viewer) return tick_impl(g_viewer, s->model, s->data, s->paused);
    if (s->paused) return true;
    mj_step(s->model, s->data);
    return true;
}

bool step_n(Robot *s, int n)
{
    for (int i = 0; i < n; ++i)
        if (!step(s)) return false;
    return true;
}

bool step(Viewer *v, mjModel *m, mjData *d) { return tick_impl(v, m, d, false); }

static void sync_robot_after_reset(Robot *r)
{
    if (!r || !r->model || !r->data) return;

    mjData *d = r->data;
    for (int i = 0; i < r->n_joints; ++i) {
        const int    qpos_id = r->kdl_to_mj_qpos[i];
        const int    dof_id  = r->kdl_to_mj_dof[i];
        const int    ctrl_id = r->kdl_to_mj_ctrl[i];
        const double q       = d->qpos[qpos_id];

        r->jnt_pos_msr[i] = q;
        r->jnt_vel_msr[i] = d->qvel[dof_id];
        r->jnt_trq_msr[i] = d->qfrc_actuator[dof_id];
        r->jnt_pos_cmd[i] = q;
        r->jnt_trq_cmd[i] = 0.0;

        d->qfrc_applied[dof_id] = 0.0;
        if (ctrl_id >= 0) d->ctrl[ctrl_id] = q;
    }

    for (auto &sensor : r->ft_sensors) sensor.wrench = KDL::Wrench::Zero();
}

static ResetInfo reset_runtime(
  Env                        *env,
  mjModel                    *model,
  mjData                     *data,
  const std::vector<Robot *> &robots,
  const ResetOptions         *options,
  const ResetHook            &hook,
  bool                        reset_mujoco
)
{
    ResetInfo info{};
    if (!model || !data) return info;

    ResetOptions default_options;
    if (!options) options = &default_options;

    if (reset_mujoco) {
        if (options->use_keyframe && options->keyframe >= 0 && options->keyframe < model->nkey) {
            mj_resetDataKeyframe(model, data, options->keyframe);
            info.used_keyframe = true;
            info.keyframe      = options->keyframe;
        } else {
            mj_resetData(model, data);
        }
    }

    ResetContext ctx;
    ctx.env     = env;
    ctx.model   = model;
    ctx.data    = data;
    ctx.options = options;
    ctx.info    = &info;
    if (hook) hook(&ctx);

    mj_forward(model, data);
    for (Robot *robot : robots) sync_robot_after_reset(robot);

    return info;
}

ResetInfo reset(Env *env, const ResetOptions *options)
{
    if (!env) return {};
    return reset_runtime(env, env->model, env->data, env->robots, options, env->on_reset, true);
}

static mjtNum clamp_ctrlrange(const mjModel *m, int ci, mjtNum u)
{
    if (m->actuator_ctrllimited[ci])
        u = std::clamp(u, m->actuator_ctrlrange[2 * ci], m->actuator_ctrlrange[2 * ci + 1]);
    return u;
}

void update(Robot *r)
{
    if (!r || !r->model || !r->data) return;

    mjModel *m = r->model;
    mjData  *d = r->data;

    for (int i = 0; i < r->n_joints; ++i) {
        const int qpos_id = r->kdl_to_mj_qpos[i];
        const int dof_id  = r->kdl_to_mj_dof[i];
        const int ctrl_id = r->kdl_to_mj_ctrl[i];

        r->jnt_pos_msr[i] = d->qpos[qpos_id];
        r->jnt_vel_msr[i] = d->qvel[dof_id];
        r->jnt_trq_msr[i] = d->qfrc_actuator[dof_id];

        switch (r->ctrl_mode) {
        case CtrlMode::POSITION:
            if (ctrl_id >= 0) d->ctrl[ctrl_id] = clamp_ctrlrange(m, ctrl_id, r->jnt_pos_cmd[i]);
            break;
        case CtrlMode::TORQUE:
            if (ctrl_id >= 0) {
                /* Null the position actuator completely (both kp and kv terms).
                 * Actuator force = kp*(ctrl-pos) - kv*vel (affine bias, biastype=1).
                 * Setting ctrl = pos + (kv/kp)*vel drives force to zero,
                 * making qfrc_applied the sole torque source -- matching the
                 * real robot's pure torque interface. */
                const double kp     = m->actuator_gainprm[ctrl_id * mjNGAIN + 0];
                const double kv     = -m->actuator_biasprm[ctrl_id * mjNBIAS + 2];
                const double vel_ff = (kp > 0.0) ? (kv / kp) * d->qvel[dof_id] : 0.0;
                d->ctrl[ctrl_id]    = d->qpos[qpos_id] + vel_ff;
            }
            d->qfrc_applied[dof_id] = r->jnt_trq_cmd[i];
            break;
        }
    }

    for (auto &sensor : r->ft_sensors) {
        const double *f = d->sensordata + sensor.force_adr;
        const double *t = d->sensordata + sensor.torque_adr;
        sensor.wrench   = KDL::Wrench(KDL::Vector(f[0], f[1], f[2]), KDL::Vector(t[0], t[1], t[2]));
    }
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

struct GLMouseState
{
    bool   btn_left = false, btn_right = false, btn_middle = false;
    double mouse_x = 0, mouse_y = 0;
    double last_click_time = -1.0;
    int    last_click_btn  = -1;
};

static void cb_keyboard(GLFWwindow *, int key, int, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (!g_viewer) return;
    if (key == GLFW_KEY_PERIOD) {
        adjust_realtime_factor(g_viewer, +1);
        return;
    }
    if (key == GLFW_KEY_COMMA) {
        adjust_realtime_factor(g_viewer, -1);
        return;
    }
    if (key == GLFW_KEY_D) {
        g_viewer->pert.select = 0;
        g_viewer->pert.active = 0;
    }
    if (!g_robot) return;
    if (key == GLFW_KEY_SPACE) g_robot->paused = !g_robot->paused;
}

static void cb_mouse_button(GLFWwindow *w, int btn, int act, int)
{
    auto *ms       = static_cast<GLMouseState *>(glfwGetWindowUserPointer(w));
    ms->btn_left   = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    ms->btn_right  = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    ms->btn_middle = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    glfwGetCursorPos(w, &ms->mouse_x, &ms->mouse_y);
    if (!g_robot || !g_viewer) return;

    if (act == GLFW_PRESS) {
        double now          = glfwGetTime();
        bool   dbl          = (now - ms->last_click_time < 0.3) && (btn == ms->last_click_btn);
        ms->last_click_time = dbl ? -1.0 : now;
        ms->last_click_btn  = btn;

        if (dbl) {
            int ww, wh;
            glfwGetWindowSize(w, &ww, &wh);
            mjtNum selpnt[3];
            int    geomid[1] = { -1 }, flexid[1] = { -1 }, skinid[1] = { -1 };
            int    body = mjv_select(
              g_robot->model,
              g_robot->data,
              &g_viewer->opt,
              (mjtNum)wh / ww,
              (mjtNum)ms->mouse_x / ww,
              (mjtNum)(wh - ms->mouse_y) / wh,
              &g_viewer->scn,
              selpnt,
              geomid,
              flexid,
              skinid
            );
            if (body > 0) {
                g_viewer->pert.select     = body;
                g_viewer->pert.skinselect = skinid[0];
                mju_copy3(g_viewer->pert.localpos, selpnt);
                mjv_initPerturb(g_robot->model, g_robot->data, &g_viewer->scn, &g_viewer->pert);
            } else {
                g_viewer->pert.select = 0;
                g_viewer->pert.active = 0;
            }
        }

        if (g_viewer->pert.select > 0) {
            g_viewer->pert.active =
              (btn == GLFW_MOUSE_BUTTON_LEFT) ? mjPERT_TRANSLATE : mjPERT_ROTATE;
            mjv_initPerturb(g_robot->model, g_robot->data, &g_viewer->scn, &g_viewer->pert);
        }
    } else {
        g_viewer->pert.active = 0;
    }
}

static void cb_mouse_move(GLFWwindow *w, double x, double y)
{
    auto *ms = static_cast<GLMouseState *>(glfwGetWindowUserPointer(w));
    if (!g_robot || !g_viewer || (!ms->btn_left && !ms->btn_right && !ms->btn_middle)) return;
    double dx = x - ms->mouse_x, dy = y - ms->mouse_y;
    ms->mouse_x = x;
    ms->mouse_y = y;
    int ww, wh;
    glfwGetWindowSize(w, &ww, &wh);
    bool shift =
      (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
       || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    if (g_viewer->pert.select > 0 && g_viewer->pert.active) {
        // Left drag = MOVE (translate body), Right drag = ROTATE (torque body)
        mjtMouse act = ms->btn_left    ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                       : ms->btn_right ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                                       : mjMOUSE_ZOOM;
        mjv_movePerturb(
          g_robot->model, g_robot->data, act, dx / wh, dy / wh, &g_viewer->scn, &g_viewer->pert
        );
    } else {
        mjtMouse act = ms->btn_left    ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                       : ms->btn_right ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                                       : mjMOUSE_ZOOM;
        mjv_moveCamera(g_robot->model, act, dx / wh, dy / wh, &g_viewer->scn, &g_viewer->cam);
    }
}

static void cb_scroll(GLFWwindow *, double, double yoff)
{
    if (g_robot && g_viewer)
        mjv_moveCamera(
          g_robot->model, mjMOUSE_ZOOM, 0, -0.05 * yoff, &g_viewer->scn, &g_viewer->cam
        );
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

bool init_window(Viewer *v, Robot *r, const char *title, int width, int height)
{
    if (!r->model) return false;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;
    apply_glfw_platform_hints();
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
    glfwWindowHint(GLFW_SAMPLES, kMsaaSamples);

    v->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!v->window) {
        glfwTerminate();
        return false;
    }

    auto *ms = new GLMouseState();
    glfwSetWindowUserPointer(v->window, ms);
    glfwSetKeyCallback(v->window, cb_keyboard);
    glfwSetMouseButtonCallback(v->window, cb_mouse_button);
    glfwSetCursorPosCallback(v->window, cb_mouse_move);
    glfwSetScrollCallback(v->window, cb_scroll);
    glfwSetWindowCloseCallback(v->window, [](GLFWwindow *w) {
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    });
    glfwMakeContextCurrent(v->window);
    glfwSwapInterval(1);

    if (!glfwGetProcAddress("glGenBuffers")) {
        delete ms;
        glfwDestroyWindow(v->window);
        v->window = nullptr;
        glfwTerminate();
        return false;
    }

    mjv_defaultCamera(&v->cam);
    mjv_defaultOption(&v->opt);
    mjv_defaultPerturb(&v->pert);
    mjv_makeScene(r->model, &v->scn, kMaxSceneGeoms);
    mjr_makeContext(r->model, &v->con, mjFONTSCALE_150);
    v->cam.type      = mjCAMERA_FREE;
    v->cam.distance  = kCamDefaultDist;
    v->cam.azimuth   = kCamDefaultAzim;
    v->cam.elevation = kCamDefaultElev;
    g_robot          = r;
    g_viewer         = v;
    return true;
}

static GLFWkeyfun g_sim_prev_key_cb = nullptr;

/* Internal state for init_window_sim(): bundles the mj::Simulate object so it
 * can be stored behind a void* in Viewer._sim_ui.
 *
 * Threading: GlfwAdapter (and therefore the GL context) is created INSIDE
 * render_thread so that glfwMakeContextCurrent() is called on the thread that
 * will own the context.  RenderLoop() runs on that same thread and processes
 * Load() requests from the main thread.  tick() does physics only -- the
 * render thread handles all calls to Render(). */
struct SimUiState
{
    mjvCamera                         cam{};
    mjvOption                         opt{};
    mjvPerturb                        pert{};
    std::unique_ptr<mujoco::Simulate> sim;
    std::thread                       render_thread;
    bool                              sim_ready = false;
    std::mutex                        sim_ready_mtx;
    std::condition_variable           sim_ready_cv;
    double                            prev_sim_time = 0.0;
    std::atomic<int>                  rtf_step{ 0 }; // + faster, - slower (render thread)
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
};

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

static void record_sim_ui_frame(SimUiState *ss, mjModel *m, mjData *d)
{
    if (!ss || !ss->recorder_active || !m || !d) return;
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

    if (!record_frame(&ss->recorder, m, d)) {
        cleanup(&ss->recorder);
        ss->recorder_active = false;
        ss->sim->SetWrapperRecorderState(2);
    }
}

static void sim_ui_key_cb(GLFWwindow *w, int key, int scancode, int action, int mods)
{
    if ((action == GLFW_PRESS || action == GLFW_REPEAT) && g_viewer && g_viewer->_sim_ui) {
        auto *ss = static_cast<SimUiState *>(g_viewer->_sim_ui);
        if (key == GLFW_KEY_PERIOD) {
            ss->rtf_step.fetch_add(+1);
            return;
        }
        if (key == GLFW_KEY_COMMA) {
            ss->rtf_step.fetch_add(-1);
            return;
        }
    }
    if (g_sim_prev_key_cb) g_sim_prev_key_cb(w, key, scancode, action, mods);
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

void cleanup(Viewer *v)
{
    if (v->_sim_ui) {
        auto *ss             = static_cast<SimUiState *>(v->_sim_ui);
        ss->sim->exitrequest = 1;
        if (ss->render_thread.joinable()) ss->render_thread.join();
        if (ss->recorder_active) cleanup(&ss->recorder);
        /* Render thread has stopped, so no one is reading user_scn now. */
        mjv_freeScene(&ss->user_scn);
        delete ss;
        v->_sim_ui = nullptr;
        if (g_viewer == v) {
            g_viewer = nullptr;
            g_robot  = nullptr;
        }
        return;
    }
    if (!v->window) return;
    mjv_freeScene(&v->scn);
    mjr_freeContext(&v->con);
    delete static_cast<GLMouseState *>(glfwGetWindowUserPointer(v->window));
    glfwDestroyWindow(v->window);
    v->window = nullptr;
    glfwTerminate();
    if (g_viewer == v) {
        g_viewer = nullptr;
        g_robot  = nullptr;
    }
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

bool is_running(const Viewer *v)
{
    if (!v) return false;
    if (v->_sim_ui) {
        auto *ss = static_cast<SimUiState *>(v->_sim_ui);
        if (!ss || !ss->sim) return false;
        return !ss->sim->exitrequest.load();
    }
    if (!v->window) return false;
    return !glfwWindowShouldClose(v->window);
}

bool render(Viewer *v, mjModel *m, mjData *d)
{
    if (!v->window) return false;
    if (glfwWindowShouldClose(v->window)) return false;
    glfwPollEvents();
    int w, h;
    glfwGetFramebufferSize(v->window, &w, &h);
    mjrRect vp = { 0, 0, w, h };
    mjv_updateScene(m, d, &v->opt, &v->pert, &v->cam, mjCAT_ALL, &v->scn);
    mjr_render(vp, &v->scn, &v->con);

    glfwSwapBuffers(v->window);
    return true;
}

bool render(Viewer *v, const Robot *r) { return render(v, r->model, r->data); }

static bool init_window_sim_core(
  Viewer *v, mjModel *m, mjData *d, const char *tcp_site, const char *title)
{
    if (!v || !m || !d) return false;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;

    // glfwInitHint() is "main thread only" per GLFW docs -- call before spawning.
    apply_glfw_platform_hints();

    auto *ss = new SimUiState();
    mjv_defaultCamera(&ss->cam);
    mjv_defaultOption(&ss->opt);
    mjv_defaultPerturb(&ss->pert);
    /* If the caller configured Viewer.cam before init_window_sim (e.g. set a
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
        if (ss->glfw_window) g_sim_prev_key_cb = glfwSetKeyCallback(ss->glfw_window, sim_ui_key_cb);
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
        mj_forward(m, d);
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

    // trace follows the robot's TCP site (Frames panel "Trace EE")
    if (tcp_site && *tcp_site) {
        ss->sim->ee_trace_site_ = mj_name2id(m, mjOBJ_SITE, tcp_site);
    }

    v->_sim_ui = ss;
    g_viewer   = v;
    return true;
}

bool init_window_sim(Viewer *v, Robot *r, const char *title)
{
    if (!r || !r->model || !r->data) return false;
    const char *tcp = r->tcp_site.empty() ? nullptr : r->tcp_site.c_str();
    if (!init_window_sim_core(v, r->model, r->data, tcp, title)) return false;
    g_robot = r;
    return true;
}

bool init_window_sim(Viewer *v, mjModel *m, mjData *d, const char *title)
{
    if (!init_window_sim_core(v, m, d, nullptr, title)) return false;
    g_robot = nullptr;
    return true;
}

static bool tick_impl(Viewer *v, mjModel *m, mjData *d, bool paused)
{
    using Clock = std::chrono::steady_clock;
    using Dur   = std::chrono::duration<double>;

    // sim UI path (init_window_sim)
    if (v->_sim_ui) {
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

        double wall_per_step =
          (v->realtime_factor > 0.0) ? m->opt.timestep / v->realtime_factor : 0.0;
        auto now = Clock::now();
        if (wall_per_step > 0.0 && v->_tick_t.time_since_epoch().count() != 0) {
            auto next = v->_tick_t + Dur(wall_per_step);
            if (now < next) std::this_thread::sleep_until(next);
        }
        v->_tick_t = Clock::now();

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
            if (time_jumped_back && g_robot) {
                std::vector<Robot *> robots = { g_robot };
                (void)reset_runtime(nullptr, m, d, robots, nullptr, {}, false);
                v->_tick_t = {};
            }
            ss->prev_sim_time = d->time;

            if (sim->run) {
                if (sim->pert.active) mjv_applyPerturbForce(m, d, &sim->pert);
                mj_step(m, d);
                sim->AddToHistory();
            } else {
                mj_forward(m, d);
            }
        }

        record_sim_ui_frame(ss, m, d);

        // Render is handled by the render thread inside RenderLoop().
        return !sim->exitrequest.load();
    }

    // simple viewer path (init_window)
    if (!v->window || glfwWindowShouldClose(v->window)) return false;

    // Real-time sync: sleep until last tick time + wall time per step.
    auto   now           = Clock::now();
    double wall_per_step = (v->realtime_factor > 0.0) ? m->opt.timestep / v->realtime_factor : 0.0;
    if (wall_per_step > 0.0 && v->_tick_t.time_since_epoch().count() != 0) {
        auto next = v->_tick_t + Dur(wall_per_step);
        if (now < next) std::this_thread::sleep_until(next);
    }
    v->_tick_t = Clock::now();

    if (!paused) {
        if (v->pert.active) mjv_applyPerturbForce(m, d, &v->pert);
        mj_step(m, d);
    }

    render(v, m, d); // includes glfwPollEvents + swap
    return is_running(v);
}

// VideoRecorder -- EGL headless offscreen recording via ffmpeg pipe

#ifdef MJ_KDL_HAS_EGL

struct VideoRecorderImpl
{
    EGLDisplay           egl_dpy = EGL_NO_DISPLAY;
    EGLContext           egl_ctx = EGL_NO_CONTEXT;
    mjvScene             scn{};
    mjrContext           con{};
    FILE                *ffmpeg = nullptr;
    std::string          out_path;
    int                  width  = 0;
    int                  height = 0;
    int                  frames = 0;
    std::vector<uint8_t> rgb_buf;
};

static bool vr_egl_init(VideoRecorderImpl *impl)
{
    const EGLint attrs[] = {
        EGL_RED_SIZE,          kEglChannelBits, EGL_GREEN_SIZE,   kEglChannelBits,
        EGL_BLUE_SIZE,         kEglChannelBits, EGL_ALPHA_SIZE,   kEglChannelBits,
        EGL_DEPTH_SIZE,        kEglDepthBits,   EGL_STENCIL_SIZE, kEglChannelBits,
        EGL_COLOR_BUFFER_TYPE, EGL_RGB_BUFFER,  EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE,   EGL_OPENGL_BIT,  EGL_NONE
    };

    impl->egl_dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (impl->egl_dpy == EGL_NO_DISPLAY) {
        LOG_ERROR("EGL: no display (error 0x" << std::hex << eglGetError() << ")");
        return false;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(impl->egl_dpy, &major, &minor)) {
        LOG_ERROR("EGL: init failed (error 0x" << std::hex << eglGetError() << ")");
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

bool init_video_recorder(
  VideoRecorder *vr,
  mjModel       *model,
  const char    *out_path,
  int            width,
  int            height,
  int            fps
)
{
    if (!vr || !model || !out_path) return false;

    // Set up default camera and options on the user-visible struct.
    mjv_defaultCamera(&vr->cam);
    mjv_defaultFreeCamera(model, &vr->cam);
    mjv_defaultOption(&vr->opt);

    auto *impl     = new VideoRecorderImpl();
    impl->width    = width;
    impl->height   = height;
    impl->out_path = out_path;
    impl->rgb_buf.resize(
      static_cast<size_t>(width) * static_cast<size_t>(height) * kRgbBytesPerPixel
    );

    if (!vr_egl_init(impl)) {
        delete impl;
        return false;
    }

    // MuJoCo rendering contexts.
    mjv_defaultScene(&impl->scn);
    mjr_defaultContext(&impl->con);
    mjv_makeScene(model, &impl->scn, 4000);
    // Shadows cause severe acne in EGL offscreen rendering due to shadow-map
    // precision limits with no antialiasing; disabled by default.
    impl->scn.flags[mjRND_SHADOW] = 0;
    mjr_makeContext(model, &impl->con, mjFONTSCALE_150);
    mjr_setBuffer(mjFB_OFFSCREEN, &impl->con);
    mjr_resizeOffscreen(width, height, &impl->con);

    // Launch ffmpeg: reads raw RGB24 frames from stdin, writes H.264 MP4.
    // -g <fps>: ~1s GOP (vs x264's 250-frame default) so scrubbing seeks land
    // near a keyframe instead of decoding seconds of frames.
    const int gop = std::max(1, fps);
    char cmd[2048];
    snprintf(
      cmd,
      sizeof(cmd),
      "ffmpeg -hide_banner -loglevel error -nostats -y "
      "-f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s %dx%d -r %d "
      "-i pipe:0 -an -vcodec libx264 -pix_fmt yuv420p -preset medium -crf 18 -g %d \"%s\" "
      "2>/dev/null",
      width,
      height,
      fps,
      gop,
      out_path
    );
    impl->ffmpeg = popen(cmd, "w");
    if (!impl->ffmpeg) {
        LOG_ERROR("popen(ffmpeg) failed - is ffmpeg installed and in PATH?");
        mjr_freeContext(&impl->con);
        mjv_freeScene(&impl->scn);
        vr_egl_done(impl);
        delete impl;
        return false;
    }

    vr->_impl = impl;
    LOG_INFO("VideoRecorder: " << width << "x" << height << " @ " << fps << " fps -> " << out_path);
    return true;
}

bool record_frame(VideoRecorder *vr, mjModel *model, mjData *data)
{
    if (!vr || !vr->_impl || !model || !data) return false;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);

    mjv_updateScene(model, data, &vr->opt, nullptr, &vr->cam, mjCAT_ALL, &impl->scn);

    mjrRect vp = { 0, 0, impl->width, impl->height };
    mjr_render(vp, &impl->scn, &impl->con);
    mjr_readPixels(impl->rgb_buf.data(), nullptr, vp, &impl->con);

    // MuJoCo fills the buffer bottom-to-top; flip before piping to ffmpeg.
    const int            row_bytes = kRgbBytesPerPixel * impl->width;
    uint8_t             *buf       = impl->rgb_buf.data();
    std::vector<uint8_t> tmp(static_cast<size_t>(row_bytes));
    for (int top = 0, bot = impl->height - 1; top < bot; ++top, --bot) {
        memcpy(tmp.data(), buf + top * row_bytes, row_bytes);
        memcpy(buf + top * row_bytes, buf + bot * row_bytes, row_bytes);
        memcpy(buf + bot * row_bytes, tmp.data(), row_bytes);
    }

    if (fwrite(buf, 1, impl->rgb_buf.size(), impl->ffmpeg) != impl->rgb_buf.size()) {
        LOG_ERROR("fwrite to ffmpeg pipe failed");
        return false;
    }
    ++impl->frames;
    return true;
}

bool init_video_recorder(
  VideoRecorder  *vr,
  mjModel        *model,
  const char     *out_path,
  VideoResolution resolution,
  int             fps
)
{
    /* 16:9 width for each named height. */
    int h = static_cast<int>(resolution);
    int w = h * 16 / 9;
    return init_video_recorder(vr, model, out_path, w, h, fps);
}

void cleanup(VideoRecorder *vr)
{
    if (!vr || !vr->_impl) return;
    auto *impl = static_cast<VideoRecorderImpl *>(vr->_impl);
    if (impl->ffmpeg) {
        int status   = pclose(impl->ffmpeg);
        impl->ffmpeg = nullptr;
        if (status == 0 && impl->frames > 0) {
            std::fprintf(stderr, "[mj_kdl] recording saved to %s\n", impl->out_path.c_str());
        } else if (status != 0) {
            LOG_ERROR("ffmpeg failed while saving recording to '" << impl->out_path << "'");
        }
    }
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
bool record_frame(VideoRecorder *, mjModel *, mjData *) { return false; }
void cleanup(VideoRecorder *vr)
{
    if (vr) vr->_impl = nullptr;
}

#endif // MJ_KDL_HAS_EGL

} // namespace mj_kdl
