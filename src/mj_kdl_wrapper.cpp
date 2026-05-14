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

#include <kdl/frames.hpp>

#include <chrono>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

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

void ensure_plugins_loaded()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char *env = std::getenv("MUJOCO_PLUGIN_DIR");
        const char *dir = env ? env : MUJOCO_PLUGIN_DIR;
        mj_loadAllPluginLibraries(dir, nullptr);
    });
}

// Global viewer/robot pointers - written by init_window / cleanup.
static Robot  *g_robot  = nullptr;
static Viewer *g_viewer = nullptr;


// Spec-API helpers

void add_skybox_to_spec(mjSpec *spec)
{
    mjsBody *wb = mjs_findBody(spec, "world");

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
    sky->width   = 200;
    sky->height  = 200;

    mjsLight *sun = mjs_addLight(wb, nullptr);
    sun->type     = mjLIGHT_DIRECTIONAL;
    sun->pos[0]   = 0;
    sun->pos[1]   = 0;
    sun->pos[2]   = 4;
}

void add_floor_to_spec(mjSpec *spec)
{
    mjsBody *wb = mjs_findBody(spec, "world");

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
    tex->width   = 300;
    tex->height  = 300;

    mjsMaterial *mat = mjs_addMaterial(spec, nullptr);
    mjs_setString(mjs_getName(mat->element), "groundplane");
    // Set texture at slot mjTEXROLE_RGB (1); vector is pre-initialised with 10 empty strings
    mjs_setInStringVec(mat->textures, mjTEXROLE_RGB, "groundplane");
    mat->texrepeat[0] = 5;
    mat->texrepeat[1] = 5;
    mat->reflectance  = 0.2f;

    mjsGeom *floor = mjs_addGeom(wb, nullptr);
    mjs_setString(mjs_getName(floor->element), "floor");
    mjs_setString(floor->material, "groundplane");
    floor->type        = mjGEOM_PLANE;
    floor->size[0]     = 10;
    floor->size[1]     = 10;
    floor->size[2]     = 0.05;
    floor->contype     = 1;
    floor->conaffinity = 1;
    floor->condim      = 3;
}

void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects)
{
    mjsBody *wb = mjs_findBody(spec, "world");
    for (const auto &obj : objects) {
        if (!obj.mjcf_path.empty()) {
            char    err[2048] = {};
            mjSpec *asset     = mj_parseXML(obj.mjcf_path.c_str(), nullptr, err, sizeof(err));
            if (!asset) {
                LOG_ERROR("mj_parseXML failed for object asset '" << obj.mjcf_path << "': " << err);
                continue;
            }

            mjsBody *asset_world = mjs_findBody(asset, "world");
            mjsElement *first = asset_world ? mjs_firstChild(asset_world, mjOBJ_BODY, 0) : nullptr;
            mjsBody    *root  = first ? mjs_asBody(first) : nullptr;
            if (!root) {
                LOG_ERROR("no root body found in object asset '" << obj.mjcf_path << "'");
                mj_deleteSpec(asset);
                continue;
            }

            mjsFrame *place = mjs_addFrame(wb, nullptr);
            place->pos[0]   = obj.pos[0];
            place->pos[1]   = obj.pos[1];
            place->pos[2]   = obj.pos[2];

            std::string prefix = obj.name.empty() ? "" : obj.name + "_";
            if (!mjs_attach(place->element, root->element, prefix.c_str(), "")) {
                LOG_ERROR("mjs_attach failed for object asset '" << obj.mjcf_path << "': " << mjs_getError(spec));
            }
            mj_deleteSpec(asset);
            continue;
        }

        mjsBody *ob = mjs_addBody(wb, nullptr);
        mjs_setString(mjs_getName(ob->element), obj.name.c_str());
        ob->pos[0] = obj.pos[0];
        ob->pos[1] = obj.pos[1];
        ob->pos[2] = obj.pos[2];

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
        }
        g->size[0] = obj.size[0];
        g->size[1] = obj.size[1];
        g->size[2] = obj.size[2];
        g->mass    = obj.mass;
        for (int k = 0; k < 4; ++k) g->rgba[k] = obj.rgba[k];
        for (int k = 0; k < 3; ++k) g->friction[k] = obj.friction[k];
        g->contype     = 1;
        g->conaffinity = 1;
        g->condim      = obj.condim;
    }
}

static void add_cameras_to_spec(mjSpec *spec, const std::vector<CameraSpec> &cameras)
{
    mjsBody *wb = mjs_findBody(spec, "world");
    for (const auto &cs : cameras) {
        mjsCamera *cam = mjs_addCamera(wb, nullptr);
        mjs_setString(mjs_getName(cam->element), cs.name.c_str());
        cam->pos[0] = cs.pos[0];
        cam->pos[1] = cs.pos[1];
        cam->pos[2] = cs.pos[2];
        cam->fovy   = cs.fovy;
        if (cs.euler[0] || cs.euler[1] || cs.euler[2]) {
            const double  d2r = M_PI / 180.0;
            KDL::Rotation rot = KDL::Rotation::RPY(cs.euler[0]*d2r, cs.euler[1]*d2r, cs.euler[2]*d2r);
            double qx, qy, qz, qw;
            rot.GetQuaternion(qx, qy, qz, qw);
            cam->quat[0] = qw;
            cam->quat[1] = qx;
            cam->quat[2] = qy;
            cam->quat[3] = qz;
        }
    }
}

void configure_spec(mjSpec *spec, const SceneSpec *sc)
{
    spec->option.timestep   = sc->timestep;
    spec->option.gravity[2] = sc->gravity_z;
    spec->compiler.balanceinertia = 1;
    spec->compiler.discardvisual  = 0;
    if (sc->add_skybox) add_skybox_to_spec(spec);
    if (sc->add_floor) add_floor_to_spec(spec);
    if (!sc->objects.empty()) add_objects_to_spec(spec, sc->objects);
    if (!sc->cameras.empty()) add_cameras_to_spec(spec, sc->cameras);
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

// Convert MuJoCo quaternion [w, x, y, z] to KDL::Rotation (KDL expects x, y, z, w).
static KDL::Rotation mj_quat_to_kdl_rot(const double *q)
{ return KDL::Rotation::Quaternion(q[1], q[2], q[3], q[0]); }

static KDL::Rotation mj_xmat_to_kdl_rot(const double *m)
{ return KDL::Rotation(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]); }

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
            if (
              s->model->actuator_trntype[ai] == mjTRN_JOINT
              && s->model->actuator_trnid[2 * ai] == id
            ) {
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
    char err[2048] = {};
    int  ok        = mj_saveLastXML(path, model, err, sizeof(err));
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
      "attach_to_spec: body='" << (a->attach_to ? a->attach_to : "(null)") << "' prefix='"
                               << (a->prefix ? a->prefix : "") << "'"
    );

    char    err[2048] = {};
    mjSpec *att       = mj_parseXML(a->mjcf_path, nullptr, err, sizeof(err));
    if (!att) {
        LOG_ERROR("mj_parseXML failed for attachment '" << a->mjcf_path << "': " << err);
        return false;
    }

    mjsBody *attach_body = mjs_findBody(robot_spec, a->attach_to);
    if (!attach_body) {
        LOG_ERROR("attach body '" << a->attach_to << "' not found in robot spec");
        mj_deleteSpec(att);
        return false;
    }

    // Create an offset frame under the attach body.
    mjsFrame *frame = mjs_addFrame(attach_body, nullptr);
    frame->pos[0]   = a->pos[0];
    frame->pos[1]   = a->pos[1];
    frame->pos[2]   = a->pos[2];
    if (a->euler[0] || a->euler[1] || a->euler[2]) {
        double        d2r = M_PI / 180.0;
        KDL::Rotation rot =
          KDL::Rotation::RPY(a->euler[0] * d2r, a->euler[1] * d2r, a->euler[2] * d2r);
        double qx, qy, qz, qw;
        rot.GetQuaternion(qx, qy, qz, qw);
        frame->quat[0] = qw;
        frame->quat[1] = qx;
        frame->quat[2] = qy;
        frame->quat[3] = qz;
    }

    // Attach the first root body of the attachment (first body child of worldbody).
    mjsBody    *att_world = mjs_findBody(att, "world");
    mjsElement *first     = att_world ? mjs_firstChild(att_world, mjOBJ_BODY, 0) : nullptr;
    mjsBody    *att_root  = first ? mjs_asBody(first) : nullptr;
    if (!att_root) {
        LOG_ERROR("no root body found in attachment spec '" << a->mjcf_path << "'");
        mj_deleteSpec(att);
        return false;
    }

    const char *pfx = a->prefix ? a->prefix : "";
    if (!mjs_attach(frame->element, att_root->element, pfx, "")) {
        LOG_ERROR("mjs_attach failed: " << mjs_getError(robot_spec));
        mj_deleteSpec(att);
        return false;
    }
    mj_deleteSpec(att); // deep-copied into robot_spec; source can be freed

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
    if (!sc || sc->robots.empty()) return false;
    ensure_plugins_loaded();
    LOG_INFO(
      "build_scene: " << sc->robots.size() << " robot(s)"
                      << ", objects=" << sc->objects.size()
    );

    mjSpec *scene = mj_makeSpec();
    if (!scene) {
        LOG_ERROR("mj_makeSpec() failed");
        return false;
    }
    mjsBody *world     = mjs_findBody(scene, "world");
    bool     first_arm = true;

    char err[2048] = {};
    for (int ai = 0; ai < (int)sc->robots.size(); ++ai) {
        const RobotSpec &rs = sc->robots[ai];
        if (!rs.path) {
            LOG_ERROR("robots[" << ai << "].path is null");
            mj_deleteSpec(scene);
            return false;
        }

        mjSpec *arm = mj_parseXML(rs.path, nullptr, err, sizeof(err));
        if (!arm) {
            LOG_ERROR("mj_parseXML failed for '" << rs.path << "': " << err);
            mj_deleteSpec(scene);
            return false;
        }

        // Inherit physics options (integrator, solver, etc.) from the first arm.
        if (first_arm) {
            scene->option = arm->option;
            first_arm     = false;
        }

        // Apply attachment chain in order (mount, sensor, gripper, etc.).
        for (const auto &att : rs.attachments) {
            if (!attach_to_spec(arm, &att)) {
                mj_deleteSpec(arm);
                mj_deleteSpec(scene);
                return false;
            }
        }

        // Create a placement frame in the scene at the desired position/orientation.
        mjsFrame *place = mjs_addFrame(world, nullptr);
        place->pos[0]   = rs.pos[0];
        place->pos[1]   = rs.pos[1];
        place->pos[2]   = rs.pos[2];
        if (rs.euler[0] || rs.euler[1] || rs.euler[2]) {
            double        d2r = M_PI / 180.0;
            KDL::Rotation rot =
              KDL::Rotation::RPY(rs.euler[0] * d2r, rs.euler[1] * d2r, rs.euler[2] * d2r);
            double qx, qy, qz, qw;
            rot.GetQuaternion(qx, qy, qz, qw);
            place->quat[0] = qw;
            place->quat[1] = qx;
            place->quat[2] = qy;
            place->quat[3] = qz;
        }

        // Attach the arm's root body (first child of worldbody) into the scene.
        mjsBody    *arm_world = mjs_findBody(arm, "world");
        mjsElement *first     = arm_world ? mjs_firstChild(arm_world, mjOBJ_BODY, 0) : nullptr;
        mjsBody    *arm_root  = first ? mjs_asBody(first) : nullptr;
        if (!arm_root) {
            LOG_ERROR("no root body found in arm spec '" << rs.path << "'");
            mj_deleteSpec(arm);
            mj_deleteSpec(scene);
            return false;
        }

        const char *pfx = rs.prefix ? rs.prefix : "";
        if (!mjs_attach(place->element, arm_root->element, pfx, "")) {
            LOG_ERROR("mjs_attach failed for arm " << ai << ": " << mjs_getError(scene));
            mj_deleteSpec(arm);
            mj_deleteSpec(scene);
            return false;
        }
        mj_deleteSpec(arm); // deep-copied into scene; source can be freed
    }

    // Apply SceneSpec settings: override timestep/gravity, add floor/skybox/objects.
    configure_spec(scene, sc);
    return compile_and_make_data(scene, out_model, out_data);
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
    *out = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
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
    *out = KDL::Frame(mj_xmat_to_kdl_rot(R), KDL::Vector(p[0], p[1], p[2]));
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
    if (!build_kdl_from_model(r, model, base_body, tip_body)) return false;
    if (!build_index_map(r, prefix ? prefix : "")) return false;

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
    } else if (tool && !Equal(tool->tcp_frame, KDL::Frame::Identity(), 1e-12)) {
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
        r->chain.addSegment(
          KDL::Segment(
            tool->tool_body, KDL::Joint(KDL::Joint::None), KDL::Frame::Identity(), tool_inertia
          )
        );
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
  mjModel    *model,
  mjData     *data,
  const char *body_name,
  const double pos[3],
  const double *quat
)
{
    if (!model || !data || !body_name) return;
    int bid = mj_name2id(model, mjOBJ_BODY, body_name);
    if (bid < 0) return;
    int jnt_start = model->body_jntadr[bid];
    int jnt_count = model->body_jntnum[bid];
    int jid = -1;
    for (int k = 0; k < jnt_count; ++k) {
        if (model->jnt_type[jnt_start + k] == mjJNT_FREE) {
            jid = jnt_start + k;
            break;
        }
    }
    if (jid < 0) return;
    int qadr = model->jnt_qposadr[jid];
    int dadr  = model->jnt_dofadr[jid];
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
    if (g_viewer)
        return tick_impl(g_viewer, s->model, s->data, s->paused);
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

bool step(Viewer *v, mjModel *m, mjData *d)
{
    return tick_impl(v, m, d, false);
}

static void sync_robot_after_reset(Robot *r)
{
    if (!r || !r->model || !r->data) return;

    mjData *d = r->data;
    for (int i = 0; i < r->n_joints; ++i) {
        const int qpos_id = r->kdl_to_mj_qpos[i];
        const int dof_id  = r->kdl_to_mj_dof[i];
        const int ctrl_id = r->kdl_to_mj_ctrl[i];
        const double q    = d->qpos[qpos_id];

        r->jnt_pos_msr[i] = q;
        r->jnt_vel_msr[i] = d->qvel[dof_id];
        r->jnt_trq_msr[i] = d->qfrc_actuator[dof_id];
        r->jnt_pos_cmd[i] = q;
        r->jnt_trq_cmd[i] = 0.0;

        d->qfrc_applied[dof_id] = 0.0;
        if (ctrl_id >= 0) d->ctrl[ctrl_id] = q;
    }
}

static ResetInfo reset_runtime(
  Env                *env,
  mjModel            *model,
  mjData             *data,
  const std::vector<Robot *> &robots,
  const ResetOptions *options,
  const ResetHook    &hook,
  bool                reset_mujoco
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
            if (ctrl_id >= 0)
                d->ctrl[ctrl_id] = clamp_ctrlrange(m, ctrl_id, r->jnt_pos_cmd[i]);
            break;
        case CtrlMode::TORQUE:
            if (ctrl_id >= 0) {
                /* Null the position actuator completely (both kp and kv terms).
                 * Actuator force = kp*(ctrl-pos) - kv*vel (affine bias, biastype=1).
                 * Setting ctrl = pos + (kv/kp)*vel drives force to zero,
                 * making qfrc_applied the sole torque source -- matching the
                 * real robot's pure torque interface. */
                const double kp =  m->actuator_gainprm[ctrl_id * mjNGAIN + 0];
                const double kv = -m->actuator_biasprm[ctrl_id * mjNBIAS + 2];
                const double vel_ff = (kp > 0.0) ? (kv / kp) * d->qvel[dof_id] : 0.0;
                d->ctrl[ctrl_id] = d->qpos[qpos_id] + vel_ff;
            }
            d->qfrc_applied[dof_id] = r->jnt_trq_cmd[i];
            break;
        }
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
        double next = v->realtime_factor * kStep;
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
    glfwWindowHint(GLFW_SAMPLES, 4);

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
    mjv_makeScene(r->model, &v->scn, 2000);
    mjr_makeContext(r->model, &v->con, mjFONTSCALE_150);
    v->cam.type      = mjCAMERA_FREE;
    v->cam.distance  = 2.5;
    v->cam.azimuth   = 135.0;
    v->cam.elevation = -20.0;
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
    double               prev_sim_time        = 0.0;
    std::atomic<int>     rtf_step{ 0 };       // + faster, - slower (render thread)
    GLFWwindow          *glfw_window          = nullptr;
    VideoRecorder        recorder;
    bool                 recorder_active      = false;
    int                  record_camera        = 0; // 0=current, 1=free, 2=tracking, 3+=fixed cam
    int                  record_frame_stride  = 1;
    int                  record_frame_counter = 0;
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
    int  camera     = 0;
    int  resolution = 2;
    int  fps        = 30;
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
        auto *ss = static_cast<SimUiState *>(v->_sim_ui);
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        return use_camera_impl(&ss->cam, model, name);
    }
    return use_camera_impl(&v->cam, model, name);
}

void cleanup(Viewer *v)
{
    if (v->_sim_ui) {
        auto *ss             = static_cast<SimUiState *>(v->_sim_ui);
        ss->sim->exitrequest = 1;
        if (ss->render_thread.joinable()) ss->render_thread.join();
        if (ss->recorder_active) cleanup(&ss->recorder);
        delete ss;
        v->_sim_ui = nullptr;
        if (g_viewer == v) g_viewer = nullptr;
        return;
    }
    if (!v->window) return;
    mjv_freeScene(&v->scn);
    mjr_freeContext(&v->con);
    delete static_cast<GLMouseState *>(glfwGetWindowUserPointer(v->window));
    glfwDestroyWindow(v->window);
    v->window = nullptr;
    glfwTerminate();
    if (g_viewer == v) g_viewer = nullptr;
}

bool is_running(const Viewer *v)
{
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

bool init_window_sim(Viewer *v, Robot *r, const char *title)
{
    if (!r->model) return false;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;

    // glfwInitHint() is "main thread only" per GLFW docs -- call before spawning.
    apply_glfw_platform_hints();

    auto *ss = new SimUiState();
    mjv_defaultCamera(&ss->cam);
    mjv_defaultOption(&ss->opt);
    mjv_defaultPerturb(&ss->pert);
    /* If the caller configured Viewer.cam before init_window_sim (e.g. set a
     * named camera via use_camera() or a free-camera distance > 0), apply it. */
    if (v->cam.type != mjCAMERA_FREE || v->cam.distance > 0.0)
        ss->cam = v->cam;

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
        if (ss->glfw_window)
            g_sim_prev_key_cb = glfwSetKeyCallback(ss->glfw_window, sim_ui_key_cb);
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
    ss->sim->Load(r->model, r->data, title);
    ss->sim->SetWrapperRealtimeFactor(v->realtime_factor);
    if (ss->sim->font != 2) {
        ss->sim->font = 2; // 150%: 0=50% 1=100% 2=150% 3=200%
        ss->sim->Load(r->model, r->data, title);
        ss->sim->SetWrapperRealtimeFactor(v->realtime_factor);
    }
    {
        std::unique_lock<std::recursive_mutex> lock(ss->sim->mtx);
        mj_forward(r->model, r->data);
    }

    v->_sim_ui = ss;
    g_viewer   = v;
    g_robot    = r;
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
        int rtf_step = ss->rtf_step.exchange(0);
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

        double wall_per_step = (v->realtime_factor > 0.0)
                                   ? m->opt.timestep / v->realtime_factor
                                   : 0.0;
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
                ss->prev_sim_time > m->opt.timestep &&
                d->time < ss->prev_sim_time - 1e-9;
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
    auto   now          = Clock::now();
    double wall_per_step = (v->realtime_factor > 0.0)
                               ? m->opt.timestep / v->realtime_factor
                               : 0.0;
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
    const EGLint attrs[] = { EGL_RED_SIZE,
                             8,
                             EGL_GREEN_SIZE,
                             8,
                             EGL_BLUE_SIZE,
                             8,
                             EGL_ALPHA_SIZE,
                             8,
                             EGL_DEPTH_SIZE,
                             24,
                             EGL_STENCIL_SIZE,
                             8,
                             EGL_COLOR_BUFFER_TYPE,
                             EGL_RGB_BUFFER,
                             EGL_SURFACE_TYPE,
                             EGL_PBUFFER_BIT,
                             EGL_RENDERABLE_TYPE,
                             EGL_OPENGL_BIT,
                             EGL_NONE };

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

    auto *impl   = new VideoRecorderImpl();
    impl->width  = width;
    impl->height = height;
    impl->out_path = out_path;
    impl->rgb_buf.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);

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
    char cmd[2048];
    snprintf(
      cmd,
      sizeof(cmd),
      "ffmpeg -hide_banner -loglevel error -nostats -y "
      "-f rawvideo -vcodec rawvideo -pix_fmt rgb24 -s %dx%d -r %d "
      "-i pipe:0 -an -vcodec libx264 -pix_fmt yuv420p -preset medium -crf 18 \"%s\" "
      "2>/dev/null",
      width,
      height,
      fps,
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
    const int            row_bytes = 3 * impl->width;
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
        int status = pclose(impl->ffmpeg);
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
