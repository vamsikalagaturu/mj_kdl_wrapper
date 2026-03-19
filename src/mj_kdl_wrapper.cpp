/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Vamsi Kalagaturu
 * See LICENSE for details. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "urdf_model/joint.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

#include "simulate.h"
#include "glfw_adapter.h"

#include <tinyxml2.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;

/* Load MuJoCo decoder plugins (STL, OBJ, ...) once at first use.
 * Required since MuJoCo 3.6.0 moved mesh decoders to plugin libraries. */
static void ensure_plugins_loaded()
{
    static std::once_flag flag;
    std::call_once(flag, []() {
        const char *env = std::getenv("MUJOCO_PLUGIN_DIR");
        const char *dir = env ? env : MUJOCO_PLUGIN_DIR;
        mj_loadAllPluginLibraries(dir, nullptr);
    });
}

namespace mj_kdl {

/* Logging ----------------------------------------------------------------- */

static LogLevel g_log_level = LogLevel::ERROR; // most verbose by default

void     set_log_level(LogLevel level) { g_log_level = level; }
LogLevel get_log_level() { return g_log_level; }

/* Emit a log message to stderr with ANSI colour, location, and function name.
 * Only file basename is shown (not the full path). */
#define MJ_FILENAME_ (::strrchr(__FILE__, '/') ? ::strrchr(__FILE__, '/') + 1 : __FILE__)

#define MJ_LOG_(lvl_enum, color, label, expr)                     \
    do {                                                          \
        if (g_log_level >= mj_kdl::LogLevel::lvl_enum) {          \
            std::ostringstream _mj_oss;                           \
            _mj_oss << expr;                                      \
            std::fprintf(stderr,                                  \
              color "[mj_kdl " label "] %s:%d (%s): %s\033[0m\n", \
              MJ_FILENAME_,                                       \
              __LINE__,                                           \
              __func__,                                           \
              _mj_oss.str().c_str());                             \
        }                                                         \
    } while (0)

#define LOG_INFO(expr) MJ_LOG_(INFO, "", "INFO ", expr)
#define LOG_WARN(expr) MJ_LOG_(WARN, "\033[33m", "WARN ", expr)
#define LOG_ERROR(expr) MJ_LOG_(ERROR, "\033[31m", "ERROR", expr)

/* URDF preprocessing */

static bool preprocess_urdf(const std::string &in, const std::string &out)
{
    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(in.c_str()) != XML_SUCCESS) return false;
    XMLElement *robot = doc.FirstChildElement("robot");
    if (!robot) return false;

    XMLElement *mj = robot->FirstChildElement("mujoco");
    if (!mj) {
        mj = doc.NewElement("mujoco");
        robot->InsertFirstChild(mj);
    }
    XMLElement *cmp = mj->FirstChildElement("compiler");
    if (!cmp) {
        cmp = doc.NewElement("compiler");
        mj->InsertFirstChild(cmp);
    }
    cmp->SetAttribute("balanceinertia", "true");
    cmp->SetAttribute(
      "meshdir", fs::absolute(fs::path(in).parent_path() / "meshes").string().c_str());

    for (XMLElement *link = robot->FirstChildElement("link"); link;
         link             = link->NextSiblingElement("link")) {
        for (const char *vis_tag : { "visual", "collision" }) {
            for (XMLElement *vis = link->FirstChildElement(vis_tag); vis;
                 vis             = vis->NextSiblingElement(vis_tag)) {
                XMLElement *geom = vis->FirstChildElement("geometry");
                if (!geom) continue;
                XMLElement *mesh = geom->FirstChildElement("mesh");
                if (!mesh) continue;
                const char *fn = mesh->Attribute("filename");
                if (fn && std::string(fn).rfind("package://", 0) == 0)
                    mesh->SetAttribute("filename", fs::path(fn).filename().string().c_str());
            }
        }
    }
    return doc.SaveFile(out.c_str()) == XML_SUCCESS;
}

/* Spec-API helpers (single-robot path) */

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

void add_table_to_spec(mjSpec *spec, const TableSpec &t)
{
    mjsBody *wb         = mjs_findBody(spec, "world");
    double   sz         = t.pos[2]; // surface z
    double   half_thick = t.thickness * 0.5;
    double   top_cz     = sz - half_thick;  // tabletop centre in world
    double   leg_h      = sz - t.thickness; // leg height (floor->bottom of top)

    mjsBody *tb = mjs_addBody(wb, nullptr);
    mjs_setString(mjs_getName(tb->element), "table");
    tb->pos[0] = t.pos[0];
    tb->pos[1] = t.pos[1];
    tb->pos[2] = top_cz;

    // tabletop
    mjsGeom *top = mjs_addGeom(tb, nullptr);
    mjs_setString(mjs_getName(top->element), "table_top");
    top->type    = mjGEOM_BOX;
    top->size[0] = t.top_size[0];
    top->size[1] = t.top_size[1];
    top->size[2] = half_thick;
    for (int k = 0; k < 4; ++k) top->rgba[k] = t.rgba[k];
    top->contype     = 1;
    top->conaffinity = 1;
    top->condim      = 3;

    // 4 legs (only if there's room)
    if (leg_h > 0.0) {
        double       half_leg      = leg_h * 0.5;
        double       leg_rel_z     = -(sz * 0.5); // relative to body = -half_thick - half_leg
        double       lx            = t.top_size[0] - t.leg_radius;
        double       ly            = t.top_size[1] - t.leg_radius;
        const double corners[4][2] = { { lx, ly }, { -lx, ly }, { lx, -ly }, { -lx, -ly } };
        for (int i = 0; i < 4; ++i) {
            mjsGeom *leg = mjs_addGeom(tb, nullptr);
            char     nm[32];
            std::snprintf(nm, sizeof(nm), "table_leg%d", i);
            mjs_setString(mjs_getName(leg->element), nm);
            leg->type    = mjGEOM_CYLINDER;
            leg->size[0] = t.leg_radius;
            leg->size[1] = half_leg;
            leg->pos[0]  = corners[i][0];
            leg->pos[1]  = corners[i][1];
            leg->pos[2]  = leg_rel_z;
            for (int k = 0; k < 4; ++k) leg->rgba[k] = t.rgba[k];
            leg->contype     = 1;
            leg->conaffinity = 1;
            leg->condim      = 3;
        }
    }
}

void add_objects_to_spec(mjSpec *spec, const std::vector<SceneObject> &objects)
{
    mjsBody *wb = mjs_findBody(spec, "world");
    for (const auto &obj : objects) {
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
        case ObjShape::BOX:
            g->type = mjGEOM_BOX;
            break;
        case ObjShape::SPHERE:
            g->type = mjGEOM_SPHERE;
            break;
        case ObjShape::CYLINDER:
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

/* Multi-robot XML helpers (N > 1 path) */

static tinyxml2::XMLElement *xml_deep_clone(const tinyxml2::XMLElement *src,
  tinyxml2::XMLDocument                                                *dst)
{
    tinyxml2::XMLElement *e = dst->NewElement(src->Name());
    for (const tinyxml2::XMLAttribute *a = src->FirstAttribute(); a; a = a->Next())
        e->SetAttribute(a->Name(), a->Value());
    for (const tinyxml2::XMLNode *c = src->FirstChild(); c; c = c->NextSibling())
        if (const tinyxml2::XMLElement *ce = c->ToElement())
            e->InsertEndChild(xml_deep_clone(ce, dst));
    return e;
}

static void xml_prefix_names(tinyxml2::XMLElement *e, const std::string &pfx)
{
    if (pfx.empty()) return;
    auto pfx_attr = [&](const char *a) {
        const char *v = e->Attribute(a);
        if (v) e->SetAttribute(a, (pfx + v).c_str());
    };
    pfx_attr("name");
    pfx_attr("class");
    pfx_attr("childclass");
    const char *tag = e->Name();
    if (!std::strcmp(tag, "geom") || !std::strcmp(tag, "site")) {
        pfx_attr("mesh");
        pfx_attr("material");
    } else if (!std::strcmp(tag, "material")) {
        pfx_attr("texture");
    } else if (!std::strcmp(tag, "joint")) {
        pfx_attr("joint");
        pfx_attr("joint1");
        pfx_attr("joint2"); // tendon and equality refs
    } else if (!std::strcmp(tag, "general") || !std::strcmp(tag, "position")) {
        pfx_attr("joint");
        pfx_attr("tendon");
    } else if (!std::strcmp(tag, "fixed")) {
        pfx_attr("joint");
    } else if (!std::strcmp(tag, "exclude") || !std::strcmp(tag, "connect")) {
        pfx_attr("body1");
        pfx_attr("body2");
    }
    for (auto *c = e->FirstChildElement(); c; c = c->NextSiblingElement()) xml_prefix_names(c, pfx);
}

/* Returns the first child element named tag; creates and appends one if absent. */
static tinyxml2::XMLElement *
  xml_get_or_create(tinyxml2::XMLElement *parent, const char *tag, tinyxml2::XMLDocument &doc)
{
    tinyxml2::XMLElement *el = parent->FirstChildElement(tag);
    if (!el) {
        el = doc.NewElement(tag);
        parent->InsertEndChild(el);
    }
    return el;
}

/* Deep-clone src into dst's document, then prefix all name/ref attributes. */
static tinyxml2::XMLElement *xml_clone_prefixed(const tinyxml2::XMLElement *src,
  tinyxml2::XMLDocument                                                    *dst,
  const std::string                                                        &pfx)
{
    tinyxml2::XMLElement *cl = xml_deep_clone(src, dst);
    xml_prefix_names(cl, pfx);
    return cl;
}

static bool urdf_to_raw_mjcf(const std::string &proc, const std::string &raw)
{
    char     err[2048] = {};
    mjModel *tmp       = mj_loadXML(proc.c_str(), nullptr, err, sizeof(err));
    if (!tmp) {
        LOG_ERROR("mj_loadXML failed for '" << proc << "': " << err);
        return false;
    }
    int r = mj_saveLastXML(raw.c_str(), tmp, err, sizeof(err));
    mj_deleteModel(tmp);
    if (!r) {
        LOG_ERROR("mj_saveLastXML failed for '" << raw << "': " << err);
        return false;
    }
    return true;
}

static bool
  combine_mjcf(const std::vector<std::string> &raws, const SceneSpec *sc, const std::string &out)
{
    using namespace tinyxml2;
    XMLDocument base;
    if (base.LoadFile(raws[0].c_str()) != XML_SUCCESS) {
        LOG_ERROR("failed to parse base MJCF '" << raws[0] << "'");
        return false;
    }
    XMLElement *bmj = base.FirstChildElement("mujoco");
    if (!bmj) return false;
    XMLElement *bwb = xml_get_or_create(bmj, "worldbody", base);
    XMLElement *bas = xml_get_or_create(bmj, "asset", base);

    auto wrap_robot =
      [&](XMLElement *parent, const std::vector<XMLElement *> &kids, const SceneRobot &r, int idx) {
          char name[128];
          std::snprintf(name, sizeof(name), "%srobot_%d_base", r.prefix, idx);
          XMLElement *w = base.NewElement("body");
          w->SetAttribute("name", name);
          char pos[64];
          std::snprintf(pos, sizeof(pos), "%.4f %.4f %.4f", r.pos[0], r.pos[1], r.pos[2]);
          w->SetAttribute("pos", pos);
          if (r.euler[0] || r.euler[1] || r.euler[2]) {
              char eu[64];
              std::snprintf(eu, sizeof(eu), "%.4f %.4f %.4f", r.euler[0], r.euler[1], r.euler[2]);
              w->SetAttribute("euler", eu);
          }
          for (auto *k : kids) w->InsertEndChild(k);
          if (std::strlen(r.prefix) > 0)
              for (auto *k = w->FirstChildElement(); k; k = k->NextSiblingElement())
                  xml_prefix_names(k, r.prefix);
          parent->InsertEndChild(w);
      };

    // robot 0: rewrap existing worldbody children
    {
        std::vector<XMLElement *> kids;
        for (auto *c = bwb->FirstChildElement(); c; c = c->NextSiblingElement())
            kids.push_back(xml_deep_clone(c, &base));
        while (bwb->FirstChild()) bwb->DeleteChild(bwb->FirstChild());
        if (std::strlen(sc->robots[0].prefix) > 0)
            for (auto *a = bas->FirstChildElement(); a; a = a->NextSiblingElement())
                xml_prefix_names(a, sc->robots[0].prefix);
        wrap_robot(bwb, kids, sc->robots[0], 0);
    }

    for (int i = 1; i < (int)raws.size(); ++i) {
        XMLDocument di;
        if (di.LoadFile(raws[i].c_str()) != XML_SUCCESS) {
            LOG_ERROR("failed to parse MJCF for robot " << i << " ('" << raws[i] << "')");
            return false;
        }
        XMLElement *mi = di.FirstChildElement("mujoco");
        if (!mi) return false;
        const std::string pfx(sc->robots[i].prefix);
        if (XMLElement *ai = mi->FirstChildElement("asset"))
            for (auto *a = ai->FirstChildElement(); a; a = a->NextSiblingElement())
                bas->InsertEndChild(xml_clone_prefixed(a, &base, pfx));
        XMLElement *wi = mi->FirstChildElement("worldbody");
        if (!wi) continue;
        std::vector<XMLElement *> kids;
        for (auto *c = wi->FirstChildElement(); c; c = c->NextSiblingElement())
            kids.push_back(xml_deep_clone(c, &base));
        wrap_robot(bwb, kids, sc->robots[i], i);
    }
    return base.SaveFile(out.c_str()) == XML_SUCCESS;
}

/* KDL helpers */

/* Convert MuJoCo quaternion [w, x, y, z] to KDL::Rotation (KDL expects x, y, z, w). */
static KDL::Rotation mj_quat_to_kdl_rot(const double *q)
{
    return KDL::Rotation::Quaternion(q[1], q[2], q[3], q[0]);
}

/* Extract the full rigid-body inertia for body bid from a compiled mjModel. */
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
    return KDL::RigidBodyInertia(mass,
      KDL::Vector(ip[0], ip[1], ip[2]),
      KDL::RotationalInertia(I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2]));
}

static bool
  load_kdl_chain(Robot *s, const std::string &urdf, const char *base_link, const char *tip_link)
{
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf, tree)) {
        LOG_ERROR("kdl_parser::treeFromFile failed for URDF '" << urdf << "'");
        return false;
    }
    if (!tree.getChain(base_link, tip_link, s->chain)) {
        LOG_ERROR("KDL chain not found: '" << base_link << "' -> '" << tip_link
                                           << "'  - verify link names in URDF");
        return false;
    }
    auto model = urdf::parseURDFFile(urdf);
    s->joint_names.clear();
    s->joint_limits.clear();
    for (unsigned i = 0; i < s->chain.getNrOfSegments(); ++i) {
        const KDL::Joint &j = s->chain.getSegment(i).getJoint();
        if (j.getType() == KDL::Joint::None) continue;
        const std::string &name = j.getName();
        s->joint_names.push_back(name);
        double lo = -M_PI, hi = M_PI;
        if (model) {
            auto uj = model->getJoint(name);
            if (uj && uj->limits && uj->type != urdf::Joint::CONTINUOUS) {
                lo = uj->limits->lower;
                hi = uj->limits->upper;
            }
        }
        s->joint_limits.emplace_back(lo, hi);
    }
    s->n_joints = (int)s->chain.getNrOfJoints();
    return true;
}

static void sync_chain_inertias(Robot *s, const std::string &pfx)
{
    if (!s->model) return;
    KDL::Chain chain;
    for (unsigned si = 0; si < s->chain.getNrOfSegments(); ++si) {
        const KDL::Segment   &seg = s->chain.getSegment(si);
        int                   bid = mj_name2id(s->model, mjOBJ_BODY, (pfx + seg.getName()).c_str());
        KDL::RigidBodyInertia inertia;
        if (bid >= 0) inertia = mj_body_inertia(s->model, bid);
        chain.addSegment(KDL::Segment(seg.getName(), seg.getJoint(), seg.getFrameToTip(), inertia));
    }
    s->chain = chain;
}

static void build_index_map(Robot *s, const std::string &pfx = "")
{
    s->kdl_to_mj_qpos.clear();
    s->kdl_to_mj_dof.clear();
    if (!s->model) return;
    for (const auto &name : s->joint_names) {
        int id = mj_name2id(s->model, mjOBJ_JOINT, (pfx + name).c_str());
        if (id < 0) {
            LOG_ERROR(
              "joint '" << pfx << name
                        << "' not found in MuJoCo model  - check robot prefix or URDF joint names");
            int idx = (int)s->kdl_to_mj_qpos.size();
            s->kdl_to_mj_qpos.push_back(idx);
            s->kdl_to_mj_dof.push_back(idx);
        } else {
            s->kdl_to_mj_qpos.push_back(s->model->jnt_qposadr[id]);
            s->kdl_to_mj_dof.push_back(s->model->jnt_dofadr[id]);
        }
    }
}

/* Build KDL chain from compiled mjModel (no URDF needed) */

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
            LOG_ERROR("'" << tip_body << "' is not a descendant of '" << base_body
                          << "'  - check body hierarchy in the model");
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
          model->body_pos[3 * bid], model->body_pos[3 * bid + 1], model->body_pos[3 * bid + 2]);
        KDL::Frame F(bR, bv);

        KDL::Joint jnt(KDL::Joint::None);
        for (int jid = model->body_jntadr[bid];
             jid < model->body_jntadr[bid] + model->body_jntnum[bid];
             ++jid) {
            if (model->jnt_type[jid] != mjJNT_HINGE && model->jnt_type[jid] != mjJNT_SLIDE)
                continue;
            const char *jname = mj_id2name(model, mjOBJ_JOINT, jid);
            KDL::Vector jp(
              model->jnt_pos[3 * jid], model->jnt_pos[3 * jid + 1], model->jnt_pos[3 * jid + 2]);
            KDL::Vector ja(
              model->jnt_axis[3 * jid], model->jnt_axis[3 * jid + 1], model->jnt_axis[3 * jid + 2]);
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

/* GLFW */

static void    cb_keyboard(GLFWwindow *, int, int, int, int);
static void    cb_mouse_button(GLFWwindow *, int, int, int);
static void    cb_mouse_move(GLFWwindow *, double, double);
static void    cb_scroll(GLFWwindow *, double, double);
static Robot  *g_robot  = nullptr;
static Viewer *g_viewer = nullptr;

struct GLMouseState
{
    bool   btn_left = false, btn_right = false, btn_middle = false;
    double mouse_x = 0, mouse_y = 0;
    double last_click_time = -1.0;
    int    last_click_btn  = -1;
};

/* Public API */

void configure_spec(mjSpec *spec, const SceneSpec *sc)
{
    spec->option.timestep         = sc->timestep;
    spec->option.gravity[2]       = sc->gravity_z;
    spec->compiler.balanceinertia = 1;
    spec->compiler.discardvisual  = 0;
    if (sc->add_skybox) add_skybox_to_spec(spec);
    if (sc->add_floor) add_floor_to_spec(spec);
    if (sc->table.enabled) add_table_to_spec(spec, sc->table);
    if (!sc->objects.empty()) add_objects_to_spec(spec, sc->objects);
}

/* Compile spec into a model and data; spec is always deleted. */
bool compile_and_make_data(mjSpec *spec, mjModel **out_model, mjData **out_data)
{
    *out_model = mj_compile(spec, nullptr);
    if (!*out_model) {
        LOG_ERROR("mj_compile failed: " << mjs_getError(spec));
        mj_deleteSpec(spec);
        return false;
    }
    LOG_INFO("scene compiled: nq=" << (*out_model)->nq << " nv=" << (*out_model)->nv
                                   << " nbody=" << (*out_model)->nbody);
    mj_deleteSpec(spec);
    *out_data = mj_makeData(*out_model);
    if (!*out_data) {
        mj_deleteModel(*out_model);
        *out_model = nullptr;
        return false;
    }
    return true;
}

bool build_scene(mjModel **out_model, mjData **out_data, const SceneSpec *sc)
{
    if (!sc || sc->robots.empty()) return false;
    ensure_plugins_loaded();
    LOG_INFO("building scene: " << sc->robots.size() << " robot(s)"
                                << ", table=" << (sc->table.enabled ? "yes" : "no")
                                << ", objects=" << sc->objects.size());

    fs::path tmp = fs::absolute(fs::path(sc->robots[0].urdf_path).parent_path());

    /* Single robot: use spec API (avoids mj_saveLastXML which is unstable for
     * URDF-sourced models on some MuJoCo versions). */
    if (sc->robots.size() == 1) {
        const SceneRobot &r    = sc->robots[0];
        fs::path          proc = tmp / "_mj_kdl_r0_proc.urdf";
        if (!preprocess_urdf(r.urdf_path, proc.string())) return false;

        char    err[2048] = {};
        mjSpec *spec      = mj_parseXML(proc.c_str(), nullptr, err, sizeof(err));
        if (!spec) {
            LOG_ERROR("mj_parseXML failed for '" << proc.string() << "': " << err);
            return false;
        }

        spec->option.timestep         = sc->timestep;
        spec->option.gravity[2]       = sc->gravity_z;
        spec->compiler.balanceinertia = 1;
        spec->compiler.discardvisual  = 0;

        if (r.pos[0] || r.pos[1] || r.pos[2] || r.euler[0] || r.euler[1] || r.euler[2]) {
            mjsBody *wb   = mjs_findBody(spec, "world");
            mjsBody *root = wb ? mjs_asBody(mjs_firstChild(wb, mjOBJ_BODY, 0)) : nullptr;
            if (root) {
                root->pos[0] = r.pos[0];
                root->pos[1] = r.pos[1];
                root->pos[2] = r.pos[2];
                if (r.euler[0] || r.euler[1] || r.euler[2]) {
                    root->alt.type     = mjORIENTATION_EULER;
                    root->alt.euler[0] = r.euler[0];
                    root->alt.euler[1] = r.euler[1];
                    root->alt.euler[2] = r.euler[2];
                }
            }
        }
        configure_spec(spec, sc);
        return compile_and_make_data(spec, out_model, out_data);
    }

    // Multi-robot: URDF -> raw MJCF -> combine -> inject scene -> load.
    std::vector<std::string> raws;
    for (int i = 0; i < (int)sc->robots.size(); ++i) {
        fs::path proc = tmp / ("_mj_kdl_r" + std::to_string(i) + "_proc.urdf");
        fs::path raw  = tmp / ("_mj_kdl_r" + std::to_string(i) + "_raw.xml");
        if (!preprocess_urdf(sc->robots[i].urdf_path, proc.string())) return false;
        if (!urdf_to_raw_mjcf(proc.string(), raw.string())) return false;
        raws.push_back(raw.string());
    }
    fs::path combined = tmp / "_mj_kdl_combined_raw.xml";
    if (!combine_mjcf(raws, sc, combined.string())) return false;
    char    err[2048] = {};
    mjSpec *spec      = mj_parseXML(combined.c_str(), nullptr, err, sizeof(err));
    if (!spec) {
        LOG_ERROR("mj_parseXML failed for combined MJCF '" << combined.string() << "': " << err);
        return false;
    }
    configure_spec(spec, sc);
    return compile_and_make_data(spec, out_model, out_data);
}

bool load_mjcf(mjModel **out_model, mjData **out_data, const char *path)
{
    ensure_plugins_loaded();
    char err[2048] = {};
    LOG_INFO("loading MJCF: '" << path << "'");
    *out_model = mj_loadXML(path, nullptr, err, sizeof(err));
    if (!*out_model) {
        LOG_ERROR("mj_loadXML failed for '" << path << "': " << err);
        return false;
    }
    *out_data = mj_makeData(*out_model);
    if (!*out_data) {
        mj_deleteModel(*out_model);
        *out_model = nullptr;
        return false;
    }
    return true;
}

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

/* XML-level helpers shared by the patch_mjcf_* functions. */

static void xml_inject_skybox(tinyxml2::XMLDocument &doc, tinyxml2::XMLElement *root)
{
    using namespace tinyxml2;
    XMLElement *asset = xml_get_or_create(root, "asset", doc);
    XMLElement *wb    = xml_get_or_create(root, "worldbody", doc);

    XMLElement *sky = doc.NewElement("texture");
    sky->SetAttribute("name", "skybox");
    sky->SetAttribute("type", "skybox");
    sky->SetAttribute("builtin", "gradient");
    sky->SetAttribute("rgb1", "0.3 0.45 0.65");
    sky->SetAttribute("rgb2", "0.65 0.8 0.95");
    sky->SetAttribute("width", "200");
    sky->SetAttribute("height", "200");
    asset->InsertEndChild(sky);

    XMLElement *light = doc.NewElement("light");
    light->SetAttribute("directional", "true");
    light->SetAttribute("pos", "0 0 4");
    wb->InsertFirstChild(light);
}

static void xml_inject_floor(tinyxml2::XMLDocument &doc, tinyxml2::XMLElement *root)
{
    using namespace tinyxml2;
    XMLElement *asset = xml_get_or_create(root, "asset", doc);
    XMLElement *wb    = xml_get_or_create(root, "worldbody", doc);

    XMLElement *tex = doc.NewElement("texture");
    tex->SetAttribute("name", "groundplane");
    tex->SetAttribute("type", "2d");
    tex->SetAttribute("builtin", "checker");
    tex->SetAttribute("rgb1", "0.2 0.3 0.4");
    tex->SetAttribute("rgb2", "0.1 0.2 0.3");
    tex->SetAttribute("width", "300");
    tex->SetAttribute("height", "300");
    asset->InsertEndChild(tex);

    XMLElement *mat = doc.NewElement("material");
    mat->SetAttribute("name", "groundplane");
    mat->SetAttribute("texture", "groundplane");
    mat->SetAttribute("texrepeat", "5 5");
    mat->SetAttribute("reflectance", "0.2");
    asset->InsertEndChild(mat);

    XMLElement *floor = doc.NewElement("geom");
    floor->SetAttribute("name", "floor");
    floor->SetAttribute("type", "plane");
    floor->SetAttribute("size", "10 10 0.05");
    floor->SetAttribute("material", "groundplane");
    floor->SetAttribute("contype", "1");
    floor->SetAttribute("conaffinity", "1");
    floor->SetAttribute("condim", "3");
    wb->InsertFirstChild(floor);
}

static bool load_mjcf_doc(const char *path,
  tinyxml2::XMLDocument              &doc,
  tinyxml2::XMLElement              **root,
  const char                         *caller)
{
    if (doc.LoadFile(path) != tinyxml2::XML_SUCCESS) {
        LOG_ERROR(caller << ": failed to load '" << path << "'");
        return false;
    }
    *root = doc.FirstChildElement("mujoco");
    return *root != nullptr;
}

bool patch_mjcf_add_skybox(const char *mjcf_path)
{
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLElement *root = nullptr;
    if (!load_mjcf_doc(mjcf_path, doc, &root, "patch_mjcf_add_skybox")) return false;
    xml_inject_skybox(doc, root);
    return doc.SaveFile(mjcf_path) == tinyxml2::XML_SUCCESS;
}

bool patch_mjcf_add_floor(const char *mjcf_path)
{
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLElement *root = nullptr;
    if (!load_mjcf_doc(mjcf_path, doc, &root, "patch_mjcf_add_floor")) return false;
    xml_inject_floor(doc, root);
    return doc.SaveFile(mjcf_path) == tinyxml2::XML_SUCCESS;
}

bool patch_mjcf_add_objects(const char *mjcf_path, const std::vector<SceneObject> &objects)
{
    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(mjcf_path) != XML_SUCCESS) {
        LOG_ERROR("patch_mjcf_add_objects: failed to load '" << mjcf_path << "'");
        return false;
    }
    XMLElement *root = doc.FirstChildElement("mujoco");
    if (!root) return false;
    XMLElement *wb = xml_get_or_create(root, "worldbody", doc);

    for (const auto &obj : objects) {
        XMLElement *body = doc.NewElement("body");
        body->SetAttribute("name", obj.name.c_str());
        char pos_str[64];
        std::snprintf(
          pos_str, sizeof(pos_str), "%.6f %.6f %.6f", obj.pos[0], obj.pos[1], obj.pos[2]);
        body->SetAttribute("pos", pos_str);

        if (!obj.fixed) {
            XMLElement *fj = doc.NewElement("freejoint");
            fj->SetAttribute("name", (obj.name + "_joint").c_str());
            body->InsertEndChild(fj);
        }

        const char *shape_str = "box";
        if (obj.shape == ObjShape::SPHERE) shape_str = "sphere";
        if (obj.shape == ObjShape::CYLINDER) shape_str = "cylinder";

        char size_str[64];
        std::snprintf(
          size_str, sizeof(size_str), "%.6f %.6f %.6f", obj.size[0], obj.size[1], obj.size[2]);
        char rgba_str[64];
        std::snprintf(rgba_str,
          sizeof(rgba_str),
          "%.3f %.3f %.3f %.3f",
          obj.rgba[0],
          obj.rgba[1],
          obj.rgba[2],
          obj.rgba[3]);
        char fric_str[64];
        std::snprintf(fric_str,
          sizeof(fric_str),
          "%.4f %.4f %.4f",
          obj.friction[0],
          obj.friction[1],
          obj.friction[2]);

        XMLElement *geom = doc.NewElement("geom");
        geom->SetAttribute("name", (obj.name + "_geom").c_str());
        geom->SetAttribute("type", shape_str);
        geom->SetAttribute("size", size_str);
        geom->SetAttribute("mass", obj.mass);
        geom->SetAttribute("rgba", rgba_str);
        geom->SetAttribute("condim", obj.condim);
        geom->SetAttribute("friction", fric_str);
        geom->SetAttribute("contype", "1");
        geom->SetAttribute("conaffinity", "1");
        body->InsertEndChild(geom);

        wb->InsertEndChild(body);
    }
    return doc.SaveFile(mjcf_path) == XML_SUCCESS;
}

/* Recursive depth-first search for the first <body name="..."> matching name. */
static tinyxml2::XMLElement *find_body(tinyxml2::XMLElement *e, const char *name)
{
    if (!e || !name) return nullptr;
    if (e->Attribute("name") && std::string(e->Attribute("name")) == name) return e;
    for (auto *ch = e->FirstChildElement("body"); ch; ch = ch->NextSiblingElement("body"))
        if (auto *found = find_body(ch, name)) return found;
    return nullptr;
}

bool attach_gripper(const char *arm_mjcf, const GripperSpec *g, const char *out_path)
{
    using namespace tinyxml2;

    XMLDocument arm_doc, grp_doc;
    LOG_INFO("attaching gripper (prefix='" << std::string(g->prefix ? g->prefix : "") << "') from '"
                                           << g->mjcf_path << "' to body '" << g->attach_to << "'");
    if (arm_doc.LoadFile(arm_mjcf) != XML_SUCCESS) {
        LOG_ERROR("failed to load arm MJCF '" << arm_mjcf << "'");
        return false;
    }
    if (grp_doc.LoadFile(g->mjcf_path) != XML_SUCCESS) {
        LOG_ERROR("failed to load gripper MJCF '" << g->mjcf_path << "'");
        return false;
    }

    XMLElement *arm_mj = arm_doc.FirstChildElement("mujoco");
    XMLElement *grp_mj = grp_doc.FirstChildElement("mujoco");
    if (!arm_mj || !grp_mj) return false;

    // Resolve arm's meshdir to absolute so the combined file can live anywhere
    fs::path arm_dir = fs::absolute(fs::path(arm_mjcf).parent_path());
    if (XMLElement *ac = arm_mj->FirstChildElement("compiler")) {
        const char *mdir = ac->Attribute("meshdir");
        if (mdir && !fs::path(mdir).is_absolute())
            ac->SetAttribute("meshdir", (arm_dir / mdir).string().c_str());
    }

    // Resolve gripper asset paths to absolute (for meshdir)
    fs::path grp_dir = fs::absolute(fs::path(g->mjcf_path).parent_path());

    /* Fix gripper mesh file paths to absolute, and add explicit names (stem of filename)
     * so that xml_prefix_names can rename them consistently. */
    auto fix_meshdir = [&](XMLElement *root, const std::string &meshdir) {
        if (XMLElement *asset = root->FirstChildElement("asset")) {
            for (auto *m = asset->FirstChildElement("mesh"); m; m = m->NextSiblingElement("mesh")) {
                const char *f = m->Attribute("file");
                if (f) {
                    if (!fs::path(f).is_absolute())
                        m->SetAttribute("file", (fs::path(meshdir) / f).string().c_str());
                    if (!m->Attribute("name"))
                        m->SetAttribute("name", fs::path(f).stem().string().c_str());
                }
            }
        }
    };
    std::string grp_meshdir = grp_dir.string() + "/assets";
    if (XMLElement *gc = grp_mj->FirstChildElement("compiler")) {
        const char *mdir = gc->Attribute("meshdir");
        if (mdir) grp_meshdir = (grp_dir / mdir).string();
    }
    fix_meshdir(grp_mj, grp_meshdir);

    std::string pfx(g->prefix ? g->prefix : "");

    // Merge assets
    XMLElement *arm_asset = xml_get_or_create(arm_mj, "asset", arm_doc);
    if (XMLElement *grp_asset = grp_mj->FirstChildElement("asset"))
        for (auto *c = grp_asset->FirstChildElement(); c; c = c->NextSiblingElement())
            arm_asset->InsertEndChild(xml_clone_prefixed(c, &arm_doc, pfx));

    // Merge defaults
    if (XMLElement *gd = grp_mj->FirstChildElement("default")) {
        XMLElement *ad = xml_get_or_create(arm_mj, "default", arm_doc);
        for (auto *c = gd->FirstChildElement(); c; c = c->NextSiblingElement())
            ad->InsertEndChild(xml_clone_prefixed(c, &arm_doc, pfx));
    }

    // Find attach body in arm worldbody and insert gripper worldbody children
    XMLElement *arm_wb = arm_mj->FirstChildElement("worldbody");
    if (!arm_wb) return false;
    XMLElement *attach_el = find_body(arm_wb, g->attach_to);
    if (!attach_el) {
        LOG_ERROR("attach body '"
                  << g->attach_to
                  << "' not found in arm worldbody  - check arm MJCF for correct body name");
        return false;
    }

    XMLElement *grp_wb = grp_mj->FirstChildElement("worldbody");
    if (grp_wb) {
        for (auto *c = grp_wb->FirstChildElement(); c; c = c->NextSiblingElement()) {
            XMLElement *cl = xml_clone_prefixed(c, &arm_doc, pfx);
            // Apply offset
            char pos_str[64];
            std::snprintf(
              pos_str, sizeof(pos_str), "%.6f %.6f %.6f", g->pos[0], g->pos[1], g->pos[2]);
            if (g->pos[0] || g->pos[1] || g->pos[2]) cl->SetAttribute("pos", pos_str);
            bool non_identity = g->quat[0] != 1 || g->quat[1] || g->quat[2] || g->quat[3];
            if (non_identity) {
                char q_str[64];
                std::snprintf(q_str,
                  sizeof(q_str),
                  "%.6f %.6f %.6f %.6f",
                  g->quat[0],
                  g->quat[1],
                  g->quat[2],
                  g->quat[3]);
                cl->SetAttribute("quat", q_str);
            }
            attach_el->InsertEndChild(cl);
        }
    }

    // Merge contacts
    auto merge_section = [&](const char *tag) {
        XMLElement *gs = grp_mj->FirstChildElement(tag);
        if (!gs) return;
        XMLElement *as = xml_get_or_create(arm_mj, tag, arm_doc);
        for (auto *c = gs->FirstChildElement(); c; c = c->NextSiblingElement())
            as->InsertEndChild(xml_clone_prefixed(c, &arm_doc, pfx));
    };
    merge_section("contact");
    merge_section("tendon");
    /* Equality constraints are copied verbatim: MuJoCo computes local body
     * offsets from the anchor at compile time, so anchor="0 0 0" is correct
     * regardless of where the gripper is attached. */
    merge_section("equality");
    merge_section("actuator");

    return arm_doc.SaveFile(out_path) == XML_SUCCESS;
}

void destroy_scene(mjModel *model, mjData *data)
{
    if (data) mj_deleteData(data);
    if (model) mj_deleteModel(model);
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
    SceneObject removed = *it;
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

/* Recursively prefix all named elements in an MJCF subtree.
 * Asset entries (mesh, material, texture, hfield) share names across arms
 * and are left unchanged.  All other name/joint/body1/body2/site attributes
 * receive pfx prepended so multiple robot instances do not conflict. */
static void mjcf_prefix_names(tinyxml2::XMLElement *e, const std::string &pfx)
{
    if (!e) return;
    const char *tag      = e->Name();
    bool        is_asset = tag
                    && (std::strcmp(tag, "mesh") == 0 || std::strcmp(tag, "texture") == 0
                        || std::strcmp(tag, "material") == 0 || std::strcmp(tag, "hfield") == 0);
    if (!is_asset)
        if (const char *v = e->Attribute("name")) e->SetAttribute("name", (pfx + v).c_str());
    for (const char *a : { "joint", "body1", "body2", "site" })
        if (const char *v = e->Attribute(a)) e->SetAttribute(a, (pfx + v).c_str());
    for (auto *c = e->FirstChildElement(); c; c = c->NextSiblingElement())
        mjcf_prefix_names(c, pfx);
}

bool build_scene_from_mjcfs(const char *out_mjcf,
  const MjcfArmSpec                    *arms,
  int                                   n_arms,
  bool                                  add_floor,
  bool                                  add_skybox)
{
    using namespace tinyxml2;
    if (!out_mjcf || !arms || n_arms <= 0) return false;

    XMLDocument doc;
    auto       *mj = doc.NewElement("mujoco");
    mj->SetAttribute("model", "scene");
    doc.InsertFirstChild(mj);

    {
        auto *opt = doc.NewElement("option");
        opt->SetAttribute("timestep", "0.002");
        opt->SetAttribute("gravity", "0 0 -9.81");
        mj->InsertEndChild(opt);
    }

    auto *asset     = doc.NewElement("asset");
    auto *worldbody = doc.NewElement("worldbody");
    auto *actuators = doc.NewElement("actuator");
    auto *contact   = doc.NewElement("contact");
    mj->InsertEndChild(asset);
    mj->InsertEndChild(worldbody);
    mj->InsertEndChild(actuators);
    mj->InsertEndChild(contact);

    for (int ai = 0; ai < n_arms; ++ai) {
        const MjcfArmSpec &spec = arms[ai];
        if (!spec.mjcf_path) return false;

        XMLDocument src;
        if (src.LoadFile(spec.mjcf_path) != XML_SUCCESS) {
            LOG_ERROR("build_scene_from_mjcfs: failed to load '" << spec.mjcf_path << "'");
            return false;
        }
        auto *src_root = src.FirstChildElement("mujoco");
        if (!src_root) return false;

        const std::string pfx = spec.prefix ? spec.prefix : "";

        /* Copy assets from the first arm only; all arms share the same meshes. */
        if (ai == 0) {
            auto *sa = src_root->FirstChildElement("asset");
            if (sa)
                for (auto *c = sa->FirstChildElement(); c; c = c->NextSiblingElement())
                    asset->InsertEndChild(c->DeepClone(&doc));
        }

        /* Placement wrapper body */
        char pos_s[64];
        std::snprintf(
          pos_s, sizeof(pos_s), "%.6g %.6g %.6g", spec.pos[0], spec.pos[1], spec.pos[2]);
        auto *wrap = doc.NewElement("body");
        wrap->SetAttribute("pos", pos_s);
        if (std::abs(spec.euler_z) > 0.01) {
            char euler_s[32];
            std::snprintf(euler_s, sizeof(euler_s), "0 0 %.6g", spec.euler_z);
            wrap->SetAttribute("euler", euler_s);
        }
        worldbody->InsertEndChild(wrap);

        auto *src_wb = src_root->FirstChildElement("worldbody");
        if (src_wb)
            for (auto *c = src_wb->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) mjcf_prefix_names(copy, pfx);
                wrap->InsertEndChild(copy);
            }

        auto *src_act = src_root->FirstChildElement("actuator");
        if (src_act)
            for (auto *c = src_act->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) mjcf_prefix_names(copy, pfx);
                actuators->InsertEndChild(copy);
            }

        auto *src_ct = src_root->FirstChildElement("contact");
        if (src_ct)
            for (auto *c = src_ct->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) mjcf_prefix_names(copy, pfx);
                contact->InsertEndChild(copy);
            }
    }

    if (doc.SaveFile(out_mjcf) != XML_SUCCESS) return false;
    if (add_skybox && !patch_mjcf_add_skybox(out_mjcf)) return false;
    if (add_floor && !patch_mjcf_add_floor(out_mjcf)) return false;
    return true;
}

bool init_robot(Robot *s,
  mjModel             *model,
  mjData              *data,
  const char          *urdf,
  const char          *base_link,
  const char          *tip_link,
  const char          *prefix)
{
    LOG_INFO("init_robot: '" << base_link << "' -> '" << tip_link << "' prefix='"
                             << (prefix ? prefix : "") << "' urdf='" << urdf << "'");
    s->model       = model;
    s->data        = data;
    s->_owns_model = false;
    if (!load_kdl_chain(s, urdf, base_link, tip_link)) return false;
    std::string pfx = prefix ? prefix : "";
    sync_chain_inertias(s, pfx);
    build_index_map(s, pfx);
    LOG_INFO(
      "chain ready: " << s->n_joints << " joints [" << base_link << " -> " << tip_link << "]");
    return true;
}

bool init_from_mjcf(Robot *s,
  mjModel                 *model,
  mjData                  *data,
  const char              *base_body,
  const char              *tip_body,
  const char              *prefix)
{
    LOG_INFO("init_from_mjcf: '" << base_body << "' -> '" << tip_body << "' prefix='"
                                 << (prefix ? prefix : "") << "'");
    s->model       = model;
    s->data        = data;
    s->_owns_model = false;
    if (!build_kdl_from_model(s, model, base_body, tip_body)) return false;
    build_index_map(s, prefix ? prefix : "");
    LOG_INFO(
      "chain ready: " << s->n_joints << " joints [" << base_body << " -> " << tip_body << "]");
    return true;
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
        glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_ANY);
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
    glfwSetWindowCloseCallback(
      v->window, [](GLFWwindow *w) { glfwSetWindowShouldClose(w, GLFW_TRUE); });
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

void cleanup(Viewer *v)
{
    if (!v->window) return;
    mjv_freeScene(&v->scn);
    mjr_freeContext(&v->con);
    delete static_cast<GLMouseState *>(glfwGetWindowUserPointer(v->window));
    glfwDestroyWindow(v->window);
    v->window = nullptr;
    glfwTerminate();
    if (g_viewer == v) g_viewer = nullptr;
}

void cleanup(Robot *s)
{
    if (s->_owns_model) {
        if (s->data) {
            mj_deleteData(s->data);
            s->data = nullptr;
        }
        if (s->model) {
            mj_deleteModel(s->model);
            s->model = nullptr;
        }
    } else {
        s->data  = nullptr;
        s->model = nullptr;
    }
    if (g_robot == s) g_robot = nullptr;
}

void step(Robot *s)
{
    if (!s->model || !s->data || s->paused) return;
    if (g_viewer && g_viewer->pert.active)
        mjv_applyPerturbForce(s->model, s->data, &g_viewer->pert);
    mj_step(s->model, s->data);
}

void step_n(Robot *s, int n)
{
    for (int i = 0; i < n; ++i) step(s);
}

void reset(Robot *s)
{
    if (s->model && s->data) {
        mj_resetData(s->model, s->data);
        mj_forward(s->model, s->data);
    }
}

bool is_running(const Viewer *v)
{
    if (!v->window) return false;
    return !glfwWindowShouldClose(v->window);
}

bool render(Viewer *v, const Robot *r)
{
    if (!v->window) return false;
    if (glfwWindowShouldClose(v->window)) return false;
    glfwPollEvents();
    int w, h;
    glfwGetFramebufferSize(v->window, &w, &h);
    mjrRect vp = { 0, 0, w, h };
    mjv_updateScene(r->model, r->data, &v->opt, &v->pert, &v->cam, mjCAT_ALL, &v->scn);
    mjr_render(vp, &v->scn, &v->con);

    char        top[256];
    const char *selname =
      (v->pert.select > 0) ? mj_id2name(r->model, mjOBJ_BODY, v->pert.select) : nullptr;
    if (selname)
        std::snprintf(top,
          sizeof(top),
          "t = %.3f s%s\nSelected: %s",
          r->data->time,
          r->paused ? "  [PAUSED]" : "",
          selname);
    else
        std::snprintf(
          top, sizeof(top), "t = %.3f s%s", r->data->time, r->paused ? "  [PAUSED]" : "");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, top, nullptr, &v->con);

    mjr_overlay(mjFONT_NORMAL,
      mjGRID_BOTTOMLEFT,
      vp,
      "DblClick: select body   D: deselect\n"
      "Left drag: push force   Right drag: apply torque\n"
      "No selection  - Left drag: orbit   Right drag: pan   Scroll: zoom\n"
      "Space: pause/resume   R: reset   J: toggle joints   Q/Esc: quit",
      nullptr,
      &v->con);

    if (v->show_joints && r->n_joints > 0 && !r->kdl_to_mj_qpos.empty()) {
        char jvals[1024];
        int  off = std::snprintf(jvals, sizeof(jvals), "Joints (rad)\n");
        for (int i = 0; i < r->n_joints && i < 16 && off < (int)sizeof(jvals) - 32; ++i)
            off += std::snprintf(jvals + off,
              sizeof(jvals) - off,
              "  %-20s %.3f\n",
              r->joint_names[i].c_str(),
              r->data->qpos[r->kdl_to_mj_qpos[i]]);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPRIGHT, vp, jvals, nullptr, &v->con);
    }

    glfwSwapBuffers(v->window);
    return true;
}

bool sync_to_kdl(const Robot *s, KDL::JntArray &q)
{
    if (!s->data) return false;
    q.resize(s->n_joints);
    for (int i = 0; i < s->n_joints; ++i) q(i) = s->data->qpos[s->kdl_to_mj_qpos[i]];
    return true;
}

void sync_from_kdl(Robot *s, const KDL::JntArray &q)
{
    if (!s->data) return;
    int n = std::min((int)q.rows(), s->n_joints);
    for (int i = 0; i < n; ++i) s->data->qpos[s->kdl_to_mj_qpos[i]] = q(i);
}

void set_torques(Robot *s, const KDL::JntArray &tau)
{
    if (!s->data) return;
    int n = std::min((int)tau.rows(), s->n_joints);
    for (int i = 0; i < n; ++i) s->data->qfrc_applied[s->kdl_to_mj_dof[i]] = tau(i);
}

/* GLFW callbacks */

static void cb_keyboard(GLFWwindow *w, int key, int, int action, int)
{
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (!g_robot || !g_viewer) return;
    if (key == GLFW_KEY_SPACE) g_robot->paused = !g_robot->paused;
    if (key == GLFW_KEY_R) reset(g_robot);
    if (key == GLFW_KEY_J) g_viewer->show_joints = !g_viewer->show_joints;
    if (key == GLFW_KEY_D) {
        g_viewer->pert.select = 0;
        g_viewer->pert.active = 0;
    }
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
            int    body = mjv_select(g_robot->model,
              g_robot->data,
              &g_viewer->opt,
              (mjtNum)wh / ww,
              (mjtNum)ms->mouse_x / ww,
              (mjtNum)(wh - ms->mouse_y) / wh,
              &g_viewer->scn,
              selpnt,
              geomid,
              flexid,
              skinid);
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
    bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS
                  || glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
    if (g_viewer->pert.select > 0 && g_viewer->pert.active) {
        // Left drag = MOVE (translate body), Right drag = ROTATE (torque body)
        mjtMouse act = ms->btn_left    ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                       : ms->btn_right ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                                       : mjMOUSE_ZOOM;
        mjv_movePerturb(
          g_robot->model, g_robot->data, act, dx / wh, dy / wh, &g_viewer->scn, &g_viewer->pert);
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
          g_robot->model, mjMOUSE_ZOOM, 0, -0.05 * yoff, &g_viewer->scn, &g_viewer->cam);
}

namespace mj = ::mujoco;

using Seconds = std::chrono::duration<double>;

static constexpr double kSyncMisalign       = 0.1; // max misalign before re-sync (sim seconds)
static constexpr double kSimRefreshFraction = 0.7; // fraction of refresh budget for physics

void run_simulate_ui(mjModel *m, mjData *d, const char *path, ControlCb physics_cb)
{
    apply_glfw_platform_hints();

    mjvCamera cam;
    mjv_defaultCamera(&cam);
    mjvOption opt;
    mjv_defaultOption(&opt);
    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    auto sim = std::make_unique<mj::Simulate>(
      std::make_unique<mj::GlfwAdapter>(), &cam, &opt, &pert, /*is_passive=*/false);
    sim->font = 1; // 100% font scale (0=50%, 1=100%, 2=150%, ...)

    /* Physics thread: real-time sync loop (mirrors PhysicsLoop from main.cc).
     * Load must happen from this thread so that LoadOnRenderThread() on the
     * render thread can acknowledge via cond_loadrequest. */
    std::thread phys([&]() {
        sim->LoadMessage(path);
        sim->Load(m, d, path);
        {
            std::unique_lock<std::recursive_mutex> lock(sim->mtx);
            mj_forward(m, d);
        }

        std::chrono::time_point<mj::Simulate::Clock> syncCPU;
        mjtNum                                       syncSim = 0;

        while (!sim->exitrequest.load()) {
            if (sim->run && sim->busywait)
                std::this_thread::yield();
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            std::unique_lock<std::recursive_mutex> lock(sim->mtx);

            if (sim->run) {
                const auto startCPU   = mj::Simulate::Clock::now();
                const auto elapsedCPU = startCPU - syncCPU;
                double     elapsedSim = d->time - syncSim;
                double     slowdown   = 100.0 / sim->percentRealTime[sim->real_time_index];
                bool       misaligned =
                  std::abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > kSyncMisalign;

                if (elapsedSim < 0 || elapsedCPU.count() < 0
                    || syncCPU.time_since_epoch().count() == 0 || misaligned
                    || sim->speed_changed) {
                    // Out-of-sync: resync clocks, take one step.
                    syncCPU            = startCPU;
                    syncSim            = d->time;
                    sim->speed_changed = false;
                    if (physics_cb) physics_cb(m, d);
                    if (sim->pert.active) mjv_applyPerturbForce(m, d, &sim->pert);
                    mj_step(m, d);
                    sim->AddToHistory();
                } else {
                    // In-sync: step until simulation is ahead of CPU or budget exhausted.
                    double refreshTime = kSimRefreshFraction / sim->refresh_rate;
                    mjtNum prevSim     = d->time;
                    while (
                      Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU
                      && mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
                        if (physics_cb) physics_cb(m, d);
                        if (sim->pert.active) mjv_applyPerturbForce(m, d, &sim->pert);
                        mj_step(m, d);
                        sim->AddToHistory();
                        if (d->time < prevSim) break; // guard against time reset
                        prevSim = d->time;
                    }
                }
            } else {
                // Paused: keep rendering up to date.
                mj_forward(m, d);
                sim->speed_changed = true;
            }
        }
    });

    sim->RenderLoop(); // blocks on main thread until window closes
    phys.join();
}

} // namespace mj_kdl
