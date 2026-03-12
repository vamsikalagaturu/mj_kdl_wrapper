#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "kdl_parser/kdl_parser.hpp"
#include "urdf_model/joint.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

#include <tinyxml2.h>

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
namespace mj_kdl {

// URDF preprocessing

static bool preprocess_urdf(const std::string& in, const std::string& out)
{
    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(in.c_str()) != XML_SUCCESS) return false;
    XMLElement* robot = doc.FirstChildElement("robot");
    if (!robot) return false;

    XMLElement* mj = robot->FirstChildElement("mujoco");
    if (!mj) { mj = doc.NewElement("mujoco"); robot->InsertFirstChild(mj); }
    XMLElement* cmp = mj->FirstChildElement("compiler");
    if (!cmp)  { cmp = doc.NewElement("compiler"); mj->InsertFirstChild(cmp); }
    cmp->SetAttribute("balanceinertia", "true");
    cmp->SetAttribute("meshdir",
        fs::absolute(fs::path(in).parent_path() / "meshes").string().c_str());

    for (XMLElement* link = robot->FirstChildElement("link"); link;
         link = link->NextSiblingElement("link")) {
        for (const char* vis_tag : {"visual", "collision"}) {
            for (XMLElement* vis = link->FirstChildElement(vis_tag); vis;
                 vis = vis->NextSiblingElement(vis_tag)) {
                XMLElement* geom = vis->FirstChildElement("geometry");
                if (!geom) continue;
                XMLElement* mesh = geom->FirstChildElement("mesh");
                if (!mesh) continue;
                const char* fn = mesh->Attribute("filename");
                if (fn && std::string(fn).rfind("package://", 0) == 0)
                    mesh->SetAttribute("filename", fs::path(fn).filename().string().c_str());
            }
        }
    }
    return doc.SaveFile(out.c_str()) == XML_SUCCESS;
}

// Spec-API helpers (single-robot path)

static void add_floor_to_spec(mjSpec* spec)
{
    mjsBody* wb = mjs_findBody(spec, "world");

    // Skybox gradient texture
    mjsTexture* sky = mjs_addTexture(spec);
    mjs_setString(mjs_getName(sky->element), "skybox");
    sky->type = mjTEXTURE_SKYBOX; sky->builtin = mjBUILTIN_GRADIENT;
    sky->rgb1[0]=0.3f; sky->rgb1[1]=0.45f; sky->rgb1[2]=0.65f;  // top: mid blue
    sky->rgb2[0]=0.65f; sky->rgb2[1]=0.80f; sky->rgb2[2]=0.95f; // bottom: pale blue
    sky->width=200; sky->height=200;

    mjsTexture* tex = mjs_addTexture(spec);
    mjs_setString(mjs_getName(tex->element), "groundplane");
    tex->type = mjTEXTURE_2D; tex->builtin = mjBUILTIN_CHECKER;
    tex->rgb1[0]=0.2; tex->rgb1[1]=0.3; tex->rgb1[2]=0.4;
    tex->rgb2[0]=0.1; tex->rgb2[1]=0.2; tex->rgb2[2]=0.3;
    tex->width=300; tex->height=300;

    mjsMaterial* mat = mjs_addMaterial(spec, nullptr);
    mjs_setString(mjs_getName(mat->element), "groundplane");
    // Set texture at slot mjTEXROLE_RGB (1); vector is pre-initialised with 10 empty strings
    mjs_setInStringVec(mat->textures, mjTEXROLE_RGB, "groundplane");
    mat->texrepeat[0]=5; mat->texrepeat[1]=5; mat->reflectance=0.2f;

    mjsGeom* floor = mjs_addGeom(wb, nullptr);
    mjs_setString(mjs_getName(floor->element), "floor");
    mjs_setString(floor->material, "groundplane");
    floor->type=mjGEOM_PLANE;
    floor->size[0]=10; floor->size[1]=10; floor->size[2]=0.05;
    floor->contype=1; floor->conaffinity=1; floor->condim=3;
}

static void add_table_to_spec(mjSpec* spec, const TableSpec& t)
{
    mjsBody* wb = mjs_findBody(spec, "world");
    double sz        = t.pos[2];          // surface z
    double half_thick = t.thickness * 0.5;
    double top_cz    = sz - half_thick;   // tabletop centre in world
    double leg_h     = sz - t.thickness;  // leg height (floor→bottom of top)

    mjsBody* tb = mjs_addBody(wb, nullptr);
    mjs_setString(mjs_getName(tb->element), "table");
    tb->pos[0] = t.pos[0];
    tb->pos[1] = t.pos[1];
    tb->pos[2] = top_cz;

    // tabletop
    mjsGeom* top = mjs_addGeom(tb, nullptr);
    mjs_setString(mjs_getName(top->element), "table_top");
    top->type = mjGEOM_BOX;
    top->size[0] = t.top_size[0];
    top->size[1] = t.top_size[1];
    top->size[2] = half_thick;
    for (int k = 0; k < 4; ++k) top->rgba[k] = t.rgba[k];
    top->contype = 1; top->conaffinity = 1; top->condim = 3;

    // 4 legs (only if there's room)
    if (leg_h > 0.0) {
        double half_leg = leg_h * 0.5;
        double leg_rel_z = -(sz * 0.5); // relative to body = -half_thick - half_leg
        double lx = t.top_size[0] - t.leg_radius;
        double ly = t.top_size[1] - t.leg_radius;
        const double corners[4][2] = {{lx, ly}, {-lx, ly}, {lx, -ly}, {-lx, -ly}};
        for (int i = 0; i < 4; ++i) {
            mjsGeom* leg = mjs_addGeom(tb, nullptr);
            char nm[32]; std::snprintf(nm, sizeof(nm), "table_leg%d", i);
            mjs_setString(mjs_getName(leg->element), nm);
            leg->type    = mjGEOM_CYLINDER;
            leg->size[0] = t.leg_radius;
            leg->size[1] = half_leg;
            leg->pos[0]  = corners[i][0];
            leg->pos[1]  = corners[i][1];
            leg->pos[2]  = leg_rel_z;
            for (int k = 0; k < 4; ++k) leg->rgba[k] = t.rgba[k];
            leg->contype = 1; leg->conaffinity = 1; leg->condim = 3;
        }
    }
}

static void add_objects_to_spec(mjSpec* spec, const std::vector<SceneObject>& objects)
{
    mjsBody* wb = mjs_findBody(spec, "world");
    for (const auto& obj : objects) {
        mjsBody* ob = mjs_addBody(wb, nullptr);
        mjs_setString(mjs_getName(ob->element), obj.name.c_str());
        ob->pos[0] = obj.pos[0];
        ob->pos[1] = obj.pos[1];
        ob->pos[2] = obj.pos[2];

        if (!obj.fixed) {
            mjsJoint* fj = mjs_addJoint(ob, nullptr);
            mjs_setString(mjs_getName(fj->element), (obj.name + "_joint").c_str());
            fj->type = mjJNT_FREE;
        }

        mjsGeom* g = mjs_addGeom(ob, nullptr);
        mjs_setString(mjs_getName(g->element), (obj.name + "_geom").c_str());
        switch (obj.shape) {
            case ObjShape::BOX:      g->type = mjGEOM_BOX;      break;
            case ObjShape::SPHERE:   g->type = mjGEOM_SPHERE;   break;
            case ObjShape::CYLINDER: g->type = mjGEOM_CYLINDER; break;
        }
        g->size[0] = obj.size[0];
        g->size[1] = obj.size[1];
        g->size[2] = obj.size[2];
        for (int k = 0; k < 4; ++k) g->rgba[k] = obj.rgba[k];
        g->contype = 1; g->conaffinity = 1; g->condim = 3;
    }
}

// Multi-robot XML helpers (N > 1 path)

static tinyxml2::XMLElement* xml_deep_clone(const tinyxml2::XMLElement* src,
                                             tinyxml2::XMLDocument* dst)
{
    tinyxml2::XMLElement* e = dst->NewElement(src->Name());
    for (const tinyxml2::XMLAttribute* a = src->FirstAttribute(); a; a = a->Next())
        e->SetAttribute(a->Name(), a->Value());
    for (const tinyxml2::XMLNode* c = src->FirstChild(); c; c = c->NextSibling())
        if (const tinyxml2::XMLElement* ce = c->ToElement())
            e->InsertEndChild(xml_deep_clone(ce, dst));
    return e;
}

static void xml_prefix_names(tinyxml2::XMLElement* e, const std::string& pfx)
{
    if (pfx.empty()) return;
    auto pfx_attr = [&](const char* a) {
        const char* v = e->Attribute(a);
        if (v) e->SetAttribute(a, (pfx + v).c_str());
    };
    pfx_attr("name");
    const char* tag = e->Name();
    if (!std::strcmp(tag, "geom") || !std::strcmp(tag, "site")) {
        pfx_attr("mesh"); pfx_attr("material");
    } else if (!std::strcmp(tag, "material")) {
        pfx_attr("texture");
    }
    for (auto* c = e->FirstChildElement(); c; c = c->NextSiblingElement())
        xml_prefix_names(c, pfx);
}

static bool urdf_to_raw_mjcf(const std::string& proc, const std::string& raw)
{
    char err[2048] = {};
    mjModel* tmp = mj_loadXML(proc.c_str(), nullptr, err, sizeof(err));
    if (!tmp) { std::cerr << "[mj_kdl] mj_loadXML: " << err << "\n"; return false; }
    int r = mj_saveLastXML(raw.c_str(), tmp, err, sizeof(err));
    mj_deleteModel(tmp);
    if (!r) { std::cerr << "[mj_kdl] mj_saveLastXML: " << err << "\n"; return false; }
    return true;
}

static bool inject_scene(const std::string& in, const std::string& out,
                          double timestep, double gravity_z, bool floor)
{
    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(in.c_str()) != XML_SUCCESS) return false;
    XMLElement* mj = doc.FirstChildElement("mujoco");
    if (!mj) return false;

    XMLElement* cmp = mj->FirstChildElement("compiler");
    if (!cmp) { cmp = doc.NewElement("compiler"); mj->InsertFirstChild(cmp); }
    cmp->SetAttribute("balanceinertia", "true");
    cmp->SetAttribute("discardvisual",  "false");

    XMLElement* opt = mj->FirstChildElement("option");
    if (!opt) { opt = doc.NewElement("option"); mj->InsertFirstChild(opt); }
    opt->SetAttribute("timestep", timestep);
    { char g[64]; std::snprintf(g, sizeof(g), "0 0 %.4f", gravity_z);
      opt->SetAttribute("gravity", g); }

    if (floor) {
        XMLElement* asset = mj->FirstChildElement("asset");
        if (!asset) { asset = doc.NewElement("asset"); mj->InsertAfterChild(opt, asset); }
        XMLElement* sky = doc.NewElement("texture");
        sky->SetAttribute("type","skybox"); sky->SetAttribute("builtin","gradient");
        sky->SetAttribute("rgb1","0.3 0.45 0.65"); sky->SetAttribute("rgb2","0.65 0.8 0.95");
        sky->SetAttribute("width","200"); sky->SetAttribute("height","200");
        asset->InsertEndChild(sky);
        XMLElement* t = doc.NewElement("texture");
        t->SetAttribute("type","2d"); t->SetAttribute("name","groundplane");
        t->SetAttribute("builtin","checker");
        t->SetAttribute("rgb1","0.2 0.3 0.4"); t->SetAttribute("rgb2","0.1 0.2 0.3");
        t->SetAttribute("width","300"); t->SetAttribute("height","300");
        asset->InsertEndChild(t);
        XMLElement* m = doc.NewElement("material");
        m->SetAttribute("name","groundplane"); m->SetAttribute("texture","groundplane");
        m->SetAttribute("texrepeat","5 5"); m->SetAttribute("reflectance","0.2");
        asset->InsertEndChild(m);
        XMLElement* wb = mj->FirstChildElement("worldbody");
        if (!wb) { wb = doc.NewElement("worldbody"); mj->InsertEndChild(wb); }
        XMLElement* f = doc.NewElement("geom");
        f->SetAttribute("name","floor"); f->SetAttribute("type","plane");
        f->SetAttribute("material","groundplane"); f->SetAttribute("size","10 10 0.05");
        f->SetAttribute("condim","3");
        wb->InsertFirstChild(f);
    }
    return doc.SaveFile(out.c_str()) == XML_SUCCESS;
}

static bool combine_mjcf(const std::vector<std::string>& raws,
                          const SceneSpec* sc, const std::string& out)
{
    using namespace tinyxml2;
    XMLDocument base;
    if (base.LoadFile(raws[0].c_str()) != XML_SUCCESS) {
        std::cerr << "[mj_kdl] cannot load " << raws[0] << "\n"; return false;
    }
    XMLElement* bmj = base.FirstChildElement("mujoco");
    if (!bmj) return false;
    XMLElement* bwb = bmj->FirstChildElement("worldbody");
    if (!bwb) { bwb = base.NewElement("worldbody"); bmj->InsertEndChild(bwb); }
    XMLElement* bas = bmj->FirstChildElement("asset");
    if (!bas)  { bas = base.NewElement("asset");     bmj->InsertEndChild(bas); }

    auto wrap_robot = [&](XMLElement* parent, std::vector<XMLElement*> kids,
                          const SceneRobot& r, int idx) {
        char name[128]; std::snprintf(name, sizeof(name), "%srobot_%d_base", r.prefix, idx);
        XMLElement* w = base.NewElement("body");
        w->SetAttribute("name", name);
        char pos[64]; std::snprintf(pos, sizeof(pos), "%.4f %.4f %.4f", r.pos[0], r.pos[1], r.pos[2]);
        w->SetAttribute("pos", pos);
        if (r.euler[0] || r.euler[1] || r.euler[2]) {
            char eu[64]; std::snprintf(eu, sizeof(eu), "%.4f %.4f %.4f", r.euler[0], r.euler[1], r.euler[2]);
            w->SetAttribute("euler", eu);
        }
        for (auto* k : kids) w->InsertEndChild(k);
        if (std::strlen(r.prefix) > 0)
            for (auto* k = w->FirstChildElement(); k; k = k->NextSiblingElement())
                xml_prefix_names(k, r.prefix);
        parent->InsertEndChild(w);
    };

    // robot 0: rewrap existing worldbody children
    {
        std::vector<XMLElement*> kids;
        for (auto* c = bwb->FirstChildElement(); c; c = c->NextSiblingElement())
            kids.push_back(xml_deep_clone(c, &base));
        while (bwb->FirstChild()) bwb->DeleteChild(bwb->FirstChild());
        if (std::strlen(sc->robots[0].prefix) > 0)
            for (auto* a = bas->FirstChildElement(); a; a = a->NextSiblingElement())
                xml_prefix_names(a, sc->robots[0].prefix);
        wrap_robot(bwb, kids, sc->robots[0], 0);
    }

    for (int i = 1; i < (int)raws.size(); ++i) {
        XMLDocument di;
        if (di.LoadFile(raws[i].c_str()) != XML_SUCCESS) {
            std::cerr << "[mj_kdl] cannot load " << raws[i] << "\n"; return false;
        }
        XMLElement* mi = di.FirstChildElement("mujoco");
        if (!mi) return false;
        const std::string pfx(sc->robots[i].prefix);
        if (XMLElement* ai = mi->FirstChildElement("asset"))
            for (auto* a = ai->FirstChildElement(); a; a = a->NextSiblingElement()) {
                XMLElement* cl = xml_deep_clone(a, &base);
                xml_prefix_names(cl, pfx);
                bas->InsertEndChild(cl);
            }
        XMLElement* wi = mi->FirstChildElement("worldbody");
        if (!wi) continue;
        std::vector<XMLElement*> kids;
        for (auto* c = wi->FirstChildElement(); c; c = c->NextSiblingElement())
            kids.push_back(xml_deep_clone(c, &base));
        wrap_robot(bwb, kids, sc->robots[i], i);
    }
    return base.SaveFile(out.c_str()) == XML_SUCCESS;
}

// Injects table and objects into an existing MJCF file (multi-robot XML path).
static bool inject_extras_xml(const std::string& path, const SceneSpec* sc)
{
    if (!sc->table.enabled && sc->objects.empty()) return true;

    using namespace tinyxml2;
    XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != XML_SUCCESS) return false;
    XMLElement* mj = doc.FirstChildElement("mujoco");
    if (!mj) return false;
    XMLElement* wb = mj->FirstChildElement("worldbody");
    if (!wb) return false;

    // Helpers
    auto set_rgba = [](XMLElement* e, const float rgba[4]) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.3f %.3f %.3f %.3f",
                      (double)rgba[0], (double)rgba[1], (double)rgba[2], (double)rgba[3]);
        e->SetAttribute("rgba", buf);
    };
    auto set_pos3 = [](XMLElement* e, double x, double y, double z) {
        char buf[64]; std::snprintf(buf, sizeof(buf), "%.5f %.5f %.5f", x, y, z);
        e->SetAttribute("pos", buf);
    };

    // Table
    if (sc->table.enabled) {
        const TableSpec& t = sc->table;
        double sz        = t.pos[2];
        double half_thick = t.thickness * 0.5;
        double top_cz    = sz - half_thick;
        double leg_h     = sz - t.thickness;

        XMLElement* tb = doc.NewElement("body");
        tb->SetAttribute("name", "table");
        set_pos3(tb, t.pos[0], t.pos[1], top_cz);

        // tabletop geom
        XMLElement* top = doc.NewElement("geom");
        top->SetAttribute("name", "table_top");
        top->SetAttribute("type", "box");
        { char sz3[64]; std::snprintf(sz3, sizeof(sz3), "%.5f %.5f %.5f",
                                      t.top_size[0], t.top_size[1], half_thick);
          top->SetAttribute("size", sz3); }
        set_rgba(top, t.rgba);
        top->SetAttribute("condim", "3");
        tb->InsertEndChild(top);

        // legs
        if (leg_h > 0.0) {
            double half_leg  = leg_h * 0.5;
            double leg_rel_z = -(sz * 0.5);
            double lx = t.top_size[0] - t.leg_radius;
            double ly = t.top_size[1] - t.leg_radius;
            const double corners[4][2] = {{lx, ly}, {-lx, ly}, {lx, -ly}, {-lx, -ly}};
            for (int i = 0; i < 4; ++i) {
                XMLElement* leg = doc.NewElement("geom");
                char nm[32]; std::snprintf(nm, sizeof(nm), "table_leg%d", i);
                leg->SetAttribute("name", nm);
                leg->SetAttribute("type", "cylinder");
                { char s2[32]; std::snprintf(s2, sizeof(s2), "%.5f %.5f", t.leg_radius, half_leg);
                  leg->SetAttribute("size", s2); }
                set_pos3(leg, corners[i][0], corners[i][1], leg_rel_z);
                set_rgba(leg, t.rgba);
                leg->SetAttribute("condim", "3");
                tb->InsertEndChild(leg);
            }
        }
        wb->InsertEndChild(tb);
    }

    // Objects
    for (const auto& obj : sc->objects) {
        XMLElement* ob = doc.NewElement("body");
        ob->SetAttribute("name", obj.name.c_str());
        set_pos3(ob, obj.pos[0], obj.pos[1], obj.pos[2]);

        if (!obj.fixed) {
            XMLElement* fj = doc.NewElement("freejoint");
            fj->SetAttribute("name", (obj.name + "_joint").c_str());
            ob->InsertEndChild(fj);
        }

        XMLElement* g = doc.NewElement("geom");
        const char* stype = (obj.shape == ObjShape::SPHERE)   ? "sphere"   :
                            (obj.shape == ObjShape::CYLINDER)  ? "cylinder" : "box";
        g->SetAttribute("type", stype);
        { char s[64];
          if (obj.shape == ObjShape::SPHERE)
              std::snprintf(s, sizeof(s), "%.5f", obj.size[0]);
          else if (obj.shape == ObjShape::CYLINDER)
              std::snprintf(s, sizeof(s), "%.5f %.5f", obj.size[0], obj.size[1]);
          else
              std::snprintf(s, sizeof(s), "%.5f %.5f %.5f",
                            obj.size[0], obj.size[1], obj.size[2]);
          g->SetAttribute("size", s); }
        set_rgba(g, obj.rgba);
        g->SetAttribute("condim", "3");
        ob->InsertEndChild(g);
        wb->InsertEndChild(ob);
    }

    return doc.SaveFile(path.c_str()) == XML_SUCCESS;
}

// KDL helpers

static bool load_kdl_chain(State* s, const std::string& urdf,
                            const char* base_link, const char* tip_link)
{
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf, tree)) {
        std::cerr << "[mj_kdl] kdl treeFromFile failed\n"; return false;
    }
    if (!tree.getChain(base_link, tip_link, s->chain)) {
        std::cerr << "[mj_kdl] getChain failed: " << base_link << " -> " << tip_link << "\n";
        return false;
    }
    auto model = urdf::parseURDFFile(urdf);
    s->joint_names.clear(); s->joint_limits.clear();
    for (unsigned i = 0; i < s->chain.getNrOfSegments(); ++i) {
        const KDL::Joint& j = s->chain.getSegment(i).getJoint();
        if (j.getType() == KDL::Joint::None) continue;
        const std::string& name = j.getName();
        s->joint_names.push_back(name);
        double lo = -M_PI, hi = M_PI;
        if (model) {
            auto uj = model->getJoint(name);
            if (uj && uj->limits && uj->type != urdf::Joint::CONTINUOUS) {
                lo = uj->limits->lower; hi = uj->limits->upper;
            }
        }
        s->joint_limits.emplace_back(lo, hi);
    }
    s->n_joints = (int)s->chain.getNrOfJoints();
    return true;
}

static void sync_chain_inertias(State* s, const std::string& pfx)
{
    if (!s->model) return;
    KDL::Chain chain;
    for (unsigned si = 0; si < s->chain.getNrOfSegments(); ++si) {
        const KDL::Segment& seg = s->chain.getSegment(si);
        int bid = mj_name2id(s->model, mjOBJ_BODY, (pfx + seg.getName()).c_str());
        KDL::RigidBodyInertia inertia;
        if (bid >= 0) {
            double mass = s->model->body_mass[bid];
            const double* ip = &s->model->body_ipos[3*bid];
            const double* iq = &s->model->body_iquat[4*bid];
            const double* id = &s->model->body_inertia[3*bid];
            double w=iq[0], x=iq[1], y=iq[2], z=iq[3];
            double R[3][3] = {
                {1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)},
                {2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)},
                {2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)}
            };
            double I[3][3] = {};
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) for (int c=0;c<3;c++)
                I[a][b] += R[a][c]*id[c]*R[b][c];
            inertia = KDL::RigidBodyInertia(mass, KDL::Vector(ip[0],ip[1],ip[2]),
                KDL::RotationalInertia(I[0][0],I[1][1],I[2][2],I[0][1],I[0][2],I[1][2]));
        }
        chain.addSegment(KDL::Segment(seg.getName(), seg.getJoint(), seg.getFrameToTip(), inertia));
    }
    s->chain = chain;
}

static void build_index_map(State* s, const std::string& pfx = "")
{
    s->kdl_to_mj_qpos.clear(); s->kdl_to_mj_dof.clear();
    if (!s->model) return;
    for (const auto& name : s->joint_names) {
        int id = mj_name2id(s->model, mjOBJ_JOINT, (pfx + name).c_str());
        if (id < 0) {
            std::cerr << "[mj_kdl] joint not found: " << pfx << name << "\n";
            int idx = (int)s->kdl_to_mj_qpos.size();
            s->kdl_to_mj_qpos.push_back(idx); s->kdl_to_mj_dof.push_back(idx);
        } else {
            s->kdl_to_mj_qpos.push_back(s->model->jnt_qposadr[id]);
            s->kdl_to_mj_dof.push_back(s->model->jnt_dofadr[id]);
        }
    }
}

// GLFW

static void cb_keyboard(GLFWwindow*, int, int, int, int);
static void cb_mouse_button(GLFWwindow*, int, int, int);
static void cb_mouse_move(GLFWwindow*, double, double);
static void cb_scroll(GLFWwindow*, double, double);
static State* g_state = nullptr;

struct GLMouseState {
    bool btn_left=false, btn_right=false, btn_middle=false;
    double mouse_x=0, mouse_y=0;
};

// Public API

bool build_scene(mjModel** out_model, mjData** out_data, const SceneSpec* sc)
{
    if (!sc || sc->robots.empty()) return false;

    fs::path tmp = fs::absolute(fs::path(sc->robots[0].urdf_path).parent_path());

    // Single robot: use spec API (avoids mj_saveLastXML which is unstable for
    // URDF-sourced models on some MuJoCo versions).
    if (sc->robots.size() == 1) {
        const SceneRobot& r = sc->robots[0];
        fs::path proc = tmp / "_mj_kdl_r0_proc.urdf";
        if (!preprocess_urdf(r.urdf_path, proc.string())) return false;

        char err[2048] = {};
        mjSpec* spec = mj_parseXML(proc.c_str(), nullptr, err, sizeof(err));
        if (!spec) { std::cerr << "[mj_kdl] parseXML: " << err << "\n"; return false; }

        spec->option.timestep   = sc->timestep;
        spec->option.gravity[2] = sc->gravity_z;
        spec->compiler.balanceinertia = 1;
        spec->compiler.discardvisual  = 0;

        if (r.pos[0] || r.pos[1] || r.pos[2] || r.euler[0] || r.euler[1] || r.euler[2]) {
            mjsBody* wb = mjs_findBody(spec, "world");
            mjsBody* root = wb ? mjs_asBody(mjs_firstChild(wb, mjOBJ_BODY, 0)) : nullptr;
            if (root) {
                root->pos[0]=r.pos[0]; root->pos[1]=r.pos[1]; root->pos[2]=r.pos[2];
                if (r.euler[0] || r.euler[1] || r.euler[2]) {
                    root->alt.type=mjORIENTATION_EULER;
                    root->alt.euler[0]=r.euler[0]; root->alt.euler[1]=r.euler[1]; root->alt.euler[2]=r.euler[2];
                }
            }
        }
        if (sc->add_floor)          add_floor_to_spec(spec);
        if (sc->table.enabled)      add_table_to_spec(spec, sc->table);
        if (!sc->objects.empty())   add_objects_to_spec(spec, sc->objects);

        *out_model = mj_compile(spec, nullptr);
        mj_deleteSpec(spec);
        if (!*out_model) { std::cerr << "[mj_kdl] compile failed\n"; return false; }
        *out_data = mj_makeData(*out_model);
        if (!*out_data) { mj_deleteModel(*out_model); *out_model=nullptr; return false; }
        return true;
    }

    // Multi-robot: URDF → raw MJCF → combine → inject scene → load.
    std::vector<std::string> raws;
    for (int i = 0; i < (int)sc->robots.size(); ++i) {
        fs::path proc = tmp / ("_mj_kdl_r" + std::to_string(i) + "_proc.urdf");
        fs::path raw  = tmp / ("_mj_kdl_r" + std::to_string(i) + "_raw.xml");
        if (!preprocess_urdf(sc->robots[i].urdf_path, proc.string())) return false;
        if (!urdf_to_raw_mjcf(proc.string(), raw.string())) return false;
        raws.push_back(raw.string());
    }
    fs::path combined = tmp / "_mj_kdl_combined_raw.xml";
    fs::path scene    = tmp / "_mj_kdl_scene.xml";
    if (!combine_mjcf(raws, sc, combined.string())) return false;
    if (!inject_scene(combined.string(), scene.string(), sc->timestep, sc->gravity_z, sc->add_floor))
        return false;
    if (!inject_extras_xml(scene.string(), sc)) return false;
    char err[2048] = {};
    *out_model = mj_loadXML(scene.c_str(), nullptr, err, sizeof(err));
    if (!*out_model) { std::cerr << "[mj_kdl] loadXML: " << err << "\n"; return false; }
    *out_data = mj_makeData(*out_model);
    if (!*out_data) { mj_deleteModel(*out_model); *out_model=nullptr; return false; }
    return true;
}

void destroy_scene(mjModel* model, mjData* data)
{
    if (data)  mj_deleteData(data);
    if (model) mj_deleteModel(model);
}

bool scene_add_object(mjModel** model, mjData** data,
                      SceneSpec* spec, const SceneObject& obj)
{
    spec->objects.push_back(obj);
    mjModel* nm = nullptr; mjData* nd = nullptr;
    if (!build_scene(&nm, &nd, spec)) {
        spec->objects.pop_back();
        return false;
    }
    destroy_scene(*model, *data);
    *model = nm; *data = nd;
    return true;
}

bool scene_remove_object(mjModel** model, mjData** data,
                         SceneSpec* spec, const std::string& name)
{
    auto it = std::find_if(spec->objects.begin(), spec->objects.end(),
                           [&](const SceneObject& o){ return o.name == name; });
    if (it == spec->objects.end()) return false;
    SceneObject removed = *it;
    spec->objects.erase(it);
    mjModel* nm = nullptr; mjData* nd = nullptr;
    if (!build_scene(&nm, &nd, spec)) {
        spec->objects.push_back(removed);
        return false;
    }
    destroy_scene(*model, *data);
    *model = nm; *data = nd;
    return true;
}

bool init_robot(State* s, mjModel* model, mjData* data,
                const char* urdf, const char* base_link, const char* tip_link, const char* prefix)
{
    s->model=model; s->data=data; s->_owns_model=false;
    if (!load_kdl_chain(s, urdf, base_link, tip_link)) return false;
    std::string pfx = prefix ? prefix : "";
    sync_chain_inertias(s, pfx);
    build_index_map(s, pfx);
    return true;
}

bool init_window(State* s, const char* title, int width, int height)
{
    if (!s->model) return false;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;
    if (!glfwInit()) return false;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_FALSE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    s->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!s->window) { glfwTerminate(); return false; }

    auto* ms = new GLMouseState();
    glfwSetWindowUserPointer(s->window, ms);
    glfwSetKeyCallback        (s->window, cb_keyboard);
    glfwSetMouseButtonCallback(s->window, cb_mouse_button);
    glfwSetCursorPosCallback  (s->window, cb_mouse_move);
    glfwSetScrollCallback     (s->window, cb_scroll);
    glfwSetWindowCloseCallback(s->window, [](GLFWwindow* w){ glfwSetWindowShouldClose(w, GLFW_TRUE); });
    glfwMakeContextCurrent(s->window);
    glfwSwapInterval(1);

    if (!glfwGetProcAddress("glGenBuffers")) {
        delete ms; glfwDestroyWindow(s->window); s->window=nullptr; glfwTerminate(); return false;
    }

    mjv_defaultCamera(&s->cam); mjv_defaultOption(&s->opt); mjv_defaultPerturb(&s->pert);
    mjv_makeScene(s->model, &s->scn, 2000);
    mjr_makeContext(s->model, &s->con, mjFONTSCALE_150);
    s->cam.type=mjCAMERA_FREE; s->cam.distance=2.5; s->cam.azimuth=135.0; s->cam.elevation=-20.0;
    g_state = s;
    return true;
}

bool init(State* s, const Config* cfg)
{
    SceneSpec sc;
    SceneRobot r; r.urdf_path=cfg->urdf_path; r.prefix="";
    sc.robots.push_back(r);
    sc.timestep=cfg->timestep; sc.gravity_z=cfg->gravity_z; sc.add_floor=cfg->add_floor;

    mjModel* model=nullptr; mjData* data=nullptr;
    if (!build_scene(&model, &data, &sc)) return false;
    s->model=model; s->data=data; s->_owns_model=true;

    if (!load_kdl_chain(s, cfg->urdf_path, cfg->base_link, cfg->tip_link)) {
        destroy_scene(model, data); s->model=nullptr; s->data=nullptr; return false;
    }
    sync_chain_inertias(s, "");
    build_index_map(s);

    if (!cfg->headless && !init_window(s, cfg->win_title, cfg->win_width, cfg->win_height))
        std::cerr << "[mj_kdl] no GL — headless\n";
    return true;
}

void cleanup(State* s)
{
    if (s->window) {
        mjv_freeScene(&s->scn); mjr_freeContext(&s->con);
        delete static_cast<GLMouseState*>(glfwGetWindowUserPointer(s->window));
        glfwDestroyWindow(s->window); s->window=nullptr; glfwTerminate();
        if (g_state == s) g_state=nullptr;
    }
    if (s->_owns_model) {
        if (s->data)  { mj_deleteData(s->data);   s->data=nullptr;  }
        if (s->model) { mj_deleteModel(s->model); s->model=nullptr; }
    } else { s->data=nullptr; s->model=nullptr; }
}

void step(State* s)
{
    if (!s->model || !s->data) return;
    if (s->pert.active) mjv_applyPerturbForce(s->model, s->data, &s->pert);
    mj_step(s->model, s->data);
}

void step_n(State* s, int n) { for (int i=0; i<n; ++i) step(s); }

void reset(State* s)
{
    if (s->model && s->data) { mj_resetData(s->model, s->data); mj_forward(s->model, s->data); }
}

bool is_running(const State* s)
{
    if (!s->window) return s->model != nullptr;
    return !glfwWindowShouldClose(s->window);
}

bool render(State* s)
{
    if (!s->window) return is_running(s);
    if (glfwWindowShouldClose(s->window)) return false;
    glfwPollEvents();
    int w, h; glfwGetFramebufferSize(s->window, &w, &h);
    mjrRect vp = {0, 0, w, h};
    mjv_updateScene(s->model, s->data, &s->opt, &s->pert, &s->cam, mjCAT_ALL, &s->scn);
    mjr_render(vp, &s->scn, &s->con);
    char top[128], bot[256];
    std::snprintf(top, sizeof(top), "t = %.3f s", s->data->time);
    std::snprintf(bot, sizeof(bot),
        "Ctrl+RightDrag: push body\nCtrl+LeftDrag:  rotate body\n"
        "RightDrag: pan  LeftDrag: orbit  Scroll: zoom");
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT,    vp, top, nullptr, &s->con);
    mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, vp, bot, nullptr, &s->con);
    glfwSwapBuffers(s->window);
    return true;
}

bool sync_to_kdl(const State* s, KDL::JntArray& q)
{
    if (!s->data) return false;
    q.resize(s->n_joints);
    for (int i=0; i<s->n_joints; ++i) q(i) = s->data->qpos[s->kdl_to_mj_qpos[i]];
    return true;
}

void sync_from_kdl(State* s, const KDL::JntArray& q)
{
    if (!s->data) return;
    int n = std::min((int)q.rows(), s->n_joints);
    for (int i=0; i<n; ++i) s->data->qpos[s->kdl_to_mj_qpos[i]] = q(i);
}

void set_torques(State* s, const KDL::JntArray& tau)
{
    if (!s->data) return;
    int n = std::min((int)tau.rows(), s->n_joints);
    for (int i=0; i<n; ++i) s->data->qfrc_applied[s->kdl_to_mj_dof[i]] = tau(i);
}

// GLFW callbacks

static void cb_keyboard(GLFWwindow* w, int key, int, int action, int)
{
    if (action == GLFW_PRESS && (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q))
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

static void cb_mouse_button(GLFWwindow* w, int btn, int act, int)
{
    auto* ms = static_cast<GLMouseState*>(glfwGetWindowUserPointer(w));
    ms->btn_left   = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT)   == GLFW_PRESS);
    ms->btn_right  = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT)  == GLFW_PRESS);
    ms->btn_middle = (glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    glfwGetCursorPos(w, &ms->mouse_x, &ms->mouse_y);
    if (!g_state) return;
    bool ctrl = (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
                 glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    if (ctrl && act == GLFW_PRESS &&
        (btn == GLFW_MOUSE_BUTTON_LEFT || btn == GLFW_MOUSE_BUTTON_RIGHT)) {
        int ww, wh; glfwGetWindowSize(w, &ww, &wh);
        mjtNum selpnt[3];
        int geomid[1]={-1}, flexid[1]={-1}, skinid[1]={-1};
        int body = mjv_select(g_state->model, g_state->data, &g_state->opt,
                              (mjtNum)wh/ww, (mjtNum)ms->mouse_x/ww,
                              (mjtNum)(wh-ms->mouse_y)/wh,
                              &g_state->scn, selpnt, geomid, flexid, skinid);
        if (body >= 0) {
            g_state->pert.select=body; g_state->pert.skinselect=skinid[0];
            mju_copy3(g_state->pert.localpos, selpnt);
            g_state->pert.active = (btn == GLFW_MOUSE_BUTTON_RIGHT) ? mjPERT_TRANSLATE : mjPERT_ROTATE;
            mjv_initPerturb(g_state->model, g_state->data, &g_state->scn, &g_state->pert);
        }
    } else if (!ms->btn_left && !ms->btn_right && !ms->btn_middle) {
        g_state->pert.active=0; g_state->pert.select=0;
        mju_zero(g_state->data->xfrc_applied, 6*g_state->model->nbody);
    }
}

static void cb_mouse_move(GLFWwindow* w, double x, double y)
{
    auto* ms = static_cast<GLMouseState*>(glfwGetWindowUserPointer(w));
    if (!g_state || (!ms->btn_left && !ms->btn_right && !ms->btn_middle)) return;
    double dx = x-ms->mouse_x, dy = y-ms->mouse_y;
    ms->mouse_x=x; ms->mouse_y=y;
    int ww, wh; glfwGetWindowSize(w, &ww, &wh);
    bool ctrl  = (glfwGetKey(w, GLFW_KEY_LEFT_CONTROL)  == GLFW_PRESS ||
                  glfwGetKey(w, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
    bool shift = (glfwGetKey(w, GLFW_KEY_LEFT_SHIFT)    == GLFW_PRESS ||
                  glfwGetKey(w, GLFW_KEY_RIGHT_SHIFT)   == GLFW_PRESS);
    if (ctrl && g_state->pert.select > 0) {
        mjtMouse act = ms->btn_right ? (shift ? mjMOUSE_MOVE_H   : mjMOUSE_MOVE_V)
                     : ms->btn_left  ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                     : mjMOUSE_ZOOM;
        mjv_movePerturb(g_state->model, g_state->data, act, dx/wh, dy/wh, &g_state->scn, &g_state->pert);
    } else {
        mjtMouse act = ms->btn_right ? (shift ? mjMOUSE_MOVE_H   : mjMOUSE_MOVE_V)
                     : ms->btn_left  ? (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V)
                     : mjMOUSE_ZOOM;
        mjv_moveCamera(g_state->model, act, dx/wh, dy/wh, &g_state->scn, &g_state->cam);
    }
}

static void cb_scroll(GLFWwindow*, double, double yoff)
{
    if (g_state) mjv_moveCamera(g_state->model, mjMOUSE_ZOOM, 0, -0.05*yoff, &g_state->scn, &g_state->cam);
}

} // namespace mj_kdl
