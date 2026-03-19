/* ex_dual_arm.cpp
 * Two Kinova Gen3 arms, each fitted with a Robotiq 2F-85 gripper,
 * in a shared MuJoCo scene.
 *
 * arm1 at x = -0.5 m, facing +X.
 * arm2 at x = +0.5 m, facing -X (180° yaw); all element names prefixed "r2_".
 *
 * Both arms hold the home pose under gravity compensation.
 * Grippers cycle open/closed every 3 s.
 *
 * Requires third_party/menagerie submodule.
 *
 * Usage:
 *   ex_dual_arm [--headless]
 *
 * --headless: run 600 steps and print both EE positions, then exit. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }

/* Recursively prefix all body/joint/actuator/geom/site names in an element
 * subtree.  Shared asset entries (mesh, material, texture, hfield) keep their
 * names so arm1 and arm2 can reference the same loaded meshes. */
static void prefix_names(tinyxml2::XMLElement *e, const std::string &pfx)
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
    for (auto *c = e->FirstChildElement(); c; c = c->NextSiblingElement()) prefix_names(c, pfx);
}

/* Add bracelet_link ↔ g_* contact exclusions to the MJCF. */
static bool patch_contact_exclusions(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto *root = doc.FirstChildElement("mujoco");
    if (!root) return false;
    auto *ct = root->FirstChildElement("contact");
    if (!ct) {
        ct = doc.NewElement("contact");
        root->InsertEndChild(ct);
    }
    struct Pair
    {
        const char *b1, *b2;
    };
    static const Pair kExclude[] = {
        { "bracelet_link", "g_base" },
        { "bracelet_link", "g_left_pad" },
        { "bracelet_link", "g_right_pad" },
        { "half_arm_2_link", "g_base" },
    };
    for (const auto &p : kExclude) {
        auto *ex = doc.NewElement("exclude");
        ex->SetAttribute("body1", p.b1);
        ex->SetAttribute("body2", p.b2);
        ct->InsertEndChild(ex);
    }
    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

/* Build a merged MJCF with both arms and their grippers. */
static bool build_dual_arm_scene(const fs::path &root, const std::string &out)
{
    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string a1       = "/tmp/ex_da_arm1.xml";
    const std::string a2       = "/tmp/ex_da_arm2.xml";

    mj_kdl::GripperSpec gs;
    gs.mjcf_path = grp_mjcf.c_str();
    gs.attach_to = "bracelet_link";
    gs.prefix    = "g_";
    gs.pos[0]    = 0.0;
    gs.pos[1]    = 0.0;
    gs.pos[2]    = -0.061525;
    gs.quat[0]   = 0.0;
    gs.quat[1]   = 1.0;
    gs.quat[2]   = 0.0;
    gs.quat[3]   = 0.0;

    if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, a1.c_str())
        || !mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, a2.c_str()))
        return false;
    if (!patch_contact_exclusions(a1) || !patch_contact_exclusions(a2)) return false;

    /* Assemble merged world document */
    using namespace tinyxml2;
    XMLDocument doc;
    auto       *mj = doc.NewElement("mujoco");
    mj->SetAttribute("model", "dual_arm_gripper");
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

    /* Merge one arm+gripper MJCF into the world.
     * copy_assets: copy <asset> children (arm1 only; arm2 reuses the same meshes).
     * pfx: element-name prefix (empty for arm1, "r2_" for arm2).
     * x,y,z,ez: placement transform. */
    auto embed = [&](const std::string &file,
                   bool                 copy_assets,
                   const std::string   &pfx,
                   double               x,
                   double               y,
                   double               z,
                   double               ez) -> bool {
        XMLDocument src;
        if (src.LoadFile(file.c_str()) != XML_SUCCESS) return false;
        auto *src_root = src.FirstChildElement("mujoco");
        if (!src_root) return false;

        if (copy_assets) {
            auto *sa = src_root->FirstChildElement("asset");
            if (sa)
                for (auto *c = sa->FirstChildElement(); c; c = c->NextSiblingElement())
                    asset->InsertEndChild(c->DeepClone(&doc));
        }

        /* Wrap arm tree in a placement body */
        char pos_s[64];
        std::snprintf(pos_s, sizeof(pos_s), "%.3f %.3f %.3f", x, y, z);
        auto *wrap = doc.NewElement("body");
        wrap->SetAttribute("pos", pos_s);
        if (std::abs(ez) > 0.1) {
            char euler_s[32];
            std::snprintf(euler_s, sizeof(euler_s), "0 0 %.1f", ez);
            wrap->SetAttribute("euler", euler_s);
        }
        worldbody->InsertEndChild(wrap);

        auto *src_wb = src_root->FirstChildElement("worldbody");
        if (src_wb)
            for (auto *c = src_wb->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) prefix_names(copy, pfx);
                wrap->InsertEndChild(copy);
            }

        auto *src_act = src_root->FirstChildElement("actuator");
        if (src_act)
            for (auto *c = src_act->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) prefix_names(copy, pfx);
                actuators->InsertEndChild(copy);
            }

        auto *src_ct = src_root->FirstChildElement("contact");
        if (src_ct)
            for (auto *c = src_ct->FirstChildElement(); c; c = c->NextSiblingElement()) {
                auto *copy = static_cast<XMLElement *>(c->DeepClone(&doc));
                if (!pfx.empty()) prefix_names(copy, pfx);
                contact->InsertEndChild(copy);
            }

        return true;
    };

    if (!embed(a1, true, "", -0.5, 0.0, 0.0, 0.0)) return false;
    if (!embed(a2, false, "r2_", 0.5, 0.0, 0.0, 180.0)) return false;
    if (doc.SaveFile(out.c_str()) != XML_SUCCESS) return false;

    return mj_kdl::patch_mjcf_add_skybox(out.c_str()) && mj_kdl::patch_mjcf_add_floor(out.c_str());
}

int main(int argc, char *argv[])
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    if (!fs::exists(root / "third_party/menagerie")) {
        std::cerr << "third_party/menagerie/ not found — run:\n"
                     "  git submodule update --init third_party/menagerie\n";
        return 1;
    }

    const std::string combined = "/tmp/ex_dual_arm.xml";
    if (!build_dual_arm_scene(root, combined)) {
        std::cerr << "build_dual_arm_scene() failed\n";
        return 1;
    }

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
        std::cerr << "load_mjcf() failed\n";
        return 1;
    }

    mj_kdl::Robot arm1, arm2;
    if (!mj_kdl::init_from_mjcf(&arm1, model, data, "base_link", "bracelet_link")
        || !mj_kdl::init_from_mjcf(&arm2, model, data, "r2_base_link", "r2_bracelet_link")) {
        std::cerr << "init_from_mjcf() failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const int n     = arm1.n_joints;
    int       fing1 = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    int       fing2 = mj_name2id(model, mjOBJ_ACTUATOR, "r2_g_fingers_actuator");

    /* Build joint → ctrl-index map from MuJoCo's actuator_trnid table. */
    auto build_ctrl_map = [&](const mj_kdl::Robot *arm) {
        std::vector<int> ctrl(arm->n_joints, -1);
        for (int i = 0; i < arm->n_joints; ++i) {
            int jid = model->dof_jntid[arm->kdl_to_mj_dof[i]];
            for (int ai = 0; ai < model->nu; ++ai) {
                if (model->actuator_trntype[ai] == mjTRN_JOINT
                    && model->actuator_trnid[2 * ai] == jid) {
                    ctrl[i] = ai;
                    break;
                }
            }
        }
        return ctrl;
    };
    const auto ctrl1 = build_ctrl_map(&arm1);
    const auto ctrl2 = build_ctrl_map(&arm2);

    /* Set home pose for both arms */
    KDL::JntArray q_home(n);
    for (int i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&arm1, q_home);
    mj_kdl::sync_from_kdl(&arm2, q_home);
    mj_forward(model, data);

    /* Hold position + feed-forward gravity via qfrc_bias (includes gripper mass). */
    auto apply_ctrl = [&](const mj_kdl::Robot *arm, const std::vector<int> &ctrl_idx) {
        for (int i = 0; i < arm->n_joints; ++i) {
            int dof = arm->kdl_to_mj_dof[i];
            int jid = model->dof_jntid[dof];
            if (ctrl_idx[i] >= 0) data->ctrl[ctrl_idx[i]] = data->qpos[model->jnt_qposadr[jid]];
            data->qfrc_applied[dof] = data->qfrc_bias[dof];
        }
    };

    if (headless) {
        for (int step = 0; step < 600; ++step) {
            apply_ctrl(&arm1, ctrl1);
            apply_ctrl(&arm2, ctrl2);
            if (fing1 >= 0) data->ctrl[fing1] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
            if (fing2 >= 0) data->ctrl[fing2] = (std::fmod(data->time, 6.0) < 3.0) ? 255.0 : 0.0;
            mj_kdl::step(&arm1);
        }

        KDL::ChainFkSolverPos_recursive fk1(arm1.chain), fk2(arm2.chain);
        KDL::JntArray                   q1, q2;
        KDL::Frame                      ee1, ee2;
        mj_kdl::sync_to_kdl(&arm1, q1);
        mj_kdl::sync_to_kdl(&arm2, q2);
        fk1.JntToCart(q1, ee1);
        fk2.JntToCart(q2, ee2);
        std::cout << "arm1 EE: [" << ee1.p.x() << ", " << ee1.p.y() << ", " << ee1.p.z() << "]\n";
        std::cout << "arm2 EE: [" << ee2.p.x() << ", " << ee2.p.y() << ", " << ee2.p.z() << "]\n";
    } else {
        mj_kdl::run_simulate_ui(model, data, combined.c_str(), [&](mjModel *m, mjData *d) {
            apply_ctrl(&arm1, ctrl1);
            apply_ctrl(&arm2, ctrl2);
            if (fing1 >= 0) d->ctrl[fing1] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
            if (fing2 >= 0) d->ctrl[fing2] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
        });
    }

    mj_kdl::cleanup(&arm1);
    mj_kdl::cleanup(&arm2);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
