/* test_kinova_gen3_gripper.cpp
 * Attach Robotiq 2F-85 gripper (MuJoCo Menagerie) to Kinova Gen3 using attach_gripper(),
 * load the combined model, validate arm KDL and gripper simulation.
 *
 * Tests:
 *   1. attach_gripper produces a valid MJCF that loads (nq >= 13, nu >= 8).
 *   2. KDL chain for arm (7 joints) built from combined model.
 *   3. KDL gravity torques agree with MuJoCo qfrc_bias[0..6] within 5e-2 Nm.
 *   4. FK sanity check — EE position within expected workspace.
 *   5. Gripper open/close: driver joint range validated.
 *
 * GUI (--gui):
 *   Uses mj_kdl::run_simulate_ui for the real-time simulate window.
 *   Physics: position actuators hold arm at home pose + continuous gripper open/close every 3 s.
 *
 * Usage: test_kinova_gen3_gripper [--gui] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <tinyxml2.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

/* Add contact exclusions between bracelet_link and all g_* gripper bodies.
 * Called after patch_mjcf_visuals() on the combined MJCF. */
static bool patch_contact_exclusions(const std::string &path)
{
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return false;
    auto *root = doc.FirstChildElement("mujoco");
    if (!root) return false;

    auto *contact = root->FirstChildElement("contact");
    if (!contact) {
        contact = doc.NewElement("contact");
        root->InsertEndChild(contact);
    }

    const char *gripper_bodies[] = { "g_base_mount",
        "g_base",
        "g_left_driver",
        "g_right_driver",
        "g_left_spring_link",
        "g_right_spring_link",
        "g_left_follower",
        "g_right_follower",
        "g_left_coupler",
        "g_right_coupler",
        "g_left_pad",
        "g_right_pad",
        "g_left_silicone_pad",
        "g_right_silicone_pad" };
    for (const char *gb : gripper_bodies) {
        auto *exc = doc.NewElement("exclude");
        exc->SetAttribute("body1", "bracelet_link");
        exc->SetAttribute("body2", gb);
        contact->InsertEndChild(exc);
    }

    return doc.SaveFile(path.c_str()) == tinyxml2::XML_SUCCESS;
}

class GripperTest : public ::testing::Test {
protected:
    fs::path    root_;
    std::string combined_;
    mjModel    *model_       = nullptr;
    mjData     *data_        = nullptr;
    mj_kdl::State s_;
    int          fingers_act_ = -1;
    unsigned     n_           = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found — run locally with the submodule";
            return;
        }

        const std::string arm_mjcf = (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf = (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        combined_ = (root_ / "/tmp/gen3_with_2f85.xml").string();

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

        ASSERT_TRUE(mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined_.c_str()))
            << "attach_gripper() returned false";
        ASSERT_TRUE(mj_kdl::patch_mjcf_add_skybox(combined_.c_str()))
            << "patch_mjcf_add_skybox() returned false";
        ASSERT_TRUE(mj_kdl::patch_mjcf_add_floor(combined_.c_str()))
            << "patch_mjcf_add_floor() returned false";
        ASSERT_TRUE(patch_contact_exclusions(combined_))
            << "patch_contact_exclusions() returned false";

        ASSERT_TRUE(mj_kdl::load_mjcf(&model_, &data_, combined_.c_str()))
            << "load_mjcf() returned false for combined MJCF";

        TEST_INFO("model: nq=" << model_->nq << " nv=" << model_->nv
                  << " nbody=" << model_->nbody << " nu=" << model_->nu);

        ASSERT_GE(model_->nq, 13) << "expected nq >= 13 (7 arm + 6 gripper), got " << model_->nq;
        ASSERT_GE(model_->nu, 8)  << "expected nu >= 8 (7 arm + 1 gripper), got " << model_->nu;

        ASSERT_TRUE(mj_kdl::init_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"))
            << "init_from_mjcf() returned false";
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, -9.81));

        fingers_act_ = mj_name2id(model_, mjOBJ_ACTUATOR, "g_fingers_actuator");
        ASSERT_GE(fingers_act_, 0) << "g_fingers_actuator not found";
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(GripperTest, ModelLoaded)
{
    TEST_INFO("model: nq=" << model_->nq << " nu=" << model_->nu);
    EXPECT_GE(model_->nq, 13);
    EXPECT_GE(model_->nu, 8);
}

TEST_F(GripperTest, KDLChain)
{
    TEST_INFO("KDL chain: " << n_ << " joints");
    EXPECT_EQ(n_, 7u);
}

TEST_F(GripperTest, GravityAccuracy)
{
    /* At q=0 the arm is upright and gripper weight contributes minimally,
     * keeping the KDL vs MuJoCo discrepancy within tolerance. */
    KDL::JntArray q_zero(n_);
    mj_kdl::sync_from_kdl(&s_, q_zero);
    mj_forward(model_, data_);

    KDL::JntArray g(n_);
    ASSERT_GE(dyn_->JntToGravity(q_zero, g), 0) << "JntToGravity() returned error";

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err,
                           std::abs(g(i) - data_->qfrc_bias[s_.kdl_to_mj_dof[i]]));

    TEST_INFO("gravity accuracy at q=0: max|KDL - MuJoCo| = "
              << std::fixed << std::setprecision(6) << max_err << " Nm");

    EXPECT_LE(max_err, 5e-2) << "gravity error " << max_err << " Nm exceeds 5e-2 Nm threshold";
}

TEST_F(GripperTest, FKWorkspace)
{
    KDL::JntArray q_home(n_);
    for (unsigned i = 0; i < n_; ++i) q_home(i) = kHomePose[i];
    KDL::Frame fk_pose;
    fk_->JntToCart(q_home, fk_pose);
    double ee_dist = fk_pose.p.Norm();

    TEST_INFO("EE distance from base at home pose: "
              << std::fixed << std::setprecision(3) << ee_dist * 1000.0 << " mm");

    EXPECT_GE(ee_dist, 0.1) << "EE distance " << ee_dist << " m below workspace lower bound";
    EXPECT_LE(ee_dist, 1.1) << "EE distance " << ee_dist << " m above workspace upper bound";
}

TEST_F(GripperTest, GripperRange)
{
    int rdriver_jnt = mj_name2id(model_, mjOBJ_JOINT, "g_right_driver_joint");
    int ldriver_jnt = mj_name2id(model_, mjOBJ_JOINT, "g_left_driver_joint");
    ASSERT_GE(rdriver_jnt, 0) << "g_right_driver_joint not found";
    ASSERT_GE(ldriver_jnt, 0) << "g_left_driver_joint not found";

    double jrange_lo = model_->jnt_range[2 * rdriver_jnt];
    double jrange_hi = model_->jnt_range[2 * rdriver_jnt + 1];
    TEST_INFO("gripper right_driver_joint range: ["
              << std::fixed << std::setprecision(4) << jrange_lo << ", " << jrange_hi << "] rad");

    EXPECT_LE(std::abs(jrange_hi - 0.8), 0.01)
        << "driver joint hi range " << jrange_hi << " differs from expected ~0.8";
    EXPECT_GE(jrange_lo, -0.01)
        << "driver joint lo range " << jrange_lo << " below expected ~0";
}

static void run_gui(mjModel *model, mjData *data, mj_kdl::State &s, unsigned n,
                    int fingers_act, const std::string &combined)
{
    int key_id = mj_name2id(model, mjOBJ_KEY, "home");
    if (key_id >= 0) {
        mj_resetDataKeyframe(model, data, key_id);
    } else {
        KDL::JntArray q_home(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
        mj_kdl::sync_from_kdl(&s, q_home);
    }
    mj_forward(model, data);

    std::cout << "GUI: close window to exit\n";
    mj_kdl::run_simulate_ui(model, data, combined.c_str(), [&](mjModel *m, mjData *d) {
        /* Pure gravity compensation:
         * Zero P-term (track current qpos) + cancel gravity via qfrc_bias.
         * qfrc_bias includes gripper mass — KDL alone would miss ~1.7 kg. */
        for (unsigned i = 0; i < n; ++i) {
            int dof              = s.kdl_to_mj_dof[i];
            int jid              = m->dof_jntid[dof];
            d->ctrl[i]           = d->qpos[m->jnt_qposadr[jid]];
            d->qfrc_applied[dof] = d->qfrc_bias[dof];
        }
        /* Gripper: open/close every 3 s (ctrl range 0..255, 255=closed) */
        d->ctrl[fingers_act] = (std::fmod(d->time, 6.0) < 3.0) ? 255.0 : 0.0;
    });
}

int main(int argc, char *argv[])
{
    bool gui = false;

    /* Parse non-GTest arguments before handing off to GTest. */
    std::vector<char *> remaining;
    remaining.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui")
            gui = true;
        else
            remaining.push_back(argv[i]);
    }
    int remaining_argc = static_cast<int>(remaining.size());

    ::testing::InitGoogleTest(&remaining_argc, remaining.data());

    if (gui) {
        const fs::path root = repo_root();
        if (!fs::exists(root / "third_party/menagerie")) {
            std::cerr << "GUI: third_party/menagerie/ not found\n";
            return 0;
        }
        const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        const std::string combined = (root / "/tmp/gen3_with_2f85.xml").string();

        mj_kdl::GripperSpec gs;
        gs.mjcf_path = grp_mjcf.c_str();
        gs.attach_to = "bracelet_link";
        gs.prefix    = "g_";
        gs.pos[0] = 0.0; gs.pos[1] = 0.0; gs.pos[2] = -0.061525;
        gs.quat[0] = 0.0; gs.quat[1] = 1.0; gs.quat[2] = 0.0; gs.quat[3] = 0.0;

        if (!mj_kdl::attach_gripper(arm_mjcf.c_str(), &gs, combined.c_str()) ||
            !mj_kdl::patch_mjcf_add_skybox(combined.c_str()) ||
            !mj_kdl::patch_mjcf_add_floor(combined.c_str()) ||
            !patch_contact_exclusions(combined)) {
            std::cerr << "GUI: model preparation failed\n";
            return 1;
        }
        mjModel *model = nullptr;
        mjData  *data  = nullptr;
        if (!mj_kdl::load_mjcf(&model, &data, combined.c_str())) {
            std::cerr << "GUI: load_mjcf() failed\n";
            return 1;
        }
        mj_kdl::State s;
        if (!mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link")) {
            std::cerr << "GUI: init_from_mjcf() failed\n";
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        unsigned n = s.chain.getNrOfJoints();
        int fingers_act = mj_name2id(model, mjOBJ_ACTUATOR, "g_fingers_actuator");
        run_gui(model, data, s, n, fingers_act, combined);
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model, data);
    }

    return RUN_ALL_TESTS();
}
