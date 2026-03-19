/* test_kinova_gen3_impedance.cpp
 * Kinova Gen3 + Robotiq 2F-85: home-pose impedance control.
 *
 * Impedance law (per joint):
 *   tau[i] = Kp[i]*(q_home[i] - q[i]) - Kd[i]*qdot[i] + qfrc_bias[dof_i]
 * Applied via qfrc_applied; position actuators are zeroed (ctrl = current_qpos).
 *
 * Tests:
 *   1. KDL chain: 7 arm joints.
 *   2. 200-step impedance hold at home pose  - EE drift < 1 mm. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <tinyxml2.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

/* Impedance gains  - tuned for Gen3 joint sizes.
 * Large joints (2,4,6): higher stiffness; small joints (1,3,5,7): lower. */
static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

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

class ImpedanceTest : public ::testing::Test
{
  protected:
    fs::path                                         root_;
    std::string                                      combined_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    int                                              fingers_act_ = -1;
    unsigned                                         n_           = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found  - run locally with the submodule";
            return;
        }

        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        combined_ = "/tmp/gen3_with_2f85_impedance.xml";

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
        ASSERT_GE(model_->nq, 13) << "expected nq>=13, got " << model_->nq;
        ASSERT_GE(model_->nu, 8) << "expected nu>=8, got " << model_->nu;

        ASSERT_TRUE(mj_kdl::init_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"))
          << "init_from_mjcf() returned false";
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);

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

    /* Apply one impedance step targeting q_target and advance the simulation. */
    void step_impedance(mjModel *m, mjData *d, const double q_target[7])
    {
        for (unsigned i = 0; i < n_; ++i) {
            int    dof           = s_.kdl_to_mj_dof[i];
            int    jid           = m->dof_jntid[dof];
            double q             = d->qpos[m->jnt_qposadr[jid]];
            double qdot          = d->qvel[dof];
            d->ctrl[i]           = q; // zero P-error
            d->qfrc_applied[dof] = kKp[i] * (q_target[i] - q) - kKd[i] * qdot + d->qfrc_bias[dof];
        }
        mj_step(m, d);
    }
};

TEST_F(ImpedanceTest, KDLChain)
{
    TEST_INFO("KDL chain: " << n_ << " joints");
    EXPECT_EQ(n_, 7u);
}

TEST_F(ImpedanceTest, ImpedanceDrift)
{
    KDL::JntArray q_home_kdl(n_);
    for (unsigned i = 0; i < n_; ++i) q_home_kdl(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&s_, q_home_kdl);
    mj_forward(model_, data_);

    KDL::Frame ee_init;
    fk_->JntToCart(q_home_kdl, ee_init);

    for (int step = 0; step < 200; ++step) step_impedance(model_, data_, kHomePose);

    KDL::JntArray q_now(n_);
    mj_kdl::sync_to_kdl(&s_, q_now);
    KDL::Frame ee_now;
    fk_->JntToCart(q_now, ee_now);
    double drift = (ee_now.p - ee_init.p).Norm() * 1000.0;

    TEST_INFO(
      "impedance hold drift (200 steps): " << std::fixed << std::setprecision(3) << drift << " mm");

    EXPECT_LE(drift, 1.0) << "EE drift " << drift << " mm exceeds 1 mm threshold";
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
