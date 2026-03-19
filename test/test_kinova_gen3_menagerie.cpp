/* test_kinova_gen3_menagerie.cpp
 * Load Kinova Gen3 from third_party/menagerie/kinova_gen3/gen3.xml (MuJoCo Menagerie MJCF)
 * and validate the KDL chain built directly from the compiled model.
 *
 * Tests:
 *   1. MJCF loads: nv=7, nbody>=9.
 *   2. KDL chain (7 joints) built from model via init_from_mjcf.
 *   3. KDL gravity torques agree with MuJoCo qfrc_bias at home pose within 1e-3 Nm.
 *   4. 500-step gravity-comp loop: EE drift < 1 mm.
 *
 */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

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

class MenagerieTest : public ::testing::Test
{
  protected:
    fs::path                                         root_;
    std::string                                      mjcf_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    int                                              key_id_ = -1;
    unsigned                                         n_      = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;
    KDL::JntArray                                    q_home_kdl_;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found  - run locally with the submodule";
            return;
        }

        /* scene.xml includes gen3.xml and already has floor, lights, and skybox  -
         * no patching or temp file needed. */
        mjcf_ = (root_ / "third_party/menagerie/kinova_gen3/scene.xml").string();

        ASSERT_TRUE(mj_kdl::load_mjcf(&model_, &data_, mjcf_.c_str()));
        ASSERT_EQ(model_->nv, 7) << "expected nv=7";
        ASSERT_GE(model_->nbody, 9) << "expected nbody>=9";

        TEST_INFO("model: nq=" << model_->nq << " nv=" << model_->nv << " nbody=" << model_->nbody
                               << " nu=" << model_->nu);

        ASSERT_TRUE(mj_kdl::init_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"));
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, -9.81));

        /* Set home pose via keyframe if available, else set manually. */
        key_id_ = mj_name2id(model_, mjOBJ_KEY, "home");
        if (key_id_ >= 0) {
            mj_resetDataKeyframe(model_, data_, key_id_);
        } else {
            KDL::JntArray q_home(n_);
            for (unsigned i = 0; i < n_; ++i) q_home(i) = kHomePose[i];
            mj_kdl::set_joint_pos(&s_, q_home);
        }
        mj_forward(model_, data_);

        q_home_kdl_.resize(n_);
        mj_kdl::get_joint_pos(&s_, q_home_kdl_);
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(MenagerieTest, GravityAccuracy)
{
    KDL::JntArray g(n_);
    ASSERT_GE(dyn_->JntToGravity(q_home_kdl_, g), 0) << "JntToGravity() returned error";

    double max_err = 0.0;
    for (unsigned i = 0; i < n_; ++i)
        max_err = std::max(max_err, std::abs(g(i) - data_->qfrc_bias[s_.kdl_to_mj_dof[i]]));

    TEST_INFO("gravity accuracy at home pose: max|KDL - MuJoCo| = "
              << std::fixed << std::setprecision(6) << max_err << " Nm");

    EXPECT_LE(max_err, 1e-3) << "gravity error " << max_err << " Nm exceeds 1e-3 Nm threshold";
}

TEST_F(MenagerieTest, GravityCompDrift)
{
    KDL::Frame fk_initial;
    fk_->JntToCart(q_home_kdl_, fk_initial);

    int saved_flags = model_->opt.disableflags;
    model_->opt.disableflags |= mjDSBL_ACTUATION;

    s_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
    KDL::JntArray q(n_), g(n_);
    for (int i = 0; i < 500; ++i) {
        mj_kdl::update(&s_);
        for (unsigned j = 0; j < n_; ++j) q(j) = s_.jnt_pos_msr[j];
        dyn_->JntToGravity(q, g);
        for (unsigned j = 0; j < n_; ++j) s_.jnt_trq_cmd[j] = g(j);
        mj_kdl::step(&s_);
    }

    model_->opt.disableflags = saved_flags;

    KDL::JntArray q_end(n_);
    for (unsigned j = 0; j < n_; ++j) q_end(j) = s_.jnt_pos_msr[j];
    KDL::Frame fk_end;
    fk_->JntToCart(q_end, fk_end);
    double drift = (fk_initial.p - fk_end.p).Norm();

    TEST_INFO("gravity-comp drift after 500 steps: " << std::fixed << std::setprecision(3)
                                                     << drift * 1000.0 << " mm");

    EXPECT_LE(drift, 0.001) << "EE drift " << drift * 1000.0 << " mm exceeds 1 mm threshold";
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
