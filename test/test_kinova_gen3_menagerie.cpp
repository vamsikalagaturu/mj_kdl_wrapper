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
 * Usage: test_kinova_gen3_menagerie [--gui] */

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

class MenagerieTest : public ::testing::Test {
protected:
    fs::path  root_;
    std::string mjcf_;
    mjModel  *model_ = nullptr;
    mjData   *data_  = nullptr;
    mj_kdl::Robot s_;
    int  key_id_ = -1;
    unsigned n_  = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;
    KDL::JntArray q_home_kdl_;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found — run locally with the submodule";
            return;
        }

        /* scene.xml includes gen3.xml and already has floor, lights, and skybox —
         * no patching or temp file needed. */
        mjcf_ = (root_ / "third_party/menagerie/kinova_gen3/scene.xml").string();

        ASSERT_TRUE(mj_kdl::load_mjcf(&model_, &data_, mjcf_.c_str()));
        ASSERT_EQ(model_->nv, 7) << "expected nv=7";
        ASSERT_GE(model_->nbody, 9) << "expected nbody>=9";

        TEST_INFO("model: nq=" << model_->nq << " nv=" << model_->nv
                  << " nbody=" << model_->nbody << " nu=" << model_->nu);

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
            mj_kdl::sync_from_kdl(&s_, q_home);
        }
        mj_forward(model_, data_);

        q_home_kdl_.resize(n_);
        mj_kdl::sync_to_kdl(&s_, q_home_kdl_);
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
        max_err = std::max(max_err,
                           std::abs(g(i) - data_->qfrc_bias[s_.kdl_to_mj_dof[i]]));

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

    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q(n_), g(n_);
        mj_kdl::sync_to_kdl(&s_, q);
        dyn_->JntToGravity(q, g);
        mj_kdl::set_torques(&s_, g);
        mj_kdl::step(&s_);
    }

    model_->opt.disableflags = saved_flags;

    KDL::JntArray q_end(n_);
    mj_kdl::sync_to_kdl(&s_, q_end);
    KDL::Frame fk_end;
    fk_->JntToCart(q_end, fk_end);
    double drift = (fk_initial.p - fk_end.p).Norm();

    TEST_INFO("gravity-comp drift after 500 steps: "
              << std::fixed << std::setprecision(3) << drift * 1000.0 << " mm");

    EXPECT_LE(drift, 0.001) << "EE drift " << drift * 1000.0 << " mm exceeds 1 mm threshold";
}

static void run_gui(mjModel *model, mjData *data, mj_kdl::Robot &s, unsigned n,
                    KDL::ChainDynParam &dyn, const std::string &mjcf,
                    int key_id, const KDL::JntArray &q_home_kdl)
{
    if (key_id >= 0) {
        mj_resetDataKeyframe(model, data, key_id);
    } else {
        mj_kdl::sync_from_kdl(&s, q_home_kdl);
    }
    mj_forward(model, data);
    for (unsigned i = 0; i < n; ++i)
        data->ctrl[i] = data->qpos[model->jnt_qposadr[model->dof_jntid[s.kdl_to_mj_dof[i]]]];

    std::cout << "GUI: close window to exit\n";
    mj_kdl::run_simulate_ui(model, data, mjcf.c_str(), [&](mjModel *m, mjData *d) {
        for (unsigned i = 0; i < n; ++i)
            d->ctrl[i] = d->qpos[m->jnt_qposadr[m->dof_jntid[s.kdl_to_mj_dof[i]]]];
        KDL::JntArray q(n), g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
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
        const std::string mjcf = (root / "third_party/menagerie/kinova_gen3/scene.xml").string();
        mjModel *model = nullptr;
        mjData  *data  = nullptr;
        if (!mj_kdl::load_mjcf(&model, &data, mjcf.c_str())) {
            std::cerr << "GUI: load_mjcf() failed\n";
            return 1;
        }
        mj_kdl::Robot s;
        if (!mj_kdl::init_from_mjcf(&s, model, data, "base_link", "bracelet_link")) {
            std::cerr << "GUI: init_from_mjcf() failed\n";
            mj_kdl::destroy_scene(model, data);
            return 1;
        }
        unsigned n = s.chain.getNrOfJoints();
        KDL::ChainDynParam dyn(s.chain, KDL::Vector(0, 0, -9.81));
        int key_id = mj_name2id(model, mjOBJ_KEY, "home");
        KDL::JntArray q_home_kdl(n);
        if (key_id >= 0) {
            mj_resetDataKeyframe(model, data, key_id);
        } else {
            for (unsigned i = 0; i < n; ++i) q_home_kdl(i) = kHomePose[i];
            mj_kdl::sync_from_kdl(&s, q_home_kdl);
        }
        mj_forward(model, data);
        mj_kdl::sync_to_kdl(&s, q_home_kdl);
        run_gui(model, data, s, n, dyn, mjcf, key_id, q_home_kdl);
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model, data);
    }

    return RUN_ALL_TESTS();
}
