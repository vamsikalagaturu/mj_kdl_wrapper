/* test_gravity_comp.cpp
 * Two-part gravity compensation test using KDL dynamics:
 *
 * GravityAccuracy  - KDL vs MuJoCo accuracy at home pose:
 *   Compares KDL::ChainDynParam::JntToGravity against MuJoCo qfrc_bias.
 *   Checks |kdl_g - mujoco_bias| < 1e-3 Nm for all joints.
 *
 * GravityCompDrift  - KDL gravity comp drift test:
 *   Sets arm to home pose, then runs 500 steps applying KDL-computed gravity
 *   torques via ChainDynParam::JntToGravity each step.  EE drift must stay < 1 mm.
 *
 * Usage: test_gravity_comp [urdf_path] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

/* urdf_ is set once from main() before RUN_ALL_TESTS(). */
static std::string g_urdf;


class GravityCompTest : public ::testing::Test
{
  protected:
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk;
    std::unique_ptr<KDL::ChainDynParam>              dyn;
    KDL::JntArray                                    q_home;
    unsigned                                         n = 0;

    void SetUp() override
    {
        mj_kdl::SceneSpec sc;
        mj_kdl::RobotSpec r;
        r.path = g_urdf.c_str();
        sc.robots.push_back(r);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc)) << "build_scene() returned false";
        ASSERT_TRUE(
          mj_kdl::init_robot(&s, model_, data_, g_urdf.c_str(), "base_link", "EndEffector_Link"))
          << "init_robot() returned false";

        n   = static_cast<unsigned>(s.n_joints);
        fk  = std::make_unique<KDL::ChainFkSolverPos_recursive>(s.chain);
        dyn = std::make_unique<KDL::ChainDynParam>(s.chain, KDL::Vector(0.0, 0.0, -9.81));

        q_home.resize(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    }

    void TearDown() override
    {
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(GravityCompTest, GravityAccuracy)
{
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(s.model, s.data);

    KDL::JntArray g(n);
    ASSERT_GE(dyn->JntToGravity(q_home, g), 0) << "JntToGravity returned error";

    double max_err = 0.0;
    for (unsigned i = 0; i < n; ++i)
        max_err = std::max(max_err, std::abs(g(i) - s.data->qfrc_bias[s.kdl_to_mj_dof[i]]));

    TEST_INFO("max|KDL - qfrc_bias| = " << std::fixed << std::setprecision(6) << max_err << " Nm");

    ASSERT_LE(max_err, 1e-3) << "gravity error " << max_err << " Nm exceeds 0.001 Nm threshold";
}

TEST_F(GravityCompTest, GravityCompDrift)
{
    /* Sync to home pose and record initial FK frame. */
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(s.model, s.data);

    KDL::Frame fk_initial;
    fk->JntToCart(q_home, fk_initial);

    for (int i = 0; i < 500; ++i) {
        KDL::JntArray q, g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn->JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
        mj_kdl::step(&s);
    }

    KDL::JntArray q_end;
    mj_kdl::sync_to_kdl(&s, q_end);
    KDL::Frame fk_end;
    fk->JntToCart(q_end, fk_end);
    double drift = (fk_initial.p - fk_end.p).Norm();

    TEST_INFO("EE drift after 500 steps: " << std::setprecision(3) << drift * 1000.0 << " mm");

    ASSERT_LE(drift, 0.001) << "EE drift " << drift * 1000.0 << " mm exceeds 1 mm threshold";
}

int main(int argc, char *argv[])
{
    g_urdf = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();

    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-') g_urdf = argv[i];

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
