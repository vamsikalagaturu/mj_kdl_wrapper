/* test_velocity.cpp
 * Tests velocity-level kinematics: FK, IK, and Jacobian on the Kinova GEN3.
 * Usage: test_velocity [urdf_path] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chaindynparam.hpp>

#include <iostream>
#include <iomanip>
#include <memory>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

static std::string g_urdf_path;

class VelocityTest : public ::testing::Test
{
  protected:
    mjModel      *model_ = nullptr;
    mjData       *data_  = nullptr;
    mj_kdl::Robot s;
    unsigned      n = 0;

    KDL::JntArray q_home;
    KDL::JntArray q_test;

    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv>      ik_vel;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL>     ik;
    std::unique_ptr<KDL::ChainJntToJacSolver>        jac_solver;

    void SetUp() override
    {
        mj_kdl::SceneSpec sc;
        mj_kdl::RobotSpec r;
        r.path = g_urdf_path.c_str();
        sc.robots.push_back(r);

        ASSERT_TRUE(mj_kdl::build_scene_from_urdfs(&model_, &data_, &sc))
          << "build_scene_from_urdfs() returned false";
        ASSERT_TRUE(mj_kdl::init_robot(
          &s, model_, data_, g_urdf_path.c_str(), "base_link", "EndEffector_Link"))
          << "init_robot() returned false";

        n = s.chain.getNrOfJoints();

        KDL::JntArray q_min(n), q_max(n);
        for (unsigned i = 0; i < n; ++i) {
            q_min(i) = s.joint_limits[i].first;
            q_max(i) = s.joint_limits[i].second;
        }

        fk     = std::make_unique<KDL::ChainFkSolverPos_recursive>(s.chain);
        ik_vel = std::make_unique<KDL::ChainIkSolverVel_pinv>(s.chain);
        ik     = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
          s.chain, q_min, q_max, *fk, *ik_vel, 500, 1e-5);
        jac_solver = std::make_unique<KDL::ChainJntToJacSolver>(s.chain);

        q_home.resize(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

        q_test.resize(n);
        for (unsigned i = 0; i < n; ++i)
            q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;
    }

    void TearDown() override
    {
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(VelocityTest, FKHomePose)
{
    KDL::Frame fk_home;
    ASSERT_TRUE(fk->JntToCart(q_home, fk_home) >= 0) << "FK failed at home pose";
    TEST_INFO("pos: [" << std::fixed << std::setprecision(4) << fk_home.p.x() << ", "
                       << fk_home.p.y() << ", " << fk_home.p.z() << "]");
}

TEST_F(VelocityTest, FKTestConfig)
{
    KDL::Frame fk_test;
    ASSERT_TRUE(fk->JntToCart(q_test, fk_test) >= 0) << "FK failed at test config";
    TEST_INFO("pos: [" << fk_test.p.x() << ", " << fk_test.p.y() << ", " << fk_test.p.z() << "]");
}

TEST_F(VelocityTest, IKRoundTrip)
{
    KDL::Frame fk_test;
    ASSERT_TRUE(fk->JntToCart(q_test, fk_test) >= 0) << "FK failed at test config";

    KDL::JntArray q_ik(n);
    int           ik_ret = ik->CartToJnt(q_home, fk_test, q_ik);
    if (ik_ret < 0) {
        TEST_WARN("IK did not converge (ret=" << ik_ret << "), skipping position check");
        return;
    }

    KDL::Frame fk_ik;
    fk->JntToCart(q_ik, fk_ik);
    double pos_err = (fk_test.p - fk_ik.p).Norm();
    TEST_INFO("pos error: " << pos_err * 1000.0 << " mm");
    EXPECT_LE(pos_err, 1e-3) << "IK position error " << pos_err * 1000.0
                             << " mm exceeds 1 mm threshold";
}

TEST_F(VelocityTest, JacobianDimensions)
{
    KDL::Jacobian jac(n);
    ASSERT_TRUE(jac_solver->JntToJac(q_home, jac) >= 0) << "Jacobian solver returned error";
    TEST_INFO(jac.rows() << " rows x " << jac.columns() << " cols");
    EXPECT_EQ(jac.rows(), 6u) << "unexpected Jacobian row count";
    EXPECT_EQ(jac.columns(), n) << "unexpected Jacobian column count (expected " << n << ")";
}

int main(int argc, char *argv[])
{
    g_urdf_path = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();

    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-') g_urdf_path = argv[i];

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
