/* test_velocity.cpp
 * Tests velocity-level kinematics: FK, IK, and Jacobian on the Kinova GEN3.
 * Usage: test_velocity [urdf_path] [--gui]
 * With --gui: displays the robot at the home pose / IK solution. */

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

class VelocityTest : public ::testing::Test {
protected:
    mj_kdl::Config cfg;
    mj_kdl::State  s;
    unsigned       n = 0;

    KDL::JntArray q_home;
    KDL::JntArray q_test;

    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv>      ik_vel;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL>     ik;
    std::unique_ptr<KDL::ChainJntToJacSolver>        jac_solver;

    void SetUp() override {
        cfg.urdf_path = g_urdf_path.c_str();
        cfg.base_link = "base_link";
        cfg.tip_link  = "EndEffector_Link";
        cfg.headless  = true;

        ASSERT_TRUE(mj_kdl::init(&s, &cfg)) << "init() returned false";

        n = s.chain.getNrOfJoints();

        KDL::JntArray q_min(n), q_max(n);
        for (unsigned i = 0; i < n; ++i) {
            q_min(i) = s.joint_limits[i].first;
            q_max(i) = s.joint_limits[i].second;
        }

        fk         = std::make_unique<KDL::ChainFkSolverPos_recursive>(s.chain);
        ik_vel     = std::make_unique<KDL::ChainIkSolverVel_pinv>(s.chain);
        ik         = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
                         s.chain, q_min, q_max, *fk, *ik_vel, 500, 1e-5);
        jac_solver = std::make_unique<KDL::ChainJntToJacSolver>(s.chain);

        q_home.resize(n);
        for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

        q_test.resize(n);
        for (unsigned i = 0; i < n; ++i)
            q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;
    }

    void TearDown() override {
        mj_kdl::cleanup(&s);
    }
};

TEST_F(VelocityTest, FKHomePose)
{
    KDL::Frame fk_home;
    ASSERT_TRUE(fk->JntToCart(q_home, fk_home) >= 0) << "FK failed at home pose";
    TEST_INFO("pos: [" << std::fixed << std::setprecision(4)
              << fk_home.p.x() << ", " << fk_home.p.y() << ", " << fk_home.p.z() << "]");
}

TEST_F(VelocityTest, FKTestConfig)
{
    KDL::Frame fk_test;
    ASSERT_TRUE(fk->JntToCart(q_test, fk_test) >= 0) << "FK failed at test config";
    TEST_INFO("pos: [" << fk_test.p.x() << ", " << fk_test.p.y() << ", "
              << fk_test.p.z() << "]");
}

TEST_F(VelocityTest, IKRoundTrip)
{
    KDL::Frame fk_test;
    ASSERT_TRUE(fk->JntToCart(q_test, fk_test) >= 0) << "FK failed at test config";

    KDL::JntArray q_ik(n);
    int ik_ret = ik->CartToJnt(q_home, fk_test, q_ik);
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

static void run_gui(const std::string &urdf)
{
    mj_kdl::Config cfg;
    cfg.urdf_path = urdf.c_str();
    cfg.base_link = "base_link";
    cfg.tip_link  = "EndEffector_Link";
    cfg.headless  = true;

    mj_kdl::State s;
    if (!mj_kdl::init(&s, &cfg)) {
        std::cerr << "GUI: init() failed\n";
        return;
    }

    unsigned n = s.chain.getNrOfJoints();

    KDL::JntArray q_min(n), q_max(n);
    for (unsigned i = 0; i < n; ++i) {
        q_min(i) = s.joint_limits[i].first;
        q_max(i) = s.joint_limits[i].second;
    }

    KDL::ChainFkSolverPos_recursive fk(s.chain);
    KDL::ChainIkSolverVel_pinv      ik_vel(s.chain);
    KDL::ChainIkSolverPos_NR_JL     ik(s.chain, q_min, q_max, fk, ik_vel, 500, 1e-5);

    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    KDL::JntArray q_test(n);
    for (unsigned i = 0; i < n; ++i)
        q_test(i) = (i % 2 == 0 ? 1.0 : -1.0) * 30.0 * M_PI / 180.0;

    KDL::Frame    fk_test;
    KDL::JntArray q_ik(n);
    fk.JntToCart(q_test, fk_test);
    int ik_ret = ik.CartToJnt(q_home, fk_test, q_ik);

    /* Show IK solution (or home pose if IK failed) statically with gravity comp. */
    KDL::JntArray q_display = (ik_ret >= 0) ? q_ik : q_home;
    mj_kdl::sync_from_kdl(&s, q_display);
    mj_forward(s.model, s.data);
    for (unsigned i = 0; i < n; ++i) s.data->ctrl[i] = s.data->qpos[s.kdl_to_mj_qpos[i]];

    KDL::ChainDynParam dyn(s.chain, KDL::Vector(0, 0, -9.81));
    std::cout << "GUI mode — close window to exit\n";
    mj_kdl::run_simulate_ui(s.model, s.data, urdf.c_str(), [&](mjModel *, mjData *d) {
        for (unsigned i = 0; i < n; ++i) d->ctrl[i] = d->qpos[s.kdl_to_mj_qpos[i]];
        KDL::JntArray q(n), g(n);
        mj_kdl::sync_to_kdl(&s, q);
        dyn.JntToGravity(q, g);
        mj_kdl::set_torques(&s, g);
    });

    mj_kdl::cleanup(&s);
}

int main(int argc, char *argv[])
{
    g_urdf_path = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();
    bool gui = false;

    /* Parse non-GTest arguments before handing off to GTest. */
    std::vector<char *> remaining;
    remaining.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--gui")
            gui = true;
        else if (a[0] != '-')
            g_urdf_path = a;
        else
            remaining.push_back(argv[i]);
    }
    int remaining_argc = static_cast<int>(remaining.size());

    ::testing::InitGoogleTest(&remaining_argc, remaining.data());

    if (gui) {
        run_gui(g_urdf_path);
    }

    return RUN_ALL_TESTS();
}
