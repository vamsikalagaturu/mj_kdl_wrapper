/* test_init.cpp
 * Loads the Kinova GEN3 URDF, runs 100 simulation steps, and verifies basic
 * model properties are consistent.
 * Usage: test_init [urdf_path] */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <filesystem>

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

namespace fs = std::filesystem;
static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

static std::string g_urdf_path;

class InitTest : public ::testing::Test {
protected:
    mjModel       *model_ = nullptr;
    mjData        *data_  = nullptr;
    mj_kdl::Robot  s;

    void SetUp() override {
        mj_kdl::SceneSpec sc;
        mj_kdl::SceneRobot r;
        r.urdf_path = g_urdf_path.c_str();
        sc.robots.push_back(r);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc)) << "build_scene() returned false";
        ASSERT_TRUE(mj_kdl::init_robot(&s, model_, data_, g_urdf_path.c_str(),
            "base_link", "EndEffector_Link")) << "init_robot() returned false";
    }

    void TearDown() override {
        mj_kdl::cleanup(&s);
        mj_kdl::destroy_scene(model_, data_);
    }
};

TEST_F(InitTest, BasicDOF)
{
    EXPECT_EQ(s.model->nq, 7) << "expected nq==7, got " << s.model->nq;
    EXPECT_EQ(s.model->nv, 7) << "expected nv==7, got " << s.model->nv;
    EXPECT_EQ(s.n_joints, 7) << "expected 7 KDL joints, got " << s.n_joints;
    TEST_INFO("nq=" << s.model->nq << " nv=" << s.model->nv
              << " kdl_joints=" << s.n_joints);
}

TEST_F(InitTest, SimulationAdvance)
{
    unsigned n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(s.model, s.data);

    const double t0 = s.data->time;
    mj_kdl::step_n(&s, 100);
    ASSERT_TRUE(s.data->time > t0) << "simulation time did not advance after 100 steps";
    TEST_INFO("sim_time=" << s.data->time);
}

int main(int argc, char *argv[])
{
    g_urdf_path = (repo_root() / "assets/gen3_urdf/GEN3_URDF_V12.urdf").string();

    for (int i = 1; i < argc; ++i)
        if (argv[i][0] != '-') g_urdf_path = argv[i];

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
