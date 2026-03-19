/* test_init.cpp
 * Loads the Kinova GEN3 URDF, runs 100 simulation steps, and verifies basic
 * model properties are consistent.
 * Usage: test_init [urdf_path] [--gui] */

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
    mj_kdl::Config cfg;
    mj_kdl::State  s;

    void SetUp() override {
        cfg.urdf_path  = g_urdf_path.c_str();
        cfg.base_link  = "base_link";
        cfg.tip_link   = "EndEffector_Link";
        cfg.headless   = true;

        ASSERT_TRUE(mj_kdl::init(&s, &cfg)) << "init() returned false";
    }

    void TearDown() override {
        mj_kdl::cleanup(&s);
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

static void run_gui(const std::string &urdf)
{
    mj_kdl::Config cfg;
    cfg.urdf_path  = urdf.c_str();
    cfg.base_link  = "base_link";
    cfg.tip_link = "EndEffector_Link";
    cfg.headless = true;

    mj_kdl::State s;
    if (!mj_kdl::init(&s, &cfg)) {
        std::cerr << "GUI: init() failed\n";
        return;
    }

    unsigned n = static_cast<unsigned>(s.n_joints);
    KDL::JntArray q_home(n);
    for (unsigned i = 0; i < n; ++i) q_home(i) = kHomePose[i];

    /* Reset to home pose before opening UI. */
    mj_kdl::sync_from_kdl(&s, q_home);
    mj_forward(s.model, s.data);
    for (unsigned i = 0; i < n; ++i) s.data->ctrl[i] = s.data->qpos[s.kdl_to_mj_qpos[i]];

    std::cout << "GUI mode — close window to exit\n";
    mj_kdl::run_simulate_ui(s.model, s.data, urdf.c_str());

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
