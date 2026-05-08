/* test_mjcf_load.cpp
 * Load Kinova GEN3 from MJCF and validate the KDL chain; also validates
 * the combined arm + Robotiq 2F-85 model. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

/* -------------------------------------------------------------------------
 * Fixture 1: arm-only from scene.xml
 * ------------------------------------------------------------------------- */

class MjcfLoadTest : public testing::Test
{
  protected:
    fs::path                                         root_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    unsigned                                         n_ = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    KDL::JntArray                                    q_home_;

    void SetUp() override
    {
        root_ = repo_root();
        // scene.xml already has floor, lights, and skybox.
        std::string mjcf = (root_ / "third_party/menagerie/kinova_gen3/scene.xml").string();
        if (!fs::exists(mjcf)) {
            GTEST_SKIP() << mjcf << " not found";
            return;
        }

        mj_kdl::SceneSpec sc;
        sc.add_floor  = false;
        sc.add_skybox = false;
        sc.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf.c_str(), .attachments = {} });

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        ASSERT_EQ(model_->nv, 7);
        ASSERT_GE(model_->nbody, 9);

        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link"));
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);

        // Use keyframe if available, else set manually.
        int key_id = mj_name2id(model_, mjOBJ_KEY, "home");
        if (key_id >= 0) {
            mj_resetDataKeyframe(model_, data_, key_id);
        } else {
            KDL::JntArray q(n_);
            for (unsigned i = 0; i < n_; ++i) q(i) = kHomePose[i];
            mj_kdl::set_joint_pos(&s_, q);
        }
        mj_forward(model_, data_);

        q_home_.resize(n_);
        for (int i = 0; i < s_.n_joints; ++i) q_home_(i) = s_.data->qpos[s_.kdl_to_mj_qpos[i]];
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(MjcfLoadTest, ModelLoaded)
{
    EXPECT_EQ(model_->nv, 7);
    EXPECT_GE(model_->nbody, 9);
}

TEST_F(MjcfLoadTest, KDLChain)
{
    EXPECT_EQ(n_, 7u);
}

TEST_F(MjcfLoadTest, FKHomePose)
{
    KDL::Frame fk_home;
    ASSERT_GE(fk_->JntToCart(q_home_, fk_home), 0) << "FK failed at home pose";
    double dist = fk_home.p.Norm();
    EXPECT_GE(dist, 0.1);
    EXPECT_LE(dist, 1.1);
}

/* -------------------------------------------------------------------------
 * Fixture 2: arm + Robotiq 2F-85 gripper from gen3.xml + 2f85.xml
 * ------------------------------------------------------------------------- */

class MjcfGripperTest : public testing::Test
{
  protected:
    fs::path                                         root_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    unsigned                                         n_ = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;

    void SetUp() override
    {
        root_ = repo_root();
        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
        if (!fs::exists(arm_mjcf)) {
            GTEST_SKIP() << arm_mjcf << " not found";
            return;
        }
        if (!fs::exists(grp_mjcf)) {
            GTEST_SKIP() << grp_mjcf << " not found";
            return;
        }

        mj_kdl::AttachmentSpec gs{
            .mjcf_path          = grp_mjcf.c_str(),
            .attach_to          = "bracelet_link",
            .prefix             = "g_",
            .pos                = { 0.0, 0.0, -0.061525 },
            .euler              = { 180.0, 0.0, 0.0 },
            .contact_exclusions = {},
        };
        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf.c_str();
        rs.attachments.push_back(gs);

        mj_kdl::SceneSpec sc;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        ASSERT_GE(model_->nq, 13);
        ASSERT_GE(model_->nu, 8);

        const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(&s_, model_, data_, "base_link", "bracelet_link", "", &tool)
        );
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);

        ASSERT_GE(mj_name2id(model_, mjOBJ_ACTUATOR, "g_fingers_actuator"), 0)
          << "g_fingers_actuator not found";
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }
};

TEST_F(MjcfGripperTest, ModelLoaded)
{
    EXPECT_GE(model_->nq, 13);
    EXPECT_GE(model_->nu, 8);
}

TEST_F(MjcfGripperTest, KDLChain)
{
    EXPECT_EQ(n_, 7u);
}

TEST_F(MjcfGripperTest, FKWorkspace)
{
    KDL::JntArray q_home(n_);
    for (unsigned i = 0; i < n_; ++i) q_home(i) = kHomePose[i];
    KDL::Frame fk_pose;
    fk_->JntToCart(q_home, fk_pose);
    double ee_dist = fk_pose.p.Norm();
    EXPECT_GE(ee_dist, 0.1);
    EXPECT_LE(ee_dist, 1.1);
}

TEST_F(MjcfGripperTest, GripperRange)
{
    int rdriver = mj_name2id(model_, mjOBJ_JOINT, "g_right_driver_joint");
    ASSERT_GE(rdriver, 0) << "g_right_driver_joint not found";

    double lo = model_->jnt_range[2 * rdriver];
    double hi = model_->jnt_range[2 * rdriver + 1];
    EXPECT_LE(std::abs(hi - 0.8), 0.01);
    EXPECT_GE(lo, -0.01);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
