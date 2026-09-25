/* test_mjcf_load.cpp
 * Load Kinova GEN3 from MJCF and validate the KDL chain; also validates
 * the combined arm + Robotiq 2F-85 model. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7] = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };

/* -------------------------------------------------------------------------
 * Fixture 1: arm-only from scene.xml
 * ------------------------------------------------------------------------- */

class MjcfLoadTest : public testing::Test
{
  protected:
    fs::path                                         root_;
    mj_kdl::Env                                      env_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    unsigned                                         n_ = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    KDL::JntArray                                    q_home_;

    void SetUp() override
    {
        // scene.xml already has floor, lights, and skybox.
        std::string mjcf = mj_kdl_examples::find_menagerie_model("kinova_gen3/scene.xml");
        if (!fs::exists(mjcf)) {
            GTEST_SKIP() << mjcf << " not found";
            return;
        }

        mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
        sc.add_floor  = false;
        sc.add_skybox = false;
        sc.robots.push_back(mj_kdl::RobotSpec{ .path = mjcf.c_str(), .attachments = {} });

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        model_ = env_.model;
        data_  = env_.data;
        ASSERT_EQ(model_->nv, 7);
        ASSERT_GE(model_->nbody, 9);

        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));
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
        mj_kdl::update(&env_);

        q_home_.resize(n_);
        for (int i = 0; i < s_.n_joints; ++i) q_home_(i) = s_.jnt_pos_msr[i];
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

TEST_F(MjcfLoadTest, JointLimitsFollowTheModel)
{
    ASSERT_EQ(s_.joint_limits.size(), n_);
    int unlimited = 0;
    for (unsigned i = 0; i < n_; ++i) {
        const int jid = mj_name2id(model_, mjOBJ_JOINT, s_.joint_names[i].c_str());
        ASSERT_GE(jid, 0);
        if (model_->jnt_limited[jid]) {
            EXPECT_DOUBLE_EQ(s_.joint_limits[i].first, model_->jnt_range[2 * jid]);
            EXPECT_DOUBLE_EQ(s_.joint_limits[i].second, model_->jnt_range[2 * jid + 1]);
        } else {
            EXPECT_TRUE(std::isinf(s_.joint_limits[i].first) && s_.joint_limits[i].first < 0);
            EXPECT_TRUE(std::isinf(s_.joint_limits[i].second) && s_.joint_limits[i].second > 0);
            ++unlimited;
        }
    }
    EXPECT_GT(unlimited, 0) << "Gen3 has continuous joints";
}

TEST_F(MjcfLoadTest, SaveModelXmlRoundTrips)
{
    const int bid = mj_name2id(model_, mjOBJ_BODY, "base_link");
    ASSERT_GE(bid, 0);
    model_->body_mass[bid] = 1.77;

    const fs::path path = fs::temp_directory_path() / "mj_kdl_save_model_xml_test.xml";
    ASSERT_TRUE(mj_kdl::save_model_xml(model_, path.c_str()));

    char     err[1000] = {};
    mjModel *loaded    = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    fs::remove(path);
    ASSERT_NE(loaded, nullptr) << err;
    EXPECT_EQ(loaded->nq, model_->nq);
    EXPECT_EQ(loaded->nbody, model_->nbody);
    const int loaded_bid = mj_name2id(loaded, mjOBJ_BODY, "base_link");
    ASSERT_GE(loaded_bid, 0);
    EXPECT_DOUBLE_EQ(loaded->body_mass[loaded_bid], 1.77) << "runtime change was saved";
    mj_deleteModel(loaded);
}

/* -------------------------------------------------------------------------
 * Fixture 2: arm + Robotiq 2F-85 gripper from gen3.xml + 2f85.xml
 * ------------------------------------------------------------------------- */

class MjcfGripperTest : public testing::Test
{
  protected:
    fs::path                                         root_;
    mj_kdl::Env                                      env_;
    mjModel                                         *model_ = nullptr;
    mjData                                          *data_  = nullptr;
    mj_kdl::Robot                                    s_;
    unsigned                                         n_ = 0;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;

    void SetUp() override
    {
        const std::string arm_mjcf =
          mj_kdl_examples::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string grp_mjcf =
          mj_kdl_examples::find_asset("robotiq_2f85/2f85.xml");
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
            .attach_to          = { mj_kdl::AttachKind::Site, "pinch_site" },
            .prefix             = "g_",
            .contact_exclusions = {},
        };
        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf.c_str();
        rs.attachments.push_back(gs);

        mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = true;
    sc.add_skybox = true;
        sc.robots.push_back(rs);

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        model_ = env_.model;
        data_  = env_.data;
        ASSERT_GE(model_->nq, 13);
        ASSERT_GE(model_->nu, 8);

        const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link", "", &tool)
        );
        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);

        ASSERT_GE(mj_name2id(model_, mjOBJ_ACTUATOR, "g_fingers_actuator"), 0)
          << "g_fingers_actuator not found";
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

TEST_F(MjcfGripperTest, JointPositionByName)
{
    int rdriver = mj_name2id(model_, mjOBJ_JOINT, "g_right_driver_joint");
    ASSERT_GE(rdriver, 0) << "g_right_driver_joint not found";

    int ldriver = mj_name2id(model_, mjOBJ_JOINT, "g_left_driver_joint");
    ASSERT_GE(ldriver, 0) << "g_left_driver_joint not found";
    data_->qpos[model_->jnt_qposadr[rdriver]] = 0.42;
    data_->qpos[model_->jnt_qposadr[ldriver]] = 0.42;

    double measured = 0.0;
    ASSERT_TRUE(mj_kdl::get_joint_position(&env_, "g_right_driver_joint", &measured));
    EXPECT_NEAR(measured, 0.42, 1e-9);

    // An actuator name resolves to its transmission joint's qpos.
    measured = 0.0;
    ASSERT_TRUE(mj_kdl::get_joint_position(&env_, "g_fingers_actuator", &measured));
    EXPECT_NEAR(measured, 0.42, 1e-9);

    EXPECT_FALSE(mj_kdl::get_joint_position(&env_, "no_such_joint", &measured));
}

TEST(MjcfPathTest, RelativeModelPathWithRelativeMeshdir)
{
    const fs::path    model    = fs::path(MJ_KDL_TEST_FIXTURES) / "meshdir/mjcf/mesh_link.xml";
    const std::string relative = fs::relative(model).string();
    ASSERT_FALSE(fs::path(relative).is_absolute());

    mj_kdl::SceneSpec spec;
    spec.timestep   = 0.002;
    spec.add_floor  = false;
    spec.add_skybox = false;
    spec.robots.push_back(mj_kdl::RobotSpec{ .path = relative.c_str() });

    mjModel *m = nullptr;
    mjData  *d = nullptr;
    ASSERT_TRUE(mj_kdl::build_scene(&m, &d, &spec)) << relative;
    EXPECT_EQ(m->nmesh, 1);
    mj_kdl::destroy_scene(m, d);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
