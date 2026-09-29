/* test_mjcf_load.cpp
 * Building scenes: the Kinova GEN3 alone and with a Robotiq 2F-85 (KDL chain vs the MuJoCo
 * frames, joint limits, saving the model), scene cameras and sites, contact exclusions, attaching
 * to a body or a named frame, a relative meshdir, and joints a chain must refuse. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chainfksolverpos_recursive.hpp>

#include <cmath>
#include <filesystem>
#include <string>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

static mj_kdl::SceneSpec bare_scene()
{
    mj_kdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = false;
    sc.add_skybox = false;
    return sc;
}

// The frame of a body or site expressed in base_link's frame.
static KDL::Frame in_base(mj_kdl::Env &env, const KDL::Frame &world_T_x)
{
    KDL::Frame world_T_base;
    mj_kdl::get_body_frame(&env, "base_link", &world_T_base);
    return world_T_base.Inverse() * world_T_x;
}

// The arm alone, from Menagerie's scene.xml (floor, lights and skybox of its own).
class MjcfLoadTest : public testing::Test
{
  protected:
    mj_kdl::Env   env_;
    mj_kdl::Robot s_;

    void SetUp() override
    {
        const std::string mjcf = ex::find_menagerie_model("kinova_gen3/scene.xml");
        if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";
        mj_kdl::SceneSpec sc = bare_scene();
        mj_kdl::RobotSpec rs;
        rs.path = mjcf;
        sc.robots.push_back(rs);
        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        ASSERT_EQ(env_.model->nv, 7);
        ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));
        ASSERT_EQ(s_.n_joints, 7);
        mj_kdl::set_joint_pos(&s_, ex::home_q(7));
    }
};

TEST_F(MjcfLoadTest, ChainFkMatchesTheModelFrames)
{
    KDL::ChainFkSolverPos_recursive fk(s_.chain);
    KDL::Frame                      fk_tip, world_T_tip;
    ASSERT_GE(fk.JntToCart(ex::home_q(7), fk_tip), 0);
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "bracelet_link", &world_T_tip));
    EXPECT_TRUE(KDL::Equal(fk_tip, in_base(env_, world_T_tip), 1e-9));
}

TEST_F(MjcfLoadTest, JointLimitsFollowTheModel)
{
    ASSERT_EQ(s_.joint_limits.size(), 7u);
    int unlimited = 0;
    for (int i = 0; i < 7; ++i) {
        const int jid = mj_name2id(env_.model, mjOBJ_JOINT, s_.joint_names[i].c_str());
        ASSERT_GE(jid, 0);
        if (env_.model->jnt_limited[jid]) {
            EXPECT_DOUBLE_EQ(s_.joint_limits[i].first, env_.model->jnt_range[2 * jid]);
            EXPECT_DOUBLE_EQ(s_.joint_limits[i].second, env_.model->jnt_range[2 * jid + 1]);
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
    const int bid = mj_name2id(env_.model, mjOBJ_BODY, "base_link");
    ASSERT_GE(bid, 0);
    env_.model->body_mass[bid] = 1.77;

    const fs::path path = fs::temp_directory_path() / "mj_kdl_save_model_xml_test.xml";
    ASSERT_TRUE(mj_kdl::save_model_xml(env_.model, path.c_str()));

    char     err[1000] = {};
    mjModel *loaded    = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
    fs::remove(path);
    ASSERT_NE(loaded, nullptr) << err;
    EXPECT_EQ(loaded->nq, env_.model->nq);
    EXPECT_EQ(loaded->nbody, env_.model->nbody);
    const int loaded_bid = mj_name2id(loaded, mjOBJ_BODY, "base_link");
    ASSERT_GE(loaded_bid, 0);
    EXPECT_DOUBLE_EQ(loaded->body_mass[loaded_bid], 1.77) << "runtime change was saved";
    mj_deleteModel(loaded);
}

// The arm with the bundled 2F-85 on its pinch site.
class MjcfGripperTest : public testing::Test
{
  protected:
    std::string       arm_;
    std::string       gripper_;
    mj_kdl::SceneSpec sc_;
    mj_kdl::Env       env_;

    void SetUp() override
    {
        arm_     = ex::find_menagerie_model("kinova_gen3/gen3.xml");
        gripper_ = ex::find_asset("robotiq_2f85/2f85.xml");
        if (!fs::exists(arm_)) GTEST_SKIP() << arm_ << " not found";
        if (!fs::exists(gripper_)) GTEST_SKIP() << gripper_ << " not found";
        sc_ = bare_scene();
        mj_kdl::RobotSpec rs;
        rs.path = arm_;
        rs.attachments.push_back(ex::gripper_attachment(gripper_));
        sc_.robots.push_back(rs);
    }

    int body(const char *name) const { return mj_name2id(env_.model, mjOBJ_BODY, name); }
};

TEST_F(MjcfGripperTest, TcpChainMatchesThePinchSite)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));
    const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
    mj_kdl::Robot               robot;
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot, &env_, "base_link", "bracelet_link", "", &tool)
    );
    ASSERT_EQ(robot.n_joints, 7);
    mj_kdl::set_joint_pos(&robot, ex::home_q(7));

    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::Frame                      fk_tcp, world_T_pinch;
    ASSERT_GE(fk.JntToCart(ex::home_q(7), fk_tcp), 0);
    ASSERT_TRUE(mj_kdl::get_site_frame(&env_, "g_pinch", &world_T_pinch));
    EXPECT_TRUE(KDL::Equal(fk_tcp, in_base(env_, world_T_pinch), 1e-9));
}

TEST_F(MjcfGripperTest, GripperDriverRangeAndCtrlrange)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));
    const int rdriver = mj_name2id(env_.model, mjOBJ_JOINT, "g_right_driver_joint");
    const int fingers = mj_name2id(env_.model, mjOBJ_ACTUATOR, "g_fingers_actuator");
    ASSERT_GE(rdriver, 0);
    ASSERT_GE(fingers, 0);
    EXPECT_DOUBLE_EQ(env_.model->jnt_range[2 * rdriver], 0.0);
    EXPECT_DOUBLE_EQ(env_.model->jnt_range[2 * rdriver + 1], 0.8);
    EXPECT_DOUBLE_EQ(env_.model->actuator_ctrlrange[2 * fingers + 1], ex::kGripperClosed);
}

TEST_F(MjcfGripperTest, JointSlotReadsAGripperJointByName)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));
    const int rdriver = mj_name2id(env_.model, mjOBJ_JOINT, "g_right_driver_joint");
    const int ldriver = mj_name2id(env_.model, mjOBJ_JOINT, "g_left_driver_joint");
    ASSERT_GE(rdriver, 0);
    ASSERT_GE(ldriver, 0);
    env_.data->qpos[env_.model->jnt_qposadr[rdriver]] = 0.42;
    env_.data->qpos[env_.model->jnt_qposadr[ldriver]] = 0.42;

    mj_kdl::SceneJointSlot *slot = mj_kdl::bind_scene_joint(&env_.scene, "g_right_driver_joint");
    ASSERT_NE(slot, nullptr);
    mj_kdl::update(&env_);
    EXPECT_NEAR(slot->position, 0.42, 1e-9);

    EXPECT_EQ(mj_kdl::bind_scene_joint(&env_.scene, "no_such_joint"), nullptr);
}

TEST_F(MjcfGripperTest, ContactExclusionsAreAdded)
{
    mjModel *plain = nullptr;
    mjData  *data  = nullptr;
    ASSERT_TRUE(mj_kdl::build_scene(&plain, &data, &sc_));
    const int nexclude = plain->nexclude;
    mj_kdl::destroy_scene(plain, data);

    sc_.robots[0].attachments[0].contact_exclusions = { { "bracelet_link", "g_base" } };
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));
    ASSERT_EQ(env_.model->nexclude, nexclude + 1);
    const int b1 = body("bracelet_link"), b2 = body("g_base");
    ASSERT_GE(b1, 0);
    ASSERT_GE(b2, 0);
    bool found = false;
    for (int i = 0; i < env_.model->nexclude; ++i) {
        const int sig = env_.model->exclude_signature[i];
        found |= sig == (b1 << 16) + b2 || sig == (b2 << 16) + b1;
    }
    EXPECT_TRUE(found) << "bracelet_link <-> g_base excluded";
}

TEST_F(MjcfGripperTest, AttachToABodyWithAnOffset)
{
    mj_kdl::AttachmentSpec &gripper = sc_.robots[0].attachments[0];
    gripper.attach_to               = { mj_kdl::AttachKind::Body, "bracelet_link" };
    gripper.pos[2]                  = 0.02;
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));

    const int mount = body("g_base_mount");
    ASSERT_GE(mount, 0);
    EXPECT_EQ(env_.model->body_parentid[mount], body("bracelet_link"));
    KDL::Frame world_T_bracelet, world_T_mount;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "bracelet_link", &world_T_bracelet));
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "g_base_mount", &world_T_mount));
    // The offset adds to the mount's own 7 mm in 2f85.xml.
    const KDL::Vector p = (world_T_bracelet.Inverse() * world_T_mount).p;
    EXPECT_NEAR(p.x(), 0.0, 1e-9);
    EXPECT_NEAR(p.y(), 0.0, 1e-9);
    EXPECT_NEAR(p.z(), 0.027, 1e-9);
}

TEST_F(MjcfGripperTest, CamerasAreAddedWhereTheSpecSays)
{
    mj_kdl::CameraSpec world_cam;
    world_cam.name   = "overview";
    world_cam.pos[0] = 1.0;
    world_cam.pos[1] = 0.0;
    world_cam.pos[2] = 1.5;
    // 90 deg about z, [x, y, z, w].
    world_cam.quat[2] = std::sqrt(0.5);
    world_cam.quat[3] = std::sqrt(0.5);
    world_cam.fovy    = 50.0;
    mj_kdl::CameraSpec wrist_cam;
    wrist_cam.name   = "mount_cam";
    wrist_cam.body   = "bracelet_link";
    wrist_cam.pos[0] = wrist_cam.pos[1] = 0.0;
    wrist_cam.pos[2]                    = 0.05;
    wrist_cam.fovy                      = 60.0;
    sc_.cameras                         = { world_cam, wrist_cam };
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));

    const mjModel *m = env_.model;
    const int      a = mj_name2id(m, mjOBJ_CAMERA, "overview");
    const int      b = mj_name2id(m, mjOBJ_CAMERA, "mount_cam");
    ASSERT_GE(a, 0);
    ASSERT_GE(b, 0);
    EXPECT_EQ(m->cam_bodyid[a], 0);
    EXPECT_EQ(m->cam_bodyid[b], body("bracelet_link"));
    EXPECT_DOUBLE_EQ(m->cam_fovy[a], 50.0);
    EXPECT_DOUBLE_EQ(m->cam_fovy[b], 60.0);
    EXPECT_DOUBLE_EQ(m->cam_pos[3 * a + 2], 1.5);
    EXPECT_DOUBLE_EQ(m->cam_pos[3 * b + 2], 0.05);
    // MuJoCo stores [w, x, y, z].
    EXPECT_NEAR(m->cam_quat[4 * a + 0], std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(m->cam_quat[4 * a + 3], std::sqrt(0.5), 1e-12);
}

TEST_F(MjcfGripperTest, SitesAreAddedButAnAuthoredSiteWins)
{
    mj_kdl::SiteSpec mark;
    mark.body    = "bracelet_link";
    mark.name    = "mark";
    mark.pos[2]  = 0.1;
    mark.quat[2] = std::sqrt(0.5);
    mark.quat[3] = std::sqrt(0.5);
    mj_kdl::SiteSpec clash;
    clash.body = "base_link";
    clash.name = "pinch_site";
    sc_.sites  = { mark, clash };
    ASSERT_TRUE(mj_kdl::init_env(&env_, &sc_));

    KDL::Frame world_T_bracelet, world_T_mark;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env_, "bracelet_link", &world_T_bracelet));
    ASSERT_TRUE(mj_kdl::get_site_frame(&env_, "mark", &world_T_mark));
    const KDL::Frame expected(KDL::Rotation::RotZ(M_PI / 2), KDL::Vector(0.0, 0.0, 0.1));
    EXPECT_TRUE(KDL::Equal(world_T_bracelet.Inverse() * world_T_mark, expected, 1e-9));

    const int pinch = mj_name2id(env_.model, mjOBJ_SITE, "pinch_site");
    ASSERT_GE(pinch, 0);
    EXPECT_NE(env_.model->site_bodyid[pinch], body("base_link"));
}

TEST(MjcfPathTest, RelativeModelPathWithRelativeMeshdir)
{
    const fs::path    model    = fs::path(MJ_KDL_TEST_FIXTURES) / "meshdir/mjcf/mesh_link.xml";
    const std::string relative = fs::relative(model).string();
    ASSERT_FALSE(fs::path(relative).is_absolute());

    mj_kdl::SceneSpec spec = bare_scene();
    mj_kdl::RobotSpec rs;
    rs.path = relative;
    spec.robots.push_back(rs);

    mjModel *m = nullptr;
    mjData  *d = nullptr;
    ASSERT_TRUE(mj_kdl::build_scene(&m, &d, &spec)) << relative;
    EXPECT_EQ(m->nmesh, 1);
    mj_kdl::destroy_scene(m, d);
}

class JointEdgeCaseTest : public testing::Test
{
  protected:
    mj_kdl::SceneSpec spec_ = bare_scene();
    mj_kdl::Env       env_;
    mj_kdl::Robot     robot_;

    void SetUp() override
    {
        mj_kdl::RobotSpec rs;
        rs.path  = std::string(MJ_KDL_TEST_FIXTURES) + "/joint_edge_cases.xml";
        rs.modes = {};
        spec_.robots.push_back(rs);
    }
};

TEST_F(JointEdgeCaseTest, ChainRefusesABodyWithTwoJoints)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
    EXPECT_FALSE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "two_base", "two_tip"));
    EXPECT_TRUE(env_.robots.empty());
}

TEST_F(JointEdgeCaseTest, ChainRefusesABallJointOnThePath)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
    EXPECT_FALSE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "ball_base", "ball_tip"));
}

TEST_F(JointEdgeCaseTest, PlainHingeChainStillBuilds)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
    ASSERT_TRUE(mj_kdl::init_robot_from_mjcf(&robot_, &env_, "plain_base", "plain_tip"));
    EXPECT_EQ(robot_.n_joints, 1);
}

TEST_F(JointEdgeCaseTest, JointSlotRefusesWhatIsNotAScalarJoint)
{
    ASSERT_TRUE(mj_kdl::init_env(&env_, &spec_));
    EXPECT_NE(mj_kdl::bind_scene_joint(&env_.scene, "plain"), nullptr);
    EXPECT_EQ(mj_kdl::bind_scene_joint(&env_.scene, "ball"), nullptr);
}

TEST(AttachToFrame, ARobotStandsOnAnObjectsNamedFrame)
{
    const std::string arm = ex::find_menagerie_model("kinova_gen3/gen3.xml");
    if (!fs::exists(arm)) GTEST_SKIP() << arm << " not found";

    // Objects are added before robots, so a robot can stand on an object's frame.
    mj_kdl::SceneSpec   spec = bare_scene();
    mj_kdl::SceneObject stand;
    stand.name      = "stand";
    stand.mjcf_path = std::string(MJ_KDL_TEST_FIXTURES) + "/joint_edge_cases.xml";
    stand.fixed     = true;
    spec.objects.push_back(stand);
    mj_kdl::RobotSpec rs;
    rs.path      = arm;
    rs.attach_to = { mj_kdl::AttachKind::Frame, "plain_frame" };
    rs.pos[0]    = 0.05;
    spec.robots.push_back(rs);

    mj_kdl::Env env;
    ASSERT_TRUE(mj_kdl::init_env(&env, &spec));
    const int base = mj_name2id(env.model, mjOBJ_BODY, "base_link");
    ASSERT_GE(base, 0);
    EXPECT_EQ(env.model->body_parentid[base], mj_name2id(env.model, mjOBJ_BODY, "plain_base"));
    // plain_base at (2, 0, 0); the frame 0.1 along y, 0.2 up, turned 90 deg about z.
    KDL::Frame world_T_base;
    ASSERT_TRUE(mj_kdl::get_body_frame(&env, "base_link", &world_T_base));
    EXPECT_TRUE(KDL::Equal(world_T_base.p, KDL::Vector(2.0, 0.15, 0.2), 1e-9));
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
