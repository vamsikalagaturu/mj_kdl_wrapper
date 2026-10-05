/* test_table_scene.cpp
 * The bundled table as an MJCF-backed SceneObject: its quat and prefix, a Kinova GEN3 standing on
 * its top site holding its pose under KDL gravity compensation, and runtime scene_add_object /
 * scene_remove_object. Self-skips when a bundled asset is missing. */

#include "mjkdl/mjkdl.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ex = mjkdl_examples;
namespace fs = std::filesystem;

static constexpr double kSurfaceZ = 0.7;

static mjkdl::SceneSpec empty_scene()
{
    mjkdl::SceneSpec sc;
    sc.timestep   = 0.002;
    sc.add_floor  = false;
    sc.add_skybox = false;
    return sc;
}

static mjkdl::SceneObject table(const std::string &path, const char *name, const char *prefix)
{
    mjkdl::SceneObject t;
    t.name      = name;
    t.mjcf_path = path;
    t.prefix    = prefix;
    t.fixed     = true;
    return t;
}

// A free primitive resting on the tabletop: a box of half-size half, or a sphere of radius half.
static mjkdl::SceneObject make_object(
  const char   *name,
  mjkdl::Shape shape,
  double        x,
  double        y,
  double        half,
  const float   rgb[3]
)
{
    mjkdl::SceneObject o;
    o.name  = name;
    o.shape = shape;
    for (int k = 0; k < 3; ++k) o.size[k] = shape == mjkdl::Shape::BOX ? half : 0.0;
    o.size[0] = half;
    o.pos[0]  = x;
    o.pos[1]  = y;
    o.pos[2]  = kSurfaceZ + half;
    for (int k = 0; k < 3; ++k) o.rgba[k] = rgb[k];
    o.rgba[3]     = 1.0f;
    o.mass        = shape == mjkdl::Shape::BOX ? 0.1 : 0.05;
    o.friction[0] = 0.8;
    o.friction[1] = 0.02;
    o.friction[2] = 0.001;
    return o;
}

TEST(SceneObjectTransform, PathBackedObjectAppliesQuat)
{
    const std::string table_mjcf = ex::find_asset("table.xml");
    if (!fs::exists(table_mjcf)) GTEST_SKIP() << "table.xml not found";
    mjkdl::SceneSpec   spec   = empty_scene();
    mjkdl::SceneObject turned = table(table_mjcf, "turned", "");
    // extrinsic XYZ euler (30, 40, 50) deg
    const double q[4] = {
        0.08080468869083995, 0.40219849353410964, 0.30337177447125957, 0.860042173697679
    };
    std::copy(q, q + 4, turned.quat);
    spec.objects.push_back(turned);

    mjkdl::Env env;
    ASSERT_TRUE(mjkdl::init_env(&env, &spec));
    KDL::Frame frame;
    ASSERT_TRUE(mjkdl::get_site_frame(&env, "table_top", &frame));
    const KDL::Vector y = frame.M * KDL::Vector(0.0, 1.0, 0.0);
    EXPECT_NEAR(y.x(), -0.456825992585671, 1e-9);
    EXPECT_NEAR(y.y(), 0.802872337479472, 1e-9);
    EXPECT_NEAR(y.z(), 0.383022221559489, 1e-9);
}

TEST(SceneObjectPrefix, NamesStayAsAuthoredUnlessAPrefixIsSet)
{
    const std::string table_mjcf = ex::find_asset("table.xml");
    if (!fs::exists(table_mjcf)) GTEST_SKIP() << "table.xml not found";
    KDL::Frame frame;

    mjkdl::SceneSpec one = empty_scene();
    one.objects           = { table(table_mjcf, "t1", "") };
    mjkdl::Env env;
    ASSERT_TRUE(mjkdl::init_env(&env, &one));
    EXPECT_TRUE(mjkdl::get_site_frame(&env, "table_top", &frame));
    mjkdl::cleanup(&env);

    mjkdl::SceneSpec twins = one;
    twins.objects           = { table(table_mjcf, "t1", ""), table(table_mjcf, "t2", "") };
    const mjkdl::Status s  = mjkdl::init_env(&env, &twins);
    EXPECT_FALSE(s);
    EXPECT_NE(s.error.find("SceneObject::prefix"), std::string::npos) << s.error;

    twins.objects = { table(table_mjcf, "t1", "a_"), table(table_mjcf, "t2", "b_") };
    ASSERT_TRUE(mjkdl::init_env(&env, &twins));
    EXPECT_TRUE(mjkdl::get_site_frame(&env, "a_table_top", &frame));
    EXPECT_TRUE(mjkdl::get_site_frame(&env, "b_table_top", &frame));
    mjkdl::cleanup(&env);
}

class TableSceneTest : public testing::Test
{
  protected:
    mjkdl::SceneSpec                   spec_;
    mjkdl::Env                         env_;
    mjkdl::Robot                       s_;
    std::unique_ptr<KDL::ChainDynParam> dyn_;
    const KDL::JntArray                 q_home_ = ex::home_q(7);

    void SetUp() override
    {
        const std::string mjcf       = ex::find_asset("kinova_gen3/gen3.xml");
        const std::string table_mjcf = ex::find_asset("table.xml");
        if (!fs::exists(mjcf)) GTEST_SKIP() << mjcf << " not found";
        if (!fs::exists(table_mjcf)) GTEST_SKIP() << "table.xml not found";

        // Free bodies must hang off the worldbody; the robot, with no freejoint, uses the site.
        const float red[3] = { 1.0f, 0.2f, 0.2f }, green[3] = { 0.2f, 1.0f, 0.2f },
                    blue[3] = { 0.2f, 0.2f, 1.0f }, orange[3] = { 1.0f, 0.55f, 0.0f },
                    purple[3]   = { 0.7f, 0.0f, 0.9f };
        const auto          box = mjkdl::Shape::BOX, sphere = mjkdl::Shape::SPHERE;
        mjkdl::SceneObject top = table(table_mjcf, "table", "");
        top.pos[2]              = kSurfaceZ;
        spec_                   = empty_scene();
        spec_.add_floor         = true;
        spec_.objects           = {
            top,
            make_object("red_cube", box, 0.35, 0.10, 0.03, red),
            make_object("green_cube", box, 0.35, -0.10, 0.03, green),
            make_object("blue_cube", box, 0.35, 0.30, 0.04, blue),
            make_object("orange_sphere", sphere, -0.20, 0.20, 0.035, orange),
            make_object("purple_sphere", sphere, -0.20, -0.20, 0.025, purple),
        };
        mjkdl::RobotSpec rs;
        rs.path      = mjcf;
        rs.attach_to = { mjkdl::AttachKind::Site, "table_top" };
        spec_.robots.push_back(rs);

        ASSERT_TRUE(mjkdl::init_env(&env_, &spec_));
        KDL::Frame world_T_table_top;
        ASSERT_TRUE(mjkdl::get_site_frame(&env_, "table_top", &world_T_table_top));
        EXPECT_NEAR(world_T_table_top.p.z(), kSurfaceZ, 1e-9);

        ASSERT_TRUE(mjkdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link"));
        dyn_ = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, spec_.gravity_z));
        mjkdl::set_joint_pos(&s_, q_home_);
    }
};

TEST_F(TableSceneTest, GravityCompDrift)
{
    KDL::ChainFkSolverPos_recursive fk(s_.chain);
    KDL::Frame                      ee_init;
    fk.JntToCart(q_home_, ee_init);

    ASSERT_TRUE(mjkdl::set_control_mode(&s_, mjkdl::CtrlMode::TORQUE));
    ex::prime_gravity(s_, *dyn_, q_home_);
    for (int i = 0; i < 500; ++i) {
        mjkdl::update(&env_);
        ex::pd_gravity(s_, *dyn_, q_home_);
        mjkdl::step(&env_);
    }
    mjkdl::update(&env_);
    EXPECT_LE((ex::tcp_frame(fk, s_).p - ee_init.p).Norm(), 1e-6) << "measured 7e-19 m";
}

TEST_F(TableSceneTest, EnvAddRemoveReinitsRobot)
{
    const int nq_before = env_.model->nq;

    // Objects are compiled ahead of the robot, so each rebuild moves every robot address.
    auto expect_maps_match_model = [&] {
        for (int i = 0; i < s_.n_joints; ++i) {
            const int jid = mj_name2id(env_.model, mjOBJ_JOINT, s_.joint_names[i].c_str());
            ASSERT_GE(jid, 0);
            env_.data->qpos[env_.model->jnt_qposadr[jid]] = 0.1 * (i + 1);
        }
        mjkdl::update(&env_);
        for (int i = 0; i < s_.n_joints; ++i) EXPECT_DOUBLE_EQ(s_.jnt_pos_msr[i], 0.1 * (i + 1));
    };

    mjkdl::SceneFreeBodySlot *red = mjkdl::bind_scene_free_body(&env_.scene, "red_cube");
    ASSERT_NE(red, nullptr);

    KDL::Frame yellow_frame;
    EXPECT_FALSE(mjkdl::get_body_frame(&env_, "yellow_cube", &yellow_frame));

    const float yellow[3] = { 1.0f, 1.0f, 0.0f };
    ASSERT_TRUE(mjkdl::scene_add_object(
      &env_, make_object("yellow_cube", mjkdl::Shape::BOX, 0.0, 0.4, 0.03, yellow)
    ));
    EXPECT_GT(env_.model->nq, nq_before) << "nq should grow after adding a free object";
    EXPECT_EQ(s_.n_joints, 7) << "robot chain should still have 7 joints after rebuild";
    EXPECT_EQ(s_.model, env_.model) << "robot model pointer should be updated";
    expect_maps_match_model();
    // A name looked up on the freed model must not answer for the new one at the same address.
    EXPECT_TRUE(mjkdl::get_body_frame(&env_, "yellow_cube", &yellow_frame));

    mjkdl::SceneFreeBodySlot *slot = mjkdl::bind_scene_free_body(&env_.scene, "yellow_cube");
    ASSERT_NE(slot, nullptr);

    // The object is removed even though a slot bound to it goes unbound with it.
    EXPECT_TRUE(mjkdl::scene_remove_object(&env_, "yellow_cube"));
    EXPECT_EQ(env_.model->nq, nq_before) << "nq should return to original after removal";
    EXPECT_EQ(s_.n_joints, 7) << "robot chain should still have 7 joints after removal";
    expect_maps_match_model();

    EXPECT_EQ(slot->qpos_adr, -1) << "yellow_cube is gone";
    const int red_bid = mj_name2id(env_.model, mjOBJ_BODY, "red_cube");
    EXPECT_EQ(red->qpos_adr, env_.model->jnt_qposadr[env_.model->body_jntadr[red_bid]]);
    mjkdl::update(&env_);
    EXPECT_GT(red->seq, 0u);
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
