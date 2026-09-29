/* test_mjcf_pick.cpp
 * Kinova GEN3 + Robotiq 2F-85 picks a 4 cm cube off the floor with the examples' IK waypoints,
 * joint impedance and phase runner (src/examples/common.hpp). */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "common.hpp"
#include "example_paths.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace ex = mj_kdl_examples;
namespace fs = std::filesystem;

class MjcfPickTest : public testing::Test
{
  protected:
    mj_kdl::Env            env_;
    mj_kdl::Robot          s_;
    ex::PickPlaceWaypoints wp_;

    void SetUp() override
    {
        const std::string arm_mjcf = ex::find_menagerie_model("kinova_gen3/gen3.xml");
        const std::string grp_mjcf = ex::find_asset("robotiq_2f85/2f85.xml");
        if (!fs::exists(arm_mjcf)) GTEST_SKIP() << arm_mjcf << " not found";
        if (!fs::exists(grp_mjcf)) GTEST_SKIP() << grp_mjcf << " not found";

        // The arm stands on the floor, so the cube lies at the examples' base-frame pick spot.
        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf;
        rs.attachments.push_back(ex::gripper_attachment(grp_mjcf));
        mj_kdl::SceneSpec sc;
        sc.timestep   = 0.002;
        sc.add_floor  = true;
        sc.add_skybox = false;
        sc.robots.push_back(rs);
        sc.objects.push_back(ex::cube_object(ex::kPickXY[0], ex::kPickXY[1], 0.0));

        ASSERT_TRUE(mj_kdl::init_env(&env_, &sc));
        const mj_kdl::ToolFrameSpec tool = ex::gripper_tool();
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(&s_, &env_, "base_link", "bracelet_link", "", &tool)
        );
        ASSERT_TRUE(ex::solve_pick_place(s_, wp_));
    }
};

TEST_F(MjcfPickTest, TcpLiesBeyondTheWrist)
{
    mj_kdl::Robot wrist;
    ASSERT_TRUE(
      mj_kdl::init_robot_from_mjcf(&wrist, &env_, "base_link", "bracelet_link", "", nullptr)
    );
    EXPECT_EQ(wrist.chain.getNrOfJoints(), s_.chain.getNrOfJoints());

    KDL::ChainFkSolverPos_recursive wrist_fk(wrist.chain), tcp_fk(s_.chain);
    KDL::Frame                      wrist_frame, tcp_frame;
    ASSERT_GE(wrist_fk.JntToCart(wp_.home, wrist_frame), 0);
    ASSERT_GE(tcp_fk.JntToCart(wp_.home, tcp_frame), 0);
    // The 2F-85 pinch site is 21.7 cm out of the bracelet.
    EXPECT_NEAR((tcp_frame.p - wrist_frame.p).Norm(), 0.217325, 1e-6);
    mj_kdl::cleanup(&wrist);
}

TEST_F(MjcfPickTest, IkStaysOnTheSeedBranch)
{
    const std::vector<std::pair<const KDL::JntArray *, const KDL::JntArray *>> steps = {
        { &wp_.pick_above, &wp_.pick }, { &wp_.pick, &wp_.lift }
    };
    for (const auto &[from, to] : steps) {
        double delta = 0.0;
        for (unsigned i = 0; i < from->rows(); ++i)
            delta = std::max(delta, std::abs((*to)(i) - (*from)(i)));
        EXPECT_LT(delta, 1.5) << "IK jumped to a distant branch (measured 0.61, 0.92 rad)";
    }
}

TEST_F(MjcfPickTest, CubeLifted)
{
    ASSERT_TRUE(mj_kdl::set_control_mode(&s_, mj_kdl::CtrlMode::TORQUE));
    mj_kdl::SceneActuatorSlot *fingers =
      mj_kdl::bind_scene_actuator(&env_.scene, "g_fingers_actuator");
    mj_kdl::SceneFreeBodySlot *cube = mj_kdl::bind_scene_free_body(&env_.scene, "cube");
    ASSERT_NE(fingers, nullptr);
    ASSERT_NE(cube, nullptr);
    KDL::ChainDynParam dyn(s_.chain, KDL::Vector(0, 0, -9.81));
    env_.on_reset = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&s_, wp_.home);
        ex::prime_gravity(s_, dyn, wp_.home);
    };
    mj_kdl::reset(&env_);

    std::vector<ex::Phase> phases = ex::pick_place_phases(wp_);
    phases.resize(5); // HOME .. LIFT
    phases.push_back({ "HOLD", &wp_.lift, 1.0, 1.0, -1.0, ex::kGripperClosed });
    bool restart = false;
    ASSERT_TRUE(ex::run_phases(
      env_,
      { { &s_, fingers } },
      phases,
      restart,
      [&](std::size_t, const KDL::JntArray &q) { ex::pd_gravity(s_, dyn, q, ex::kKp, ex::kKd); }
    ));
    mj_kdl::update(&env_);
    EXPECT_GT(cube->pose.p.z(), 0.28) << "cube was not lifted (measured 0.312 m)";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
