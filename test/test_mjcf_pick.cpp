/* test_mjcf_pick.cpp
 * Kinova GEN3 (MJCF) + Robotiq 2F-85: scripted pick-and-place of an orange cube.
 *
 * Tests:
 *   1. KDL chain has 7 joints.
 *   2. IK converges for pre-grasp, grasp, and lift waypoints (pos error < 2 mm each).
 *   3. Full pick sequence lifts the cube above 0.20 m. */

#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <kdl/chaindynparam.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace fs = std::filesystem;

static constexpr double kHomePose[7]  = { 0.0, 0.2618, 3.1416, -2.2689, 0.0, 0.9599, 1.5708 };
static constexpr double kCubeX        = 0.4;
static constexpr double kCubeY        = 0.0;
static constexpr double kCubeHS       = 0.02; // cube half-size [m]
static constexpr double kCubeZ        = kCubeHS;
static constexpr double kGripperReach = 0.18422; // bracelet_link -> pad tip [m]

static constexpr double kKp[7] = { 100, 200, 100, 200, 100, 200, 100 };
static constexpr double kKd[7] = { 10, 20, 10, 20, 10, 20, 10 };

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path(); }

static double clamp01(double v) { return std::max(0.0, std::min(1.0, v)); }

static void lerp_q(const KDL::JntArray &a, const KDL::JntArray &b, double t, KDL::JntArray &out)
{
    for (unsigned i = 0; i < a.rows(); ++i) out(i) = a(i) + t * (b(i) - a(i));
}

static void
  impedance_ctrl(mj_kdl::Robot &s, const KDL::JntArray &q_des, unsigned n, KDL::ChainDynParam &dyn)
{
    KDL::JntArray q(n), g(n);
    for (unsigned i = 0; i < n; ++i) q(i) = s.jnt_pos_msr[i];
    dyn.JntToGravity(q, g);
    for (unsigned i = 0; i < n; ++i) {
        s.jnt_trq_cmd[i] =
          g(i) + kKp[i] * (q_des(i) - s.jnt_pos_msr[i]) - kKd[i] * s.jnt_vel_msr[i];
    }
}

class MjcfPickTest : public testing::Test
{
  protected:
    fs::path root_;
    mjModel *model_ = nullptr;
    mjData  *data_  = nullptr;

    mj_kdl::Robot s_;
    int           fingers_act_ = -1;
    int           cube_jnt_    = -1;
    unsigned      n_           = 0;

    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
    std::unique_ptr<KDL::ChainDynParam>              dyn_;
    std::unique_ptr<KDL::ChainIkSolverVel_pinv>      ik_vel_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR_JL>     ik_;

    KDL::JntArray q_home_, q_pregrasp_, q_grasp_, q_lift_;

    void SetUp() override
    {
        root_ = repo_root();
        if (!fs::exists(root_ / "third_party/menagerie")) {
            GTEST_SKIP() << "third_party/menagerie/ not found";
            return;
        }

        const std::string arm_mjcf =
          (root_ / "third_party/menagerie/kinova_gen3/gen3.xml").string();
        const std::string grp_mjcf =
          (root_ / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

        mj_kdl::AttachmentSpec gs;
        gs.mjcf_path = grp_mjcf.c_str();
        gs.attach_to = "bracelet_link";
        gs.prefix    = "g_";
        gs.pos[2]    = -0.061525;
        gs.euler[0]  = 180.0;

        mj_kdl::RobotSpec rs;
        rs.path = arm_mjcf.c_str();
        rs.attachments.push_back(gs);

        mj_kdl::SceneObject cube;
        cube.name    = "cube";
        cube.shape   = mj_kdl::Shape::BOX;
        cube.size[0] = cube.size[1] = cube.size[2] = kCubeHS;
        cube.pos[0]                                = kCubeX;
        cube.pos[1]                                = kCubeY;
        cube.pos[2]                                = kCubeZ;
        cube.rgba[0]                               = 1.0f;
        cube.rgba[1]                               = 0.5f;
        cube.rgba[2]                               = 0.0f;
        cube.rgba[3]                               = 1.0f;
        cube.mass                                  = 0.1;
        cube.condim                                = 4;
        cube.friction[0]                           = 0.8;
        cube.friction[1]                           = 0.02;
        cube.friction[2]                           = 0.001;

        mj_kdl::SceneSpec sc;
        sc.robots.push_back(rs);
        sc.objects.push_back(cube);

        ASSERT_TRUE(mj_kdl::build_scene(&model_, &data_, &sc));
        ASSERT_TRUE(
          mj_kdl::init_robot_from_mjcf(
            &s_, model_, data_, "base_link", "bracelet_link", "", "g_base"
          )
        );

        n_ = s_.chain.getNrOfJoints();
        ASSERT_EQ(n_, 7u);

        fingers_act_ = mj_name2id(model_, mjOBJ_ACTUATOR, "g_fingers_actuator");
        ASSERT_GE(fingers_act_, 0) << "g_fingers_actuator not found";
        cube_jnt_ = mj_name2id(model_, mjOBJ_JOINT, "cube_joint");
        ASSERT_GE(cube_jnt_, 0) << "cube_joint not found";

        // IK setup.
        KDL::JntArray q_min(n_), q_max(n_);
        for (unsigned i = 0; i < n_; ++i) {
            int jid = model_->dof_jntid[s_.kdl_to_mj_dof[i]];
            if (model_->jnt_limited[jid]) {
                q_min(i) = model_->jnt_range[2 * jid];
                q_max(i) = model_->jnt_range[2 * jid + 1];
            } else {
                q_min(i) = -2 * M_PI;
                q_max(i) = 2 * M_PI;
            }
        }
        fk_     = std::make_unique<KDL::ChainFkSolverPos_recursive>(s_.chain);
        dyn_    = std::make_unique<KDL::ChainDynParam>(s_.chain, KDL::Vector(0, 0, -9.81));
        ik_vel_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(s_.chain);
        ik_     = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
          s_.chain, q_min, q_max, *fk_, *ik_vel_, 500, 1e-5
        );

        q_home_.resize(n_);
        for (unsigned i = 0; i < n_; ++i) q_home_(i) = kHomePose[i];

        const double        kGraspZ    = kCubeZ + kGripperReach + 0.02;
        const double        kPreGraspZ = kGraspZ + 0.20;
        const double        kLiftZ     = kGraspZ + 0.30;
        const KDL::Rotation kDownRot   = KDL::Rotation::Identity();

        q_pregrasp_.resize(n_);
        q_grasp_.resize(n_);
        q_lift_.resize(n_);

        struct WP
        {
            double               z;
            KDL::JntArray       *out;
            const KDL::JntArray *seed;
        };
        WP wps[] = {
            { kPreGraspZ, &q_pregrasp_, &q_home_ },
            { kGraspZ, &q_grasp_, &q_pregrasp_ },
            { kLiftZ, &q_lift_, &q_grasp_ },
        };
        for (auto &wp : wps) {
            KDL::Frame tgt(kDownRot, KDL::Vector(kCubeX, kCubeY, wp.z));
            int        ret = ik_->CartToJnt(*wp.seed, tgt, *wp.out);
            if (ret < 0)
                TEST_WARN("IK warning: did not converge for z=" << wp.z << " (ret=" << ret << ")");
        }
    }

    void TearDown() override
    {
        if (model_) {
            mj_kdl::cleanup(&s_);
            mj_kdl::destroy_scene(model_, data_);
        }
    }

    // Reset arm to home and cube to initial position.
    void reset_scene()
    {
        mj_resetData(model_, data_);
        mj_kdl::set_joint_pos(&s_, q_home_, false);
        int qadr              = model_->jnt_qposadr[cube_jnt_];
        data_->qpos[qadr]     = kCubeX;
        data_->qpos[qadr + 1] = kCubeY;
        data_->qpos[qadr + 2] = kCubeZ;
        data_->qpos[qadr + 3] = 1.0;
        data_->qpos[qadr + 4] = data_->qpos[qadr + 5] = data_->qpos[qadr + 6] = 0.0;
        mj_forward(model_, data_);
        s_.ctrl_mode = mj_kdl::CtrlMode::TORQUE;
        for (unsigned i = 0; i < n_; ++i) {
            s_.jnt_pos_cmd[i]                        = data_->qpos[s_.kdl_to_mj_qpos[i]];
            s_.jnt_trq_cmd[i]                        = 0.0;
            data_->qfrc_applied[s_.kdl_to_mj_dof[i]] = 0.0;
        }
        mj_kdl::update(&s_);
        data_->ctrl[fingers_act_] = 0.0;
    }

    /*
     * Run an impedance trajectory from q_enter to q_target over duration seconds.
     * Returns when duration elapses or settle_tol is met (negative = skip pose check).
     */
    void run_phase(
      const KDL::JntArray &q_enter,
      const KDL::JntArray &q_target,
      double               duration,
      double               timeout,
      double               settle_tol,
      double               gripper_cmd
    )
    {
        double        t_enter = data_->time;
        KDL::JntArray q_des(n_);

        while (true) {
            mj_kdl::update(&s_); // TORQUE: applies old jnt_trq_cmd, reads sensors
            double alpha = clamp01((data_->time - t_enter) / duration);
            lerp_q(q_enter, q_target, alpha, q_des);
            impedance_ctrl(s_, q_des, n_, *dyn_); // writes jnt_trq_cmd
            // Apply current torques immediately (same step, no delay).
            for (unsigned i = 0; i < n_; ++i)
                data_->qfrc_applied[s_.kdl_to_mj_dof[i]] = s_.jnt_trq_cmd[i];
            data_->ctrl[fingers_act_] = gripper_cmd;
            mj_step(model_, data_);

            double t_rel     = data_->time - t_enter;
            bool   done_time = t_rel >= duration;
            bool   done_pose = (settle_tol < 0.0);
            if (!done_pose) {
                double max_err = 0.0;
                for (unsigned i = 0; i < n_; ++i)
                    max_err = std::max(max_err, std::abs(q_target(i) - s_.jnt_pos_msr[i]));
                done_pose = (max_err <= settle_tol);
            }
            if ((done_time && done_pose) || t_rel >= timeout) break;
        }
    }
};

TEST_F(MjcfPickTest, KDLChain)
{
    TEST_INFO("arm KDL chain: " << n_ << " joints");
    EXPECT_EQ(n_, 7u);
}

TEST_F(MjcfPickTest, IKConvergence)
{
    const double        kMaxPosErr = 2e-3; // 2 mm
    const KDL::Rotation kDownRot   = KDL::Rotation::Identity();

    const double kGraspZ    = kCubeZ + kGripperReach + 0.02;
    const double kPreGraspZ = kGraspZ + 0.20;
    const double kLiftZ     = kGraspZ + 0.30;

    struct WP
    {
        const char          *label;
        double               z;
        const KDL::JntArray *q;
    };
    WP wps[] = {
        { "pre-grasp", kPreGraspZ, &q_pregrasp_ },
        { "grasp", kGraspZ, &q_grasp_ },
        { "lift", kLiftZ, &q_lift_ },
    };
    for (auto &wp : wps) {
        KDL::Frame tgt(kDownRot, KDL::Vector(kCubeX, kCubeY, wp.z));
        KDL::Frame fk_out;
        fk_->JntToCart(*wp.q, fk_out);
        double pos_err = (tgt.p - fk_out.p).Norm();
        TEST_INFO(
          "IK " << wp.label << " pos error: " << std::fixed << std::setprecision(3)
                << pos_err * 1000.0 << " mm"
        );
        EXPECT_LE(pos_err, kMaxPosErr) << wp.label << " IK error " << pos_err * 1000.0 << " mm";
    }
}

TEST_F(MjcfPickTest, CubeLifted)
{
    reset_scene(); // sets ctrl_mode = TORQUE

    KDL::JntArray q_enter(n_);

    auto snap = [&]() {
        for (unsigned i = 0; i < n_; ++i) q_enter(i) = s_.jnt_pos_msr[i];
    };

    snap();
    run_phase(q_enter, q_home_, 1.0, 2.5, 0.08, 0.0); // HOME
    snap();
    run_phase(q_enter, q_pregrasp_, 2.0, 4.0, 0.08, 0.0); // PREGRASP
    snap();
    run_phase(q_enter, q_grasp_, 2.0, 4.0, 0.06, 0.0); // GRASP
    snap();
    run_phase(q_enter, q_grasp_, 1.5, 2.5, -1.0, 255.0); // CLOSE
    snap();
    run_phase(q_enter, q_lift_, 3.0, 5.0, 0.08, 255.0); // LIFT
    run_phase(q_lift_, q_lift_, 1.0, 1.0, -1.0, 255.0); // HOLD

    int    qadr   = model_->jnt_qposadr[cube_jnt_];
    double cube_z = data_->qpos[qadr + 2];
    TEST_INFO("cube Z after pick: " << std::fixed << std::setprecision(3) << cube_z << " m");
    EXPECT_GT(cube_z, 0.20) << "cube was not lifted (z=" << cube_z << " m)";
}

int main(int argc, char *argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
