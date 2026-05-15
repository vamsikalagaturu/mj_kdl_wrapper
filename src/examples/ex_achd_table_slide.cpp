#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include "chainhdsolver_vereshchagin_fixed_joint.hpp"
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/kinfam_io.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kTableZ       = 0.447;
static constexpr double kMoveX        = 0.20;
static constexpr double kVMaxLin      = 0.08;
static constexpr double kTauMax       = 59.0;
static constexpr double kKpLin        = 200.0;
static constexpr double kKdLin        = 30.0;
static constexpr double kKpRot        = 175.0;
static constexpr double kKdRot        = 28.0;
static constexpr double kBetaMax      = 120.0;

static constexpr double kTablePose[7] = {
    -0.00258, 1.43, 3.14, -1.70, -0.018, 1.74, 1.57
};

static fs::path repo_root() { return fs::path(__FILE__).parent_path().parent_path().parent_path(); }
static double clamp_abs(double v, double limit) { return std::max(-limit, std::min(limit, v)); }

static void print_array(const char *label, const KDL::JntArray &x)
{
    std::cout << label << "=" << x << "\n";
}

static void fill_q_state(const mj_kdl::Robot &robot, KDL::JntArray &q, KDL::JntArray &qd)
{
    for (unsigned i = 0; i < q.rows(); ++i) {
        q(i)  = robot.jnt_pos_msr[i];
        qd(i) = robot.jnt_vel_msr[i];
    }
}

static void set_alpha_no_linear_z(KDL::Jacobian &alpha)
{
    alpha.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    alpha.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0)));
    alpha.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0)));
    alpha.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0)));
    alpha.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));
}

static void print_contact_heights(mjModel *model, mjData *data)
{
    for (const char *name : { "spherical_wrist_2_link", "bracelet_link" }) {
        KDL::Frame frame;
        if (mj_kdl::get_body_frame(model, data, name, &frame)) {
            std::cout << name << "_z_above_table=" << std::fixed << std::setprecision(4)
                      << frame.p.z() - kTableZ << "\n";
        }
    }
    KDL::Frame tcp;
    if (mj_kdl::get_site_frame(model, data, "g_pinch", &tcp)) {
        std::cout << "tcp_z_above_table=" << std::fixed << std::setprecision(4)
                  << tcp.p.z() - kTableZ << "\n";
    }
}

static bool control_step(
  mj_kdl::Robot                               &robot,
  KDL::ChainFkSolverPos_recursive             &fk_pos,
  KDL::ChainHdSolver_Vereshchagin_Fixed_Joint &achd,
  KDL::ChainIdSolver_RNE                      &rnea,
  const KDL::Frame                            &target,
  KDL::JntArray                               &q,
  KDL::JntArray                               &qd,
  KDL::JntArray                               &qdd,
  KDL::Jacobian                               &alpha,
  KDL::JntArray                               &beta,
  KDL::Wrenches                               &f_ext_achd,
  KDL::Wrenches                               &f_ext_rnea_zero,
  KDL::JntArray                               &ff_tau,
  KDL::JntArray                               &constraint_tau,
  KDL::JntArray                               &tau_cmd,
  std::array<double, 5>                       &err_prev,
  bool                                        &first_pid,
  bool                                         print_debug
)
{
    mj_kdl::update(&robot);
    fill_q_state(robot, q, qd);

    KDL::Frame current;
    fk_pos.JntToCart(q, current);
    const KDL::Twist err = KDL::diff(current, target);
    const double dt      = robot.model ? robot.model->opt.timestep : 0.002;

    const double e[5] = { err.vel.x(), err.vel.y(), err.rot.x(), err.rot.y(), err.rot.z() };
    if (first_pid) {
        for (unsigned i = 0; i < 5; ++i) err_prev[i] = e[i];
        first_pid = false;
    }

    const double de[5] = {
        (e[0] - err_prev[0]) / dt,
        (e[1] - err_prev[1]) / dt,
        (e[2] - err_prev[2]) / dt,
        (e[3] - err_prev[3]) / dt,
        (e[4] - err_prev[4]) / dt,
    };

    beta(0) = clamp_abs(kKpLin * e[0] + kKdLin * de[0], kBetaMax);
    beta(1) = clamp_abs(kKpLin * e[1] + kKdLin * de[1], kBetaMax);
    beta(2) = clamp_abs(kKpRot * e[2] + kKdRot * de[2], kBetaMax);
    beta(3) = clamp_abs(kKpRot * e[3] + kKdRot * de[3], kBetaMax);
    beta(4) = clamp_abs(kKpRot * e[4] + kKdRot * de[4], kBetaMax);
    for (unsigned i = 0; i < 5; ++i) err_prev[i] = e[i];

    KDL::SetToZero(ff_tau);
    if (achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext_achd, ff_tau, constraint_tau) < 0) return false;
    if (rnea.CartToJnt(q, qd, qdd, f_ext_rnea_zero, tau_cmd) < 0) return false;

    for (unsigned i = 0; i < q.rows(); ++i) robot.jnt_trq_cmd[i] = clamp_abs(tau_cmd(i), kTauMax);
    mj_kdl::update(&robot);

    if (print_debug) {
        print_array("beta_no_lin_z", beta);
        print_array("achd_qdd", qdd);
        print_array("achd_constraint_tau", constraint_tau);
        print_array("rnea_full_tau_cmd", tau_cmd);
    }
    return true;
}

int main(int argc, char **argv)
{
    bool headless = false;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--headless") headless = true;

    const fs::path root = repo_root();
    const std::string arm_mjcf   = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf   = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();
    const std::string table_mjcf = (root / "src/examples/assets/table.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = "bracelet_link";
    gripper.prefix    = "g_";
    gripper.pos[2]    = -0.061525;
    gripper.euler[0]  = 180.0;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = arm_mjcf.c_str();
    robot_spec.pos[2] = kTableZ;
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.robots.push_back(robot_spec);
    scene.objects.push_back(mj_kdl::SceneObject{
      .name      = "table",
      .mjcf_path = table_mjcf,
      .pos       = { 0.0, 0.0, kTableZ },
      .fixed     = true,
    });

    mjModel *model = nullptr;
    mjData  *data  = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene)) return 1;

    const mj_kdl::ToolFrameSpec tool{ .tool_body = "g_base", .tcp_site = "g_pinch" };
    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", &tool)) {
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    const unsigned n  = robot.chain.getNrOfJoints();
    const unsigned ns = robot.chain.getNrOfSegments();
    KDL::JntArray q_start(n);
    for (unsigned i = 0; i < n; ++i) q_start(i) = kTablePose[i];

    robot.ctrl_mode = mj_kdl::CtrlMode::TORQUE;

    mj_kdl::Env env;
    env.spec  = scene;
    env.model = model;
    env.data  = data;
    mj_kdl::env_add_robot(&env, &robot);
    env.on_reset = [&](mj_kdl::ResetContext *) { mj_kdl::set_joint_pos(&robot, q_start, false); };
    mj_kdl::reset(&env);

    // Let contacts settle with the table before starting the horizontal task.
    for (int i = 0; i < 300; ++i) mj_kdl::step(&robot);
    print_contact_heights(model, data);

    KDL::ChainFkSolverPos_recursive fk_pos(robot.chain);
    KDL::Frame target;
    mj_kdl::update(&robot);
    KDL::JntArray q(n), qd(n);
    fill_q_state(robot, q, qd);
    fk_pos.JntToCart(q, target);
    KDL::Frame tracked = target;
    target.p += KDL::Vector(kMoveX, 0.0, 0.0);

    KDL::Twist root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd(robot.chain, root_acc, 5);
    KDL::ChainIdSolver_RNE rnea(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));

    KDL::JntArray qdd(n), beta(5), ff_tau(n), constraint_tau(n), tau_cmd(n);
    KDL::Wrenches f_ext_achd(ns, KDL::Wrench::Zero());
    KDL::Wrenches f_ext_rnea_zero(ns, KDL::Wrench::Zero());
    KDL::Jacobian alpha(5);
    set_alpha_no_linear_z(alpha);

    // one-shot comparison: nc=6 (lin Z constrained) vs nc=5 (lin Z free)
    {
        mj_kdl::update(&robot);
        fill_q_state(robot, q, qd);
        KDL::Frame cmp_current;
        fk_pos.JntToCart(q, cmp_current);
        const KDL::Twist cmp_err = KDL::diff(cmp_current, target);

        KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd6(robot.chain, root_acc, 6);
        KDL::JntArray qdd6(n), beta6(6), ff6(n), ctau6(n), tau6(n);
        KDL::Jacobian alpha6(6);
        alpha6.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
        alpha6.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0)));
        alpha6.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0)));
        alpha6.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0)));
        alpha6.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0)));
        alpha6.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));
        beta6(0) = clamp_abs(kKpLin * cmp_err.vel.x(), kBetaMax);
        beta6(1) = clamp_abs(kKpLin * cmp_err.vel.y(), kBetaMax);
        beta6(2) = clamp_abs(kKpLin * cmp_err.vel.z(), kBetaMax);
        beta6(3) = clamp_abs(kKpRot * cmp_err.rot.x(), kBetaMax);
        beta6(4) = clamp_abs(kKpRot * cmp_err.rot.y(), kBetaMax);
        beta6(5) = clamp_abs(kKpRot * cmp_err.rot.z(), kBetaMax);
        KDL::SetToZero(ff6);
        achd6.CartToJnt(q, qd, qdd6, alpha6, beta6, f_ext_achd, ff6, ctau6);
        rnea.CartToJnt(q, qd, qdd6, f_ext_rnea_zero, tau6);
        std::cout << "\n--- nc=6 (lin Z constrained) ---\n";
        print_array("beta6", beta6);
        print_array("qdd6", qdd6);
        print_array("constraint_tau6", ctau6);
        print_array("tau_cmd6", tau6);

        KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd5(robot.chain, root_acc, 5);
        KDL::JntArray qdd5(n), beta5(5), ff5(n), ctau5(n), tau5(n);
        KDL::Jacobian alpha5(5);
        set_alpha_no_linear_z(alpha5);
        beta5(0) = clamp_abs(kKpLin * cmp_err.vel.x(), kBetaMax);
        beta5(1) = clamp_abs(kKpLin * cmp_err.vel.y(), kBetaMax);
        beta5(2) = clamp_abs(kKpRot * cmp_err.rot.x(), kBetaMax);
        beta5(3) = clamp_abs(kKpRot * cmp_err.rot.y(), kBetaMax);
        beta5(4) = clamp_abs(kKpRot * cmp_err.rot.z(), kBetaMax);
        KDL::SetToZero(ff5);
        achd5.CartToJnt(q, qd, qdd5, alpha5, beta5, f_ext_achd, ff5, ctau5);
        rnea.CartToJnt(q, qd, qdd5, f_ext_rnea_zero, tau5);
        std::cout << "\n--- nc=5 (lin Z free) ---\n";
        print_array("beta5", beta5);
        print_array("qdd5", qdd5);
        print_array("constraint_tau5", ctau5);
        print_array("tau_cmd5", tau5);
        std::cout << "\n";
    }

    std::array<double, 5> err_prev{};
    bool first_pid = true;
    int step_count = 0;
    double prev_sim_time = data->time;

    auto reset_scene = [&]() {
        mj_kdl::reset(&env);
        mj_kdl::update(&robot);
        fill_q_state(robot, q, qd);
        fk_pos.JntToCart(q, tracked);
        err_prev    = {};
        first_pid   = true;
        step_count  = 0;
        prev_sim_time = data->time;
    };

    auto step_control = [&]() {
        const double dt = robot.model->opt.timestep;
        const KDL::Vector to_goal = target.p - tracked.p;
        const double dist = to_goal.Norm();
        if (dist > 1e-4)
            tracked.p += (to_goal / dist) * std::min(dist, kVMaxLin * dt);
        const bool print_debug = headless && step_count == 0;
        bool ok = control_step(
          robot, fk_pos, achd, rnea, tracked, q, qd, qdd, alpha, beta, f_ext_achd,
          f_ext_rnea_zero, ff_tau, constraint_tau, tau_cmd, err_prev, first_pid,
          print_debug
        );
        ++step_count;
        return ok;
    };

    if (headless) {
        for (int i = 0; i < 2000; ++i) {
            if (!step_control() || !mj_kdl::step(&robot)) break;
        }
        mj_kdl::update(&robot);
        fill_q_state(robot, q, qd);
        KDL::Frame current;
        fk_pos.JntToCart(q, current);
        KDL::Twist err = KDL::diff(current, target);
        print_contact_heights(model, data);
        print_array("final_achd_constraint_tau", constraint_tau);
        print_array("final_rnea_full_tau_cmd", tau_cmd);
        std::cout << "tcp_xy_err_mm=" << std::fixed << std::setprecision(3)
                  << std::sqrt(err.vel.x() * err.vel.x() + err.vel.y() * err.vel.y()) * 1000.0
                  << " tcp_z_error_unconstrained_mm=" << err.vel.z() * 1000.0
                  << " tcp_rot_err_rad=" << err.rot.Norm() << "\n";
    } else {
        mj_kdl::Viewer viewer;
        if (!mj_kdl::init_window_sim(&viewer, &robot)) return 1;
        while (true) {
            if (data->time < prev_sim_time - 1e-6)
                reset_scene();
            prev_sim_time = data->time;
            if (!step_control() || !mj_kdl::step(&robot)) break;
        }
        mj_kdl::cleanup(&viewer);
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
