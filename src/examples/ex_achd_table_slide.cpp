/* ex_achd_table_slide.cpp
 * Kinova GEN3, no gripper, slides its wrist flange (the TCP, the bracelet's pinch_site) 0.2 m
 * along +X on a table while pressing down
 * with kPressForce, under ACHD (KDL ChainHdSolver_Vereshchagin, linear Z left free) + RNEA.
 * The press goes through ACHD's external-force input with driver weights 1, the pinned KDL
 * fork's setDriverWeights(); the table's contact normal force is measured against it. Before
 * the press, a guarded approach (all six directions tracked, nc=6) lowers the TCP at a speed
 * that falls with its height, and the press ramps up once the table is touched.
 *
 * Usage:
 *   ex_achd_table_slide [--headless]
 *
 * Runs the slide once and prints the mean table reaction over its second half, the commanded
 * press and their ratio (not judged), plus a one-shot nc=6 vs nc=5 comparison; --headless
 * skips the viewer and exits 1 if the table contact was held for less than kContactHeld of
 * that window, the touchdown was faster than kMaxTouchdown or the elbow went below
 * kMinElbowHeight. */

#include "common.hpp"
#include "example_paths.hpp"

#include <kdl/chainhdsolver_vereshchagin.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/kinfam_io.hpp>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace ex = mj_kdl_examples;

static constexpr int    kApproachSteps  = 7500; // no contact within 15 s fails the run
static constexpr int    kPressSteps     = 250;  // the press ramps up over 0.5 s
static constexpr int    kSlideSteps     = 2000;
static constexpr double kTableZ         = 0.447;
static constexpr double kMoveX          = 0.20;
static constexpr double kVMaxLin        = 0.08; // [m/s]
static constexpr double kVTouch         = 0.02; // [m/s] slowest descent, the touchdown speed
static constexpr double kDescentGain    = 0.5;  // [1/s] descent speed per metre above the table
static constexpr double kKpLin          = 200.0;
static constexpr double kKdLin          = 30.0;
static constexpr double kKpRot          = 175.0;
static constexpr double kKdRot          = 28.0;
static constexpr double kBetaMax        = 120.0;
static constexpr double kPressForce     = 10.0; // [N] commanded straight down at the TCP
static constexpr double kContactHeld    = 0.6;  // the contact chatters while sliding
static constexpr double kMaxTouchdown   = 0.05; // [m/s] TCP vertical speed at first contact
static constexpr double kMinElbowHeight = 0.30; // [m] above the table
static constexpr double kNoTouchdown    = std::numeric_limits<double>::quiet_NaN();

static constexpr const char *kElbowBody = "forearm_link"; // its origin is the elbow joint
static constexpr const char *kTcpSite   = "pinch_site";   // the flange face, z out of the arm

// [m] TCP start in the base frame, flange down; the base stands on the table, so z is its height.
static constexpr double kStartTcp[3] = { 0.40, 0.0, 0.12 };

enum class Phase {
    Approach, // all six TCP directions tracked, the reference descends onto the table
    Press,    // linear Z free, the press ramps up in place
    Slide     // linear Z free under the full press, the reference moves kMoveX along +X
};

// Sum of contact normal forces on one geom; positive pushes the bodies apart.
static double contact_normal_force(const mjModel *model, mjData *data, int geom)
{
    double total = 0.0;
    for (int i = 0; i < data->ncon; ++i) {
        const mjContact &c = data->contact[i];
        if (c.geom1 != geom && c.geom2 != geom) continue;
        double f[6];
        mj_contactForce(model, data, i, f);
        total += f[0];
    }
    return total;
}

static void set_alpha_no_linear_z(KDL::Jacobian &alpha)
{
    alpha.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
    alpha.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0)));
    alpha.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0)));
    alpha.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0)));
    alpha.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));
}

static double elbow_height(mj_kdl::Env &env)
{
    KDL::Frame elbow;
    mj_kdl::get_body_frame(&env, kElbowBody, &elbow);
    return elbow.p.z() - kTableZ;
}

static bool touches(const mjData *data, int geom)
{
    for (int i = 0; i < data->ncon; ++i)
        if (data->contact[i].geom1 == geom || data->contact[i].geom2 == geom) return true;
    return false;
}

// Downward world-frame speed of a site.
static double site_sink_speed(const mjModel *model, const mjData *data, int site)
{
    double vel[6];
    mj_objectVelocity(model, data, mjOBJ_SITE, site, vel, 0);
    return -vel[5];
}

static void print_contact_heights(mj_kdl::Env &env)
{
    for (const char *name : { "spherical_wrist_2_link", "bracelet_link" }) {
        KDL::Frame frame;
        if (mj_kdl::get_body_frame(&env, name, &frame)) {
            std::cout << name << "_z_above_table=" << std::fixed << std::setprecision(4)
                      << frame.p.z() - kTableZ << "\n";
        }
    }
    KDL::Frame tcp;
    if (mj_kdl::get_site_frame(&env, kTcpSite, &tcp)) {
        std::cout << "tcp_z_above_table=" << std::fixed << std::setprecision(4)
                  << tcp.p.z() - kTableZ << "\n";
    }
}

// Prints qdd and the RNEA torque for the current error with nc task constraints (6 or 5).
static void print_nc_comparison(
  const mj_kdl::Robot    &robot,
  const KDL::Twist       &err,
  const KDL::JntArray    &q,
  const KDL::JntArray    &qd,
  const KDL::Twist       &root_acc,
  KDL::ChainIdSolver_RNE &rnea,
  unsigned                nc
)
{
    const unsigned                  n = robot.chain.getNrOfJoints();
    KDL::ChainHdSolver_Vereshchagin achd(robot.chain, root_acc, nc);
    KDL::JntArray                   qdd(n), beta(nc), ff(n), ctau(n), tau(n);
    KDL::Jacobian                   alpha(nc);
    const KDL::Wrenches             f_ext(robot.chain.getNrOfSegments(), KDL::Wrench::Zero());
    if (nc == 6) {
        for (unsigned i = 0; i < 6; ++i) alpha(i, i) = 1.0;
        const double e[6] = { err.vel.x(), err.vel.y(), err.vel.z(),
                              err.rot.x(), err.rot.y(), err.rot.z() };
        for (unsigned i = 0; i < 6; ++i)
            beta(i) = ex::clamp_abs((i < 3 ? kKpLin : kKpRot) * e[i], kBetaMax);
    } else {
        set_alpha_no_linear_z(alpha);
        const double e[5] = { err.vel.x(), err.vel.y(), err.rot.x(), err.rot.y(), err.rot.z() };
        for (unsigned i = 0; i < 5; ++i)
            beta(i) = ex::clamp_abs((i < 2 ? kKpLin : kKpRot) * e[i], kBetaMax);
    }
    achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff, ctau);
    rnea.CartToJnt(q, qd, qdd, f_ext, tau);
    std::cout << "\n--- nc=" << nc << (nc == 6 ? " (lin Z constrained)" : " (lin Z free)")
              << " ---\nbeta=" << beta << "\nqdd=" << qdd << "\nconstraint_tau=" << ctau
              << "\ntau_cmd=" << tau << "\n";
}

int main(int argc, char **argv)
{
    const bool headless = ex::parse_args(argc, argv).headless;

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path   = ex::menagerie_model("kinova_gen3/gen3.xml");
    robot_spec.pos[2] = kTableZ;

    mj_kdl::SceneSpec scene = ex::scene_spec();
    scene.robots.push_back(robot_spec);
    scene.objects.push_back(ex::table_object(ex::asset("table.xml"), kTableZ));

    mj_kdl::Env env;
    if (!mj_kdl::init_env(&env, &scene)) return 1;
    mj_kdl::ToolFrameSpec tool;
    tool.tcp_site = kTcpSite;
    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, &env, "base_link", "bracelet_link", "", &tool))
        return 1;
    const int table_geom = mj_name2id(env.model, mjOBJ_GEOM, "top");
    const int tcp_site   = mj_name2id(env.model, mjOBJ_SITE, kTcpSite);
    if (table_geom < 0 || tcp_site < 0) return 1;

    const unsigned n  = robot.chain.getNrOfJoints();
    const unsigned ns = robot.chain.getNrOfSegments();
    KDL::JntArray  q_start(n);
    {
        KDL::ChainFkSolverPos_recursive fk_ik(robot.chain);
        KDL::ChainIkSolverVel_wdls      ik_vel(robot.chain, 1e-5, 150);
        ik_vel.setLambda(0.05);
        const KDL::Vector start(kStartTcp[0], kStartTcp[1], kStartTcp[2]);
        if (!ex::solve_near_seed(
              ik_vel,
              fk_ik,
              robot,
              ex::home_q(n),
              KDL::Frame(KDL::Rotation::RotX(M_PI), start),
              q_start
            )) {
            std::cerr << "IK failed for the start pose\n";
            return 1;
        }
    }

    bool restarted = false;
    env.on_reset   = [&](mj_kdl::ResetContext *) {
        mj_kdl::set_joint_pos(&robot, q_start);
        for (unsigned i = 0; i < n; ++i) robot.jnt_pos_cmd[i] = q_start(i);
        restarted = true;
    };
    mj_kdl::reset(&env);
    print_contact_heights(env);
    if (!mj_kdl::set_control_mode(&robot, mj_kdl::CtrlMode::TORQUE)) return 1;

    const KDL::Twist root_acc(KDL::Vector(0.0, 0.0, -scene.gravity_z), KDL::Vector::Zero());
    KDL::ChainFkSolverPos_recursive fk(robot.chain);
    KDL::ChainHdSolver_Vereshchagin achd_approach(robot.chain, root_acc, 6);
    KDL::ChainHdSolver_Vereshchagin achd(robot.chain, root_acc, 5);
    KDL::ChainIdSolver_RNE          rnea(robot.chain, KDL::Vector(0.0, 0.0, scene.gravity_z));
    KDL::JntArray       q(n), qd(n), qdd(n), beta6(6), beta5(5), ff_tau(n), ctau(n), tau(n);
    const KDL::JntArray zero(n);
    KDL::Wrenches       f_ext(ns, KDL::Wrench::Zero());
    const KDL::Wrenches f_zero(ns, KDL::Wrench::Zero());
    KDL::Jacobian       alpha6(6), alpha5(5);
    for (unsigned i = 0; i < 6; ++i) alpha6(i, i) = 1.0;
    set_alpha_no_linear_z(alpha5);

    mj_kdl::update(&env);
    ex::read_q(robot, q, qd);
    KDL::Frame tracked;
    fk.JntToCart(q, tracked);
    KDL::Frame target = tracked;
    target.p += KDL::Vector(kMoveX, 0.0, 0.0);
    print_nc_comparison(robot, KDL::diff(tracked, target), q, qd, root_acc, rnea, 6);
    print_nc_comparison(robot, KDL::diff(tracked, target), q, qd, root_acc, rnea, 5);
    std::cout << "\n";

    // Driver weights 1 pass the commanded wrench through to the environment.
    achd.setDriverWeights(Eigen::VectorXd::Ones(5), Eigen::VectorXd::Zero(5));

    std::array<double, 6> err_prev{};
    Phase                 phase       = Phase::Approach;
    bool                  first       = true;
    int                   phase_steps = 0;
    double                touchdown   = kNoTouchdown; // [m/s] TCP sink speed at first contact
    double                elbow_min   = elbow_height(env);
    // Measured over the second half of the slide: the contact chatters while the TCP moves.
    int    contact_steps  = 0;
    double reaction_sum   = 0.0;
    int    reaction_count = 0;

    const auto restart_task = [&] {
        mj_kdl::update(&env);
        ex::read_q(robot, q, qd);
        fk.JntToCart(q, tracked);
        phase          = Phase::Approach;
        first          = true;
        phase_steps    = 0;
        touchdown      = kNoTouchdown;
        elbow_min      = elbow_height(env);
        contact_steps  = 0;
        reaction_sum   = 0.0;
        reaction_count = 0;
    };

    // One cycle: read the state (and apply the previous command), then command this one.
    const auto control = [&] {
        const double dt = env.model->opt.timestep;
        if (phase == Phase::Approach) {
            // The base stands on the table, so base-frame z is the height above it.
            const double v = std::clamp(kDescentGain * tracked.p.z(), kVTouch, kVMaxLin);
            tracked.p -= KDL::Vector(0.0, 0.0, v * dt);
        } else if (phase == Phase::Slide) {
            const KDL::Vector to_goal = target.p - tracked.p;
            const double      dist    = to_goal.Norm();
            if (dist > 1e-4) tracked.p += (to_goal / dist) * std::min(dist, kVMaxLin * dt);
        }

        mj_kdl::update(&env);
        ex::read_q(robot, q, qd);
        KDL::Frame current;
        fk.JntToCart(q, current);
        const KDL::Twist err  = KDL::diff(current, tracked);
        const double     e[6] = { err.vel.x(), err.vel.y(), err.vel.z(),
                                  err.rot.x(), err.rot.y(), err.rot.z() };
        if (first) std::copy(e, e + 6, err_prev.begin());
        first = false;
        double b[6];
        for (unsigned i = 0; i < 6; ++i) {
            const double kp = i < 3 ? kKpLin : kKpRot, kd = i < 3 ? kKdLin : kKdRot;
            b[i]        = ex::clamp_abs(kp * e[i] + kd * (e[i] - err_prev[i]) / dt, kBetaMax);
            err_prev[i] = e[i];
        }
        // Gravity as feed-forward, so only the commanded press pushes the free linear Z down.
        if (rnea.CartToJnt(q, zero, zero, f_zero, ff_tau) < 0) return false;
        if (phase == Phase::Approach) {
            for (unsigned i = 0; i < 6; ++i) beta6(i) = b[i];
            if (achd_approach.CartToJnt(q, qd, qdd, alpha6, beta6, f_zero, ff_tau, ctau) < 0)
                return false;
        } else {
            const double ramp = phase == Phase::Press
                                  ? std::min(1.0, static_cast<double>(phase_steps) / kPressSteps)
                                  : 1.0;
            f_ext.back() =
              KDL::Wrench(KDL::Vector(0.0, 0.0, -ramp * kPressForce), KDL::Vector::Zero());
            const unsigned no_z[5] = { 0, 1, 3, 4, 5 };
            for (unsigned i = 0; i < 5; ++i) beta5(i) = b[no_z[i]];
            if (achd.CartToJnt(q, qd, qdd, alpha5, beta5, f_ext, ff_tau, ctau) < 0) return false;
        }
        if (rnea.CartToJnt(q, qd, qdd, f_zero, tau) < 0) return false;
        // update() clamps each torque to its joint's limit and reports it in jnt_saturated.
        for (unsigned i = 0; i < n; ++i) robot.jnt_trq_cmd[i] = tau(i);
        ++phase_steps;
        return true;
    };

    if (!headless && !mj_kdl::open_viewer(&env)) return 1;
    restarted      = false;
    bool solver_ok = true;
    bool finished  = false;
    while (true) {
        if (!control()) {
            solver_ok = false;
            break;
        }
        if (!mj_kdl::step(&env)) break;
        elbow_min = std::min(elbow_min, elbow_height(env));
        if (phase == Phase::Approach && touches(env.data, table_geom)) {
            touchdown = site_sink_speed(env.model, env.data, tcp_site);
            target    = tracked;
            target.p += KDL::Vector(kMoveX, 0.0, 0.0);
            phase       = Phase::Press;
            phase_steps = 0;
        } else if (phase == Phase::Press && phase_steps >= kPressSteps) {
            phase       = Phase::Slide;
            phase_steps = 0;
        } else if (phase == Phase::Slide && phase_steps > kSlideSteps / 2) {
            const double reaction = contact_normal_force(env.model, env.data, table_geom);
            if (reaction > 0.0) ++contact_steps;
            reaction_sum += reaction;
            ++reaction_count;
        }
        if (restarted) {
            restarted = false;
            restart_task();
            continue;
        }
        mj_kdl::pace_realtime(&env);
        if (phase == Phase::Approach && phase_steps >= kApproachSteps) break;
        if (phase == Phase::Slide && phase_steps >= kSlideSteps) {
            finished = true;
            break;
        }
    }
    mj_kdl::update(&env);

    ex::read_q(robot, q, qd);
    KDL::Frame current;
    fk.JntToCart(q, current);
    const KDL::Twist err = KDL::diff(current, target);
    print_contact_heights(env);
    std::cout << std::fixed << std::setprecision(3)
              << "tcp_xy_err_mm=" << std::hypot(err.vel.x(), err.vel.y()) * 1000.0
              << " tcp_z_error_unconstrained_mm=" << err.vel.z() * 1000.0
              << " tcp_rot_err_rad=" << err.rot.Norm() << "\n";

    const double contact_fraction =
      reaction_count > 0 ? static_cast<double>(contact_steps) / reaction_count : 0.0;
    const double mean_reaction = reaction_count > 0 ? reaction_sum / reaction_count : 0.0;
    std::cout << "table_contact_fraction=" << contact_fraction
              << " mean_table_reaction_N=" << mean_reaction << " commanded_press_N=" << kPressForce
              << " reaction_over_command=" << mean_reaction / kPressForce << "\n"
              << "touchdown_speed_m_s=" << std::setprecision(4) << touchdown << " (limit "
              << kMaxTouchdown << ") elbow_min_z_above_table=" << elbow_min << " (limit "
              << kMinElbowHeight << ")\n";
    mj_kdl::cleanup(&env);
    const bool ok = solver_ok && finished && contact_fraction >= kContactHeld
                    && touchdown <= kMaxTouchdown && elbow_min >= kMinElbowHeight;
    return headless
             ? ex::verdict(ok, "the TCP touched down gently and held the table during the slide")
             : 0;
}
