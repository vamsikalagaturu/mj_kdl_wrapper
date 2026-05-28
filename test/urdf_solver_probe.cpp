/* urdf_solver_probe.cpp
 * Single Kinova GEN3 URDF/MuJoCo dynamics probe:
 *   - parses the in-repo Kinova URDF with kdl_parser
 *   - runs ACHD fixed-joint solver at the home pose with 6 TCP constraints
 *   - builds the MuJoCo-derived KDL chain and compares RNEA gravity torques
 */

#include "chainhdsolver_vereshchagin_fixed_joint.hpp"
#include "mj_kdl_wrapper/mj_kdl_wrapper.hpp"

#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainidsolver_recursive_newton_euler.hpp>
#include <kdl/frames_io.hpp>
#include <kdl/kinfam_io.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static constexpr double kHome[7] = {
    0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633
};

static fs::path repo_root()
{
    return fs::path(__FILE__).parent_path().parent_path();
}

static void set_alpha_identity(KDL::Jacobian &alpha)
{
    for (unsigned i = 0; i < 6; ++i) {
        alpha.setColumn(
          i,
          KDL::Twist(KDL::Vector(i == 0, i == 1, i == 2), KDL::Vector(i == 3, i == 4, i == 5))
        );
    }
}

static void print_chain(const KDL::Chain &chain, const char *label)
{
    std::cout << label << ": " << chain.getNrOfJoints() << " joints, "
              << chain.getNrOfSegments() << " segments\n";
    for (unsigned i = 0; i < chain.getNrOfSegments(); ++i) {
        const KDL::Segment &seg = chain.getSegment(i);
        const KDL::RigidBodyInertia &ri = seg.getInertia();
        const KDL::RotationalInertia &I = ri.getRotationalInertia();
        const bool fixed = seg.getJoint().getType() == KDL::Joint::None;
        std::cout << "  [" << i << "] " << seg.getName()
                  << (fixed ? " [fixed]" : " [rev]")
                  << " joint=" << seg.getJoint().getName()
                  << " m=" << std::setprecision(4) << ri.getMass()
                  << " Ixx=" << I.data[0]
                  << " Iyy=" << I.data[4]
                  << " Izz=" << I.data[8] << "\n";
    }
    std::cout << "\n";
}

static bool parse_urdf_chain(const fs::path &urdf, const std::string &tip, KDL::Chain &chain)
{
    KDL::Tree tree;
    if (!kdl_parser::treeFromFile(urdf.string(), tree)) {
        std::cerr << "kdl_parser: failed to parse " << urdf << "\n";
        return false;
    }
    if (!tree.getChain("base_link", tip, chain)) {
        std::cerr << "getChain base_link->" << tip << " failed\n";
        return false;
    }
    return true;
}

static bool run_achd_probe(const KDL::Chain &chain)
{
    const unsigned n = chain.getNrOfJoints();
    const unsigned ns = chain.getNrOfSegments();
    if (n != 7) {
        std::cerr << "ACHD chain has unexpected joint count: " << n << "\n";
        return false;
    }

    KDL::JntArray q(n), qd(n), qdd(n), beta(6), ff_tau(n), constraint_tau(n);
    for (unsigned i = 0; i < n; ++i) q(i) = kHome[i];
    KDL::SetToZero(qd);
    KDL::SetToZero(beta);
    KDL::SetToZero(ff_tau);

    KDL::ChainFkSolverPos_recursive fk_pos(chain);
    KDL::Frame home_pose;
    fk_pos.JntToCart(q, home_pose);

    KDL::Jacobian alpha(6);
    set_alpha_identity(alpha);
    KDL::Wrenches f_ext(ns, KDL::Wrench::Zero());

    KDL::Twist root_acc(KDL::Vector(0.0, 0.0, 9.81), KDL::Vector::Zero());
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd(chain, root_acc, 6);
    const int rc = achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext, ff_tau, constraint_tau);
    if (rc < 0) {
        std::cerr << "ACHD CartToJnt failed: " << rc << "\n";
        return false;
    }

    std::vector<KDL::Twist> xdd(ns + 1);
    achd.getTransformedLinkAcceleration(xdd);

    std::cout << "=== ACHD fixed-joint probe: URDF base_link -> EndEffector_Link ===\n";
    print_chain(chain, "URDF ACHD");
    std::cout << "home_pose=" << home_pose << "\n";
    std::cout << "q=" << q << "\n";
    std::cout << "qd=" << qd << "\n";
    std::cout << "alpha=\n" << alpha.data << "\n";
    std::cout << "beta=" << beta << "\n";
    std::cout << "ff_tau=" << ff_tau << "\n";
    std::cout << "qdd=" << qdd << "\n";
    std::cout << "constraint_tau=" << constraint_tau << "\n";
    std::cout << "tcp_acc_base=" << xdd.back() << "\n\n";
    return true;
}

static bool run_rnea(
  const KDL::Chain &chain,
  const char       *label,
  std::vector<double> &tau_out
)
{
    const unsigned n = chain.getNrOfJoints();
    const unsigned s = chain.getNrOfSegments();
    if (n != 7) {
        std::cerr << label << ": unexpected joint count " << n << "\n";
        return false;
    }

    KDL::JntArray q(n), qdot(n), qddot(n), torques(n);
    for (unsigned i = 0; i < n; ++i) q(i) = kHome[i];

    KDL::Wrenches f_ext(s, KDL::Wrench::Zero());
    KDL::ChainIdSolver_RNE rnea(chain, KDL::Vector(0.0, 0.0, -9.81));
    if (rnea.CartToJnt(q, qdot, qddot, f_ext, torques) < 0) {
        std::cerr << label << ": RNEA failed\n";
        return false;
    }

    tau_out.resize(n);
    std::cout << label << " RNEA gravity torques at home:\n";
    for (unsigned i = 0; i < n; ++i) {
        tau_out[i] = torques(i);
        std::cout << "  j[" << i << "] " << std::setw(10) << std::fixed
                  << std::setprecision(5) << torques(i) << " Nm\n";
    }

    KDL::ChainFkSolverPos_recursive fk(chain);
    KDL::Frame ee;
    fk.JntToCart(q, ee);
    std::cout << "  EE: (" << std::setprecision(4)
              << ee.p.x() << ", " << ee.p.y() << ", " << ee.p.z() << ")\n\n";
    return true;
}

int main(int argc, char *argv[])
{
    const fs::path root = repo_root();
    const fs::path urdf = (argc > 1)
                            ? fs::path(argv[1])
                            : root / "third_party/kinova/GEN3_URDF_V12.urdf";

    KDL::Chain urdf_achd_chain;
    if (!parse_urdf_chain(urdf, "EndEffector_Link", urdf_achd_chain)) return 1;
    if (!run_achd_probe(urdf_achd_chain)) return 1;

    KDL::Chain urdf_rnea_chain;
    if (!parse_urdf_chain(urdf, "Bracelet_Link", urdf_rnea_chain)) return 1;

    const std::string arm_mjcf = (root / "third_party/menagerie/kinova_gen3/gen3.xml").string();
    const std::string grp_mjcf = (root / "third_party/menagerie/robotiq_2f85/2f85.xml").string();

    mj_kdl::AttachmentSpec gripper;
    gripper.mjcf_path = grp_mjcf.c_str();
    gripper.attach_to = { mj_kdl::AttachKind::Site, "pinch_site" };
    gripper.prefix    = "g_";

    mj_kdl::RobotSpec robot_spec;
    robot_spec.path = arm_mjcf.c_str();
    robot_spec.attachments.push_back(gripper);

    mj_kdl::SceneSpec scene;
    scene.timestep   = 0.002;
    scene.add_floor  = true;
    scene.add_skybox = true;
    scene.robots.push_back(robot_spec);

    mjModel *model = nullptr;
    mjData  *data = nullptr;
    if (!mj_kdl::build_scene(&model, &data, &scene)) {
        std::cerr << "build_scene failed\n";
        return 1;
    }

    mj_kdl::Robot robot;
    if (!mj_kdl::init_robot_from_mjcf(&robot, model, data, "base_link", "bracelet_link", "", nullptr)) {
        std::cerr << "init_robot_from_mjcf failed\n";
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    std::cout << "=== RNEA comparison: URDF Bracelet_Link vs MuJoCo bracelet_link ===\n";
    print_chain(urdf_rnea_chain, "URDF RNEA");
    print_chain(robot.chain, "MuJoCo RNEA");

    std::vector<double> tau_urdf, tau_mj;
    const bool rnea_ok = run_rnea(urdf_rnea_chain, "URDF", tau_urdf)
                         && run_rnea(robot.chain, "MuJoCo", tau_mj);
    if (!rnea_ok) {
        mj_kdl::cleanup(&robot);
        mj_kdl::destroy_scene(model, data);
        return 1;
    }

    std::cout << "MuJoCo - URDF gravity torque difference:\n";
    const unsigned n = std::min(tau_urdf.size(), tau_mj.size());
    for (unsigned i = 0; i < n; ++i) {
        const double d = tau_mj[i] - tau_urdf[i];
        std::cout << "  j[" << i << "] " << std::setw(9) << std::fixed
                  << std::setprecision(5) << d << " Nm"
                  << (std::abs(d) > 0.05 ? "  <-- MISMATCH" : "") << "\n";
    }

    mj_kdl::cleanup(&robot);
    mj_kdl::destroy_scene(model, data);
    return 0;
}
