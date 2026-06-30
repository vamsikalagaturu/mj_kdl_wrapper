#include "admittance_ft_common.hpp"

namespace
{

static constexpr double kKpLin = 5760.0;
static constexpr double kKdLin = 378.0;
static constexpr double kKpRot = 3600.0;
static constexpr double kKdRot = 441.0;
static constexpr double kBetaLinMax = 3600.0;
static constexpr double kBetaRotMax = 2520.0;
static constexpr double kTauMax = 212.4;

class AchdController final : public admittance_ft::Controller
{
public:
    explicit AchdController(admittance_ft::SceneHandles &handles)
      : h(handles),
        fk(h.robot.chain),
        achd(h.robot.chain, KDL::Twist(KDL::Vector(0.0, 0.0, -h.scene.gravity_z), KDL::Vector::Zero()), 6),
        rnea(h.robot.chain, KDL::Vector(0.0, 0.0, h.scene.gravity_z)),
        q(h.robot.n_joints),
        qd(h.robot.n_joints),
        qdd(h.robot.n_joints),
        beta(6),
        ff_tau(h.robot.n_joints),
        constraint_tau(h.robot.n_joints),
        tau(h.robot.n_joints),
        alpha(6),
        f_ext_achd(h.robot.chain.getNrOfSegments(), KDL::Wrench::Zero()),
        f_ext_rnea(h.robot.chain.getNrOfSegments(), KDL::Wrench::Zero())
    {
        alpha.setColumn(0, KDL::Twist(KDL::Vector(1, 0, 0), KDL::Vector(0, 0, 0)));
        alpha.setColumn(1, KDL::Twist(KDL::Vector(0, 1, 0), KDL::Vector(0, 0, 0)));
        alpha.setColumn(2, KDL::Twist(KDL::Vector(0, 0, 1), KDL::Vector(0, 0, 0)));
        alpha.setColumn(3, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(1, 0, 0)));
        alpha.setColumn(4, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 1, 0)));
        alpha.setColumn(5, KDL::Twist(KDL::Vector(0, 0, 0), KDL::Vector(0, 0, 1)));
    }

    const char *name() const override { return "ex_admittance_ft_achd"; }
    mj_kdl::CtrlMode mode() const override { return mj_kdl::CtrlMode::TORQUE; }

    void reset() override
    {
        err_prev = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
        first_pid = true;
    }

    void track(const KDL::Frame &target) override
    {
        fill_state();
        KDL::Frame current;
        fk.JntToCart(q, current);
        const KDL::Twist err = KDL::diff(current, target);
        const double dt = h.model ? h.model->opt.timestep : 0.002;
        const double e[6] = { err.vel.x(), err.vel.y(), err.vel.z(), err.rot.x(), err.rot.y(), err.rot.z() };
        if (first_pid) {
            for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];
            first_pid = false;
        }

        beta(0) = admittance_ft::clamp(kKpLin * e[0] + kKdLin * ((e[0] - err_prev[0]) / dt), -kBetaLinMax, kBetaLinMax);
        beta(1) = admittance_ft::clamp(kKpLin * e[1] + kKdLin * ((e[1] - err_prev[1]) / dt), -kBetaLinMax, kBetaLinMax);
        beta(2) = admittance_ft::clamp(kKpLin * e[2] + kKdLin * ((e[2] - err_prev[2]) / dt), -kBetaLinMax, kBetaLinMax);
        beta(3) = admittance_ft::clamp(kKpRot * e[3] + kKdRot * ((e[3] - err_prev[3]) / dt), -kBetaRotMax, kBetaRotMax);
        beta(4) = admittance_ft::clamp(kKpRot * e[4] + kKdRot * ((e[4] - err_prev[4]) / dt), -kBetaRotMax, kBetaRotMax);
        beta(5) = admittance_ft::clamp(kKpRot * e[5] + kKdRot * ((e[5] - err_prev[5]) / dt), -kBetaRotMax, kBetaRotMax);
        for (unsigned i = 0; i < 6; ++i) err_prev[i] = e[i];

        KDL::SetToZero(ff_tau);
        if (achd.CartToJnt(q, qd, qdd, alpha, beta, f_ext_achd, ff_tau, constraint_tau) < 0) return;
        if (rnea.CartToJnt(q, qd, qdd, f_ext_rnea, tau) < 0) return;
        for (int i = 0; i < h.robot.n_joints; ++i) {
            h.robot.jnt_trq_cmd[i] = admittance_ft::clamp(tau(i), -kTauMax, kTauMax);
        }
    }

private:
    void fill_state()
    {
        for (int i = 0; i < h.robot.n_joints; ++i) {
            q(i) = h.robot.jnt_pos_msr[i];
            qd(i) = h.robot.jnt_vel_msr[i];
        }
    }

    admittance_ft::SceneHandles &h;
    KDL::ChainFkSolverPos_recursive fk;
    KDL::ChainHdSolver_Vereshchagin_Fixed_Joint achd;
    KDL::ChainIdSolver_RNE rnea;
    KDL::JntArray q;
    KDL::JntArray qd;
    KDL::JntArray qdd;
    KDL::JntArray beta;
    KDL::JntArray ff_tau;
    KDL::JntArray constraint_tau;
    KDL::JntArray tau;
    KDL::Jacobian alpha;
    KDL::Wrenches f_ext_achd;
    KDL::Wrenches f_ext_rnea;
    std::array<double, 6> err_prev = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    bool first_pid = true;
};

std::unique_ptr<admittance_ft::Controller> make_controller(admittance_ft::SceneHandles &h)
{
    return std::make_unique<AchdController>(h);
}

} // namespace

int main(int argc, char **argv)
{
    return admittance_ft::run(argc, argv, make_controller);
}
