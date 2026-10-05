"""DLFTContactAFT: analytic Jacobians against finite differences, plus the two
degenerate reductions (n_F = 0 -> DLFTContact, n_N = n_F = 0 -> AFT).

Self-contained: random modal data through NumericalFRF, whose dY/domega is
analytic, so the omega check tests the method and not the FRF provider.
"""
import numpy as np
import pytest

from pyfbs.nonlinearFBS import (
    AFT, DLFTContact, DLFTContactAFT, FBSProblem, FBS_System, Fourier,
    FourierOmegaPoint, HarmonicBalanceMethod, NumericalFRF,
)

HARMONICS = np.array([1, 2, 3])
# harmonic 0 is the static-preload case, and the only one that exercises the DC
# halving of JacobianFourier_Real
HARMONIC_SETS = [HARMONICS, np.array([0, 1, 2, 3])]
SAMPLE_NUMBER = 64            # >= 4H+1 = 13
N_INT = 4                     # DOF 0 normal, DOFs 1-2 friction, DOF 3 f^nl,A only
OMEGA = 150.0
EPSILON = 5.0e3
G_ZERO = 2.0e-4

L_N = np.eye(N_INT)[[0]]           # (1, 4)
L_F = np.eye(N_INT)[[1, 2]]        # (2, 4)
EMPTY = np.zeros((0, N_INT))


class _Synthetic(FBS_System):
    """Two random modal substructures coupled on N_INT DOFs.

    f^nl,A = k q^3 + 0.1 k qdot^3 on the full interface (G^A and Gdot^A both live),
    f^F    = mu f^N tanh(alpha qdot^F) + k_t q^F on the friction DOFs, so G^F,
    Gdot^F and P = df^F/df^N are all non-trivial.
    """

    def __init__(self, aft_gain=1.0, mu=0.4, k=1.0e3, k_t=50.0, alpha=2.0e3, seed=0):
        rng = np.random.default_rng(seed)
        n_modes = 6
        self.total_dimension = 2 * N_INT
        self.angular_eig_freq = np.sort(rng.uniform(40.0, 400.0, n_modes))
        self.eig_vec = rng.standard_normal((self.total_dimension, n_modes))
        self.zeta = 0.02
        self.B_coupling = np.hstack([np.eye(N_INT), -np.eye(N_INT)])   # q_rel = u_A - u_B
        self.sample_number = SAMPLE_NUMBER
        self.omega_ref = self.angular_eig_freq[0]
        self.F0 = rng.standard_normal((self.total_dimension,))
        self.aft_gain, self.mu, self.k, self.k_t, self.alpha = aft_gain, mu, k, k_t, alpha

    def external_term(self, tau):
        return self.F0[None, :, None] * np.cos(tau)[:, None, None]

    # --- f^nl,A on the FULL interface ---

    def interface_force(self, u_rel, udot_rel, tau):
        return self.aft_gain * (self.k * u_rel ** 3 + 0.1 * self.k * udot_rel ** 3)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return self.aft_gain * 3.0 * self.k * u_rel[:, :, 0][:, :, None] ** 2 * np.eye(N_INT)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return self.aft_gain * 0.3 * self.k * udot_rel[:, :, 0][:, :, None] ** 2 * np.eye(N_INT)

    # --- f^F on the friction DOFs, bounded by the DLFT normal force ---

    # the slip bound rides on the FIRST normal DOF, so the shapes stay valid for any
    # (n_F, n_N) -- including the degenerate n_F = 0 / n_N = 0 reductions below

    def friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        return (self.mu * f_normal[:, :1, :] * np.tanh(self.alpha * udot_F_rel)
                + self.k_t * u_F_rel)

    def jacobian_friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        return self.k_t * np.eye(u_F_rel.shape[1]) * np.ones((len(tau), 1, 1))

    def jacobian_friction_force_qdot(self, u_F_rel, udot_F_rel, f_normal, tau):
        t = np.tanh(self.alpha * udot_F_rel[:, :, 0])
        slope = self.mu * f_normal[:, :1, 0] * self.alpha * (1.0 - t ** 2)
        return slope[:, :, None] * np.eye(u_F_rel.shape[1])

    def jacobian_friction_force_normal(self, u_F_rel, udot_F_rel, f_normal, tau):
        J = np.zeros((len(tau), u_F_rel.shape[1], f_normal.shape[1]))
        J[:, :, :1] = self.mu * np.tanh(self.alpha * udot_F_rel)
        return J                                               # (Nt, n_F, n_N)


def _build(method, harmonics=HARMONICS, **system_kwargs):
    system = _Synthetic(**system_kwargs)
    HarmonicBalanceMethod.update_dependencies(harmonics, system.sample_number)
    provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    return FBSProblem(system, provider, method)


def _coefficients(harmonics=HARMONICS, seed=11, scale=1.0e-3):
    """Build AFTER the problem: Fourier's class tables must already be sized."""
    rng = np.random.default_rng(seed)
    shape = (len(harmonics), N_INT, 1)
    return scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _evaluate(problem, coefficients, omega=OMEGA):
    """F_int in RI form at a FRESH point -- the caches live on the point."""
    x = FourierOmegaPoint(Fourier(coefficients.copy()), omega)
    F = problem.method.compute_F_int(x, problem.ode)
    return np.concatenate((F.real, F.imag)), x


def _combined_problem(harmonics=HARMONICS):
    return _build(DLFTContactAFT(L_N, L_F, epsilon=EPSILON, g_zero=G_ZERO),
                  harmonics=harmonics)


def test_contact_state_is_mixed_and_both_blocks_are_live():
    """The evaluation point must actually exercise the DLFT branch."""
    problem = _combined_problem()
    _, x = _evaluate(problem, _coefficients())
    assert x.contact_mask.any() and not x.contact_mask.all()
    assert np.abs(x.f_normal).max() > 0.0


@pytest.mark.parametrize("harmonics", HARMONIC_SETS, ids=["odd_even", "with_dc"])
def test_jacobian_matches_finite_differences(harmonics):
    problem = _combined_problem(harmonics)
    coefficients = _coefficients(harmonics)
    x = FourierOmegaPoint(Fourier(coefficients.copy()), OMEGA)
    J = problem.method.compute_J_int_RI(x, problem.ode)

    step = 1e-7 * np.abs(coefficients).max()
    rng = np.random.default_rng(3)
    for _ in range(3):
        direction = rng.standard_normal((2 * len(harmonics) * N_INT, 1))
        direction /= np.linalg.norm(direction)
        perturbation = Fourier.new_from_RI(step * direction).coefficients

        F_plus, x_plus = _evaluate(problem, coefficients + perturbation)
        F_minus, x_minus = _evaluate(problem, coefficients - perturbation)
        # the scheme is semismooth: a switching sample inside the probe would make
        # the comparison meaningless, so require the contact state to hold
        assert np.array_equal(x_plus.contact_mask, x.contact_mask)
        assert np.array_equal(x_minus.contact_mask, x.contact_mask)

        finite_difference = (F_plus - F_minus) / (2.0 * step)
        analytic = J @ direction
        assert np.allclose(finite_difference, analytic,
                           rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


@pytest.mark.parametrize("harmonics", HARMONIC_SETS, ids=["odd_even", "with_dc"])
def test_domega_matches_finite_differences(harmonics):
    problem = _combined_problem(harmonics)
    coefficients = _coefficients(harmonics)
    x = FourierOmegaPoint(Fourier(coefficients.copy()), OMEGA)
    analytic = problem.method.compute_dF_int_domega_RI(x, problem.ode)

    step = 1e-6 * OMEGA
    F_plus, x_plus = _evaluate(problem, coefficients, OMEGA + step)
    F_minus, x_minus = _evaluate(problem, coefficients, OMEGA - step)
    assert np.array_equal(x_plus.contact_mask, x.contact_mask)
    assert np.array_equal(x_minus.contact_mask, x.contact_mask)

    finite_difference = (F_plus - F_minus) / (2.0 * step)
    assert np.allclose(finite_difference, analytic,
                       rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


def test_no_friction_dofs_reproduces_dlft_contact():
    """L_N = I, no friction DOFs, f^nl,A == 0  =>  plain DLFTContact."""
    combined = _build(DLFTContactAFT(np.eye(N_INT), EMPTY, epsilon=EPSILON, g_zero=G_ZERO),
                      aft_gain=0.0)
    reference = _build(DLFTContact(epsilon=EPSILON, g_zero=G_ZERO), aft_gain=0.0)
    coefficients = _coefficients()

    for problem in (combined, reference):
        problem._x = FourierOmegaPoint(Fourier(coefficients.copy()), OMEGA)

    for call in ("compute_F_int", "compute_J_int_RI", "compute_dF_int_domega_RI"):
        got = getattr(combined.method, call)(combined._x, combined.ode)
        want = getattr(reference.method, call)(reference._x, reference.ode)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-14 * np.abs(want).max()), call


def test_no_localised_dofs_reproduces_aft():
    """No normal and no friction DOFs  =>  plain AFT."""
    combined = _build(DLFTContactAFT(EMPTY, EMPTY, epsilon=EPSILON, g_zero=G_ZERO))
    reference = _build(AFT())
    coefficients = _coefficients()

    for problem in (combined, reference):
        problem._x = FourierOmegaPoint(Fourier(coefficients.copy()), OMEGA)

    for call in ("compute_F_int", "compute_J_int_RI", "compute_dF_int_domega_RI"):
        got = getattr(combined.method, call)(combined._x, combined.ode)
        want = getattr(reference.method, call)(reference._x, reference.ode)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-14 * max(np.abs(want).max(), 1.0)), call


def test_overlapping_localisations_are_rejected():
    with pytest.raises(ValueError, match="must be zero"):
        DLFTContactAFT(np.eye(N_INT)[[0]], np.eye(N_INT)[[0, 1]])


def test_non_orthonormal_localisation_is_rejected():
    with pytest.raises(ValueError, match="orthonormal rows"):
        DLFTContactAFT(2.0 * np.eye(N_INT)[[0]], np.eye(N_INT)[[1]])
