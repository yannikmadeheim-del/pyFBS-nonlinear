"""ComplexStiffness (Extension A): analytic Jacobians against finite differences,
the linear limit against closed-form LM-FBS, and the first-harmonic equivalence
with AFT for a diagonal cubic spring-damper law.

Self-contained: random modal data through NumericalFRF, whose dY/domega is
analytic, so the omega checks test the method and not the FRF provider.
"""
import numpy as np
import pytest

from pyfbs.nonlinearFBS import (
    AFT, AnalyticalStiffnessMap, ComplexStiffness, FBSProblem, FBS_System,
    Fourier, FourierOmegaPoint, HarmonicBalanceMethod, NumericalFRF,
)

HARMONICS = np.array([1])
SAMPLE_NUMBER = 64            # first harmonic of a cubic law is alias-free for Nt > 4
N_INT = 3
OMEGA = 150.0
AMPLITUDE = 1.0e-3            # physical amplitude scale of the random test points

# f = k q + alpha q^3 + c qdot + beta qdot^3 on every interface DOF; at
# AMPLITUDE and OMEGA all four terms are of comparable size
CUBIC = dict(k=1.0e3, alpha=1.0e9, c=0.5, beta=10.0)
SPRING_DAMPER = dict(k=1.0e3, alpha=0.0, c=0.5, beta=0.0)


class _Synthetic(FBS_System):
    """Two random modal substructures coupled on N_INT DOFs, monoharmonic
    excitation, diagonal law f = k q + alpha q^3 + c qdot + beta qdot^3 for AFT."""

    def __init__(self, k, alpha, c, beta, seed=0):
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
        self.k, self.alpha, self.c, self.beta = k, alpha, c, beta

    def external_term(self, tau):
        return self.F0[None, :, None] * np.cos(tau)[:, None, None]

    def interface_force(self, u_rel, udot_rel, tau):
        return (self.k * u_rel + self.alpha * u_rel ** 3
                + self.c * udot_rel + self.beta * udot_rel ** 3)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return (self.k + 3.0 * self.alpha * u_rel[:, :, 0][:, :, None] ** 2) * np.eye(N_INT)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return (self.c + 3.0 * self.beta * udot_rel[:, :, 0][:, :, None] ** 2) * np.eye(N_INT)


def _equivalent_map(k, alpha, c, beta):
    """First-harmonic (describing-function) stiffness of the _Synthetic law:
    k* = k + 3/4 alpha a^2 + i omega (c + 3/4 beta omega^2 a^2)."""
    return AnalyticalStiffnessMap(
        lambda w, a: k + 0.75 * alpha * a ** 2 + 1j * w * (c + 0.75 * beta * w ** 2 * a ** 2),
        lambda w, a: 1.5 * alpha * a + 1j * w * 1.5 * beta * w ** 2 * a,
        lambda w, a: 1j * (c + 2.25 * beta * w ** 2 * a ** 2),
    )


def _build(method, law=CUBIC, harmonics=HARMONICS):
    system = _Synthetic(**law)
    HarmonicBalanceMethod.update_dependencies(harmonics, system.sample_number)
    provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    return FBSProblem(system, provider, method)


def _coefficients(seed=11):
    """Random first-harmonic coefficients of physical size AMPLITUDE, with the
    last DOF at exactly zero amplitude so the zero guard is always exercised.
    Build AFTER the problem: Fourier's class tables must already be sized."""
    rng = np.random.default_rng(seed)
    shape = (1, N_INT, 1)
    scale = 0.5 * Fourier.number_of_time_samples * AMPLITUDE
    coefficients = scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    coefficients[0, -1, 0] = 0.0
    return coefficients


def _point(coefficients, omega=OMEGA):
    """A FRESH point -- the caches live on the point."""
    return FourierOmegaPoint(Fourier(coefficients.copy()), omega)


def _force_RI(problem, coefficients, omega=OMEGA):
    F = problem.method.compute_F_int(_point(coefficients, omega), problem.ode)
    return np.concatenate((F.real, F.imag))


def _directions(n, count=3, seed=3):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        direction = rng.standard_normal((2 * n, 1))
        yield direction / np.linalg.norm(direction)


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------

def test_bind_rejects_higher_harmonics():
    with pytest.raises(ValueError):
        _build(ComplexStiffness(_equivalent_map(**CUBIC)), harmonics=np.array([1, 3]))


# ---------------------------------------------------------------------------
# V0: analytic Jacobians against finite differences
# ---------------------------------------------------------------------------

def test_force_jacobian_matches_finite_differences():
    problem = _build(ComplexStiffness(_equivalent_map(**CUBIC)))
    coefficients = _coefficients()
    J = problem.method.compute_J_int_RI(_point(coefficients), problem.ode)

    step = 1e-7 * np.abs(coefficients).max()
    for direction in _directions(N_INT):
        perturbation = Fourier.new_from_RI(step * direction).coefficients
        finite_difference = (_force_RI(problem, coefficients + perturbation)
                             - _force_RI(problem, coefficients - perturbation)) / (2.0 * step)
        analytic = J @ direction
        assert np.allclose(finite_difference, analytic,
                           rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


def test_jacobian_converges_under_step_refinement():
    """Central differences converge with step^2 only if the analytic Jacobian is right."""
    problem = _build(ComplexStiffness(_equivalent_map(**CUBIC)))
    coefficients = _coefficients()
    J = problem.method.compute_J_int_RI(_point(coefficients), problem.ode)
    direction = next(_directions(N_INT))

    errors = []
    for step in 1e-2 * np.abs(coefficients).max() * np.array([1.0, 0.1, 0.01]):
        perturbation = Fourier.new_from_RI(step * direction).coefficients
        finite_difference = (_force_RI(problem, coefficients + perturbation)
                             - _force_RI(problem, coefficients - perturbation)) / (2.0 * step)
        errors.append(np.abs(finite_difference - J @ direction).max())

    assert errors[1] < errors[0] / 50.0
    assert errors[2] < errors[1] / 50.0


def test_domega_matches_finite_differences():
    problem = _build(ComplexStiffness(_equivalent_map(**CUBIC)))
    coefficients = _coefficients()
    analytic = problem.method.compute_dF_int_domega_RI(_point(coefficients), problem.ode)

    step = 1e-6 * OMEGA
    finite_difference = (_force_RI(problem, coefficients, OMEGA + step)
                         - _force_RI(problem, coefficients, OMEGA - step)) / (2.0 * step)
    assert np.allclose(finite_difference, analytic,
                       rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


def test_residual_jacobians_match_finite_differences():
    """The same checks through FBSProblem: RI layout and admittance product."""
    problem = _build(ComplexStiffness(_equivalent_map(**CUBIC)))
    coefficients = _coefficients()
    J = problem.compute_jacobian_of_residue_RI(_point(coefficients))

    step = 1e-7 * np.abs(coefficients).max()
    for direction in _directions(N_INT):
        perturbation = Fourier.new_from_RI(step * direction).coefficients
        finite_difference = (problem.compute_residue_RI(_point(coefficients + perturbation))
                             - problem.compute_residue_RI(_point(coefficients - perturbation))
                             ) / (2.0 * step)
        analytic = J @ direction
        assert np.allclose(finite_difference, analytic,
                           rtol=1e-6, atol=1e-8 * np.abs(analytic).max())

    analytic = problem.compute_derivative_wrt_omega_RI(_point(coefficients))
    step = 1e-6 * OMEGA
    finite_difference = (problem.compute_residue_RI(_point(coefficients, OMEGA + step))
                         - problem.compute_residue_RI(_point(coefficients, OMEGA - step))
                         ) / (2.0 * step)
    assert np.allclose(finite_difference, analytic,
                       rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


# ---------------------------------------------------------------------------
# V1/V2: linear limit against the closed-form relaxed-compatibility LM-FBS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("damping", [0.0, SPRING_DAMPER["c"]],
                         ids=["real_stiffness", "spring_damper"])
def test_constant_stiffness_reproduces_linear_lmfbs(damping):
    """linear_initialization solves (I + Y_int (K + i omega C)) Q = F_adm in closed
    form; the residual of ComplexStiffness must vanish at that point."""
    k = SPRING_DAMPER["k"]
    stiffness_map = AnalyticalStiffnessMap(lambda w, a: k + 1j * w * damping,
                                           lambda w, a: 0.0,
                                           lambda w, a: 1j * damping)
    problem = _build(ComplexStiffness(stiffness_map), law=SPRING_DAMPER)
    x = problem.linear_initialization(OMEGA, k * np.ones(N_INT), damping * np.ones(N_INT))

    residual = problem.compute_residue_RI(x)
    assert np.abs(residual).max() <= 1e-10 * np.abs(x.fourier.coefficients).max()


# ---------------------------------------------------------------------------
# V3: equivalence with first-harmonic AFT
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("law", [SPRING_DAMPER, CUBIC], ids=["spring_damper", "cubic"])
def test_matches_first_harmonic_aft(law):
    """With harmonics [1], AFT returns exactly the first Fourier coefficient of the
    law, i.e. its describing function: force and both Jacobians must agree. The
    spring-damper case checks the sign and phase of the loss stiffness."""
    aft = _build(AFT(), law=law)
    complex_stiffness = _build(ComplexStiffness(_equivalent_map(**law)), law=law)
    coefficients = _coefficients()

    for name in ("compute_F_int", "compute_J_int_RI", "compute_dF_int_domega_RI"):
        reference = getattr(aft.method, name)(_point(coefficients), aft.ode)
        result = getattr(complex_stiffness.method, name)(_point(coefficients),
                                                         complex_stiffness.ode)
        assert np.allclose(result, reference,
                           rtol=1e-9, atol=1e-12 * np.abs(reference).max()), name


def test_solution_matches_first_harmonic_aft():
    """Fixed-frequency Newton solves from the same linear seed land on the same point."""
    solver_kwargs = {"maximum_iterations": 100, "absolute_tolerance": 1e-10}
    solutions = []
    for method in (AFT(), ComplexStiffness(_equivalent_map(**CUBIC))):
        problem = _build(method)
        solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem)
        seed = problem.linear_initialization(OMEGA, CUBIC["k"] * np.ones(N_INT),
                                             CUBIC["c"] * np.ones(N_INT))
        solution, _, success, _ = solver.solve_fixed_frequency(seed, **solver_kwargs)
        assert success
        solutions.append(solution.fourier.coefficients)

    assert np.allclose(solutions[1], solutions[0],
                       rtol=1e-8, atol=1e-10 * np.abs(solutions[0]).max())
