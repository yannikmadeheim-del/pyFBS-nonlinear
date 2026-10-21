"""ComplexStiffness (Extension A): analytic Jacobians against finite differences,
the linear limit against closed-form LM-FBS, the first-harmonic equivalence
with AFT for a diagonal cubic spring-damper law, and the tabulated stiffness of
FBS_SystemComplexStiffnessExperimental.

Self-contained: random modal data through NumericalFRF, whose dY/domega is
analytic, so the omega checks test the method and not the FRF provider. The
same setup and law parameters feed a time-domain system (for AFT) and a
frequency-domain system (for ComplexStiffness).
"""
import warnings

import numpy as np
import pytest

from pyfbs.nonlinearFBS import (
    AFT, ComplexStiffness, FBSProblem, FBS_System, FBS_SystemComplexStiffness,
    FBS_SystemComplexStiffnessExperimental,
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
SPRING = dict(k=1.0e3, alpha=0.0, c=0.0, beta=0.0)
ACTIVE_DAMPER = dict(k=1.0e3, alpha=0.0, c=-0.5, beta=0.0)   # Im k* = omega c < 0: not passive


def _setup(system, k, alpha, c, beta, seed=0):
    """Two random modal substructures coupled on N_INT DOFs, monoharmonic excitation."""
    rng = np.random.default_rng(seed)
    n_modes = 6
    system.total_dimension = 2 * N_INT
    system.angular_eig_freq = np.sort(rng.uniform(40.0, 400.0, n_modes))
    system.eig_vec = rng.standard_normal((system.total_dimension, n_modes))
    system.zeta = 0.02
    system.B_coupling = np.hstack([np.eye(N_INT), -np.eye(N_INT)])   # q_rel = u_A - u_B
    system.sample_number = SAMPLE_NUMBER
    system.omega_ref = system.angular_eig_freq[0]
    system.F0 = rng.standard_normal((system.total_dimension,))
    system.k, system.alpha, system.c, system.beta = k, alpha, c, beta


class _TimeDomain(FBS_System):
    """Diagonal law f = k q + alpha q^3 + c qdot + beta qdot^3, for AFT."""

    def __init__(self, k, alpha, c, beta):
        _setup(self, k, alpha, c, beta)

    def external_term(self, tau):
        return self.F0[None, :, None] * np.cos(tau)[:, None, None]

    def interface_force(self, u_rel, udot_rel, tau):
        return (self.k * u_rel + self.alpha * u_rel ** 3
                + self.c * udot_rel + self.beta * udot_rel ** 3)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return (self.k + 3.0 * self.alpha * u_rel[:, :, 0][:, :, None] ** 2) * np.eye(N_INT)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return (self.c + 3.0 * self.beta * udot_rel[:, :, 0][:, :, None] ** 2) * np.eye(N_INT)


class _FrequencyDomain(FBS_SystemComplexStiffness):
    """First-harmonic (describing-function) stiffness of the _TimeDomain law:
    k* = k + 3/4 alpha a^2 + i omega (c + 3/4 beta omega^2 a^2)."""

    def __init__(self, k, alpha, c, beta):
        _setup(self, k, alpha, c, beta)

    def external_term(self, tau):
        return self.F0[None, :, None] * np.cos(tau)[:, None, None]

    def complex_stiffness(self, omega, amplitude):
        return (self.k + 0.75 * self.alpha * amplitude ** 2
                + 1j * omega * (self.c + 0.75 * self.beta * omega ** 2 * amplitude ** 2))

    def complex_stiffness_derivative_amplitude(self, omega, amplitude):
        return 1.5 * self.alpha * amplitude + 1j * omega * 1.5 * self.beta * omega ** 2 * amplitude

    def complex_stiffness_derivative_omega(self, omega, amplitude):
        return 1j * (self.c + 2.25 * self.beta * omega ** 2 * amplitude ** 2)


class _ScalarSpring(_FrequencyDomain):
    """Constant law returned as plain scalars, to exercise the broadcast in ComplexStiffness."""

    def complex_stiffness(self, omega, amplitude):
        return self.k

    def complex_stiffness_derivative_amplitude(self, omega, amplitude):
        return 0.0

    def complex_stiffness_derivative_omega(self, omega, amplitude):
        return 0.0


def _build(method, law=CUBIC, harmonics=HARMONICS, system_class=None):
    """FBSProblem with the system type that matches the method, unless given."""
    if system_class is None:
        system_class = _FrequencyDomain if isinstance(method, ComplexStiffness) else _TimeDomain
    system = system_class(**law)
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
        _build(ComplexStiffness(), harmonics=np.array([1, 3]))


def test_bind_rejects_time_domain_system():
    with pytest.raises(TypeError):
        _build(ComplexStiffness(), system_class=_TimeDomain)


def test_scalar_law_is_broadcast():
    """A law that returns plain numbers yields one value per interface DOF."""
    problem = _build(ComplexStiffness(), law=SPRING, system_class=_ScalarSpring)
    coefficients = _coefficients()
    point = _point(coefficients)

    force = problem.method.compute_F_int(point, problem.ode)
    assert force.shape == (N_INT, 1)
    assert np.allclose(force, SPRING["k"] * coefficients.reshape(-1, 1))
    assert problem.method.compute_J_int_RI(point, problem.ode).shape == (2 * N_INT, 2 * N_INT)
    assert problem.method.compute_dF_int_domega_RI(point, problem.ode).shape == (2 * N_INT, 1)


def test_negative_loss_stiffness_warns():
    """Im k* < 0 feeds energy into the structure every cycle: the law is not passive."""
    problem = _build(ComplexStiffness(), law=ACTIVE_DAMPER)
    with pytest.warns(UserWarning, match="not passive"):
        problem.method.compute_F_int(_point(_coefficients()), problem.ode)


def test_passive_law_does_not_warn():
    problem = _build(ComplexStiffness(), law=CUBIC)
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning would fail the test
        problem.method.compute_F_int(_point(_coefficients()), problem.ode)


# ---------------------------------------------------------------------------
# V0: analytic Jacobians against finite differences
# ---------------------------------------------------------------------------

def test_force_jacobian_matches_finite_differences():
    problem = _build(ComplexStiffness())
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
    problem = _build(ComplexStiffness())
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
    problem = _build(ComplexStiffness())
    coefficients = _coefficients()
    analytic = problem.method.compute_dF_int_domega_RI(_point(coefficients), problem.ode)

    step = 1e-6 * OMEGA
    finite_difference = (_force_RI(problem, coefficients, OMEGA + step)
                         - _force_RI(problem, coefficients, OMEGA - step)) / (2.0 * step)
    assert np.allclose(finite_difference, analytic,
                       rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


def test_residual_jacobians_match_finite_differences():
    """The same checks through FBSProblem: RI layout and admittance product."""
    problem = _build(ComplexStiffness())
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

@pytest.mark.parametrize("law", [SPRING, SPRING_DAMPER], ids=["real_stiffness", "spring_damper"])
def test_constant_stiffness_reproduces_linear_lmfbs(law):
    """linear_initialization solves (I + Y_int (K + i omega C)) Q = F_adm in closed
    form; the residual of ComplexStiffness must vanish at that point."""
    problem = _build(ComplexStiffness(), law=law)
    x = problem.linear_initialization(OMEGA, law["k"] * np.ones(N_INT), law["c"] * np.ones(N_INT))

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
    complex_stiffness = _build(ComplexStiffness(), law=law)
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
    for method in (AFT(), ComplexStiffness()):
        problem = _build(method)
        solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem)
        seed = problem.linear_initialization(OMEGA, CUBIC["k"] * np.ones(N_INT),
                                             CUBIC["c"] * np.ones(N_INT))
        solution, _, success, _ = solver.solve_fixed_frequency(seed, **solver_kwargs)
        assert success
        solutions.append(solution.fourier.coefficients)

    assert np.allclose(solutions[1], solutions[0],
                       rtol=1e-8, atol=1e-10 * np.abs(solutions[0]).max())


# ---------------------------------------------------------------------------
# Tabulated stiffness: FBS_SystemComplexStiffnessExperimental
# ---------------------------------------------------------------------------

# each DOF has its own amplitude grid; the grids enclose OMEGA and the test points
OMEGA_GRID = np.linspace(50.0, 300.0, 12)
AMPLITUDE_SCALE = np.array([1.0, 1.5, 2.0])
A_REF = 2.0e-3


def _amplitude_grid(n, start=0.0, stop=8.0e-3):
    return np.linspace(start, stop, n)[:, None] * AMPLITUDE_SCALE      # (n, N_INT)


def _cubic_law(omega, amplitude):
    """Describing function of CUBIC, the same as _FrequencyDomain.complex_stiffness."""
    return _FrequencyDomain(**CUBIC).complex_stiffness(omega, amplitude)


def _soft_law(omega, amplitude):
    """Smooth, non-polynomial law: amplitude softening and frequency-dependent loss."""
    return (CUBIC["k"] * (1.0 - 0.5 * np.tanh(amplitude / A_REF))
            + 1j * omega * CUBIC["c"] * (1.0 + 0.5 * np.sin(omega / 40.0)))


def _table(law, omega_grid=OMEGA_GRID, amplitude_grid=None):
    """K[p, m, j] = law(omega_p, a_mj) on the given grids."""
    amplitude_grid = _amplitude_grid(10) if amplitude_grid is None else amplitude_grid
    return omega_grid, amplitude_grid, law(omega_grid[:, None, None], amplitude_grid[None, :, :])


class _Tabulated(FBS_SystemComplexStiffnessExperimental):
    """Same setup as the other test systems, law from a table."""

    def __init__(self, omega, amplitude, K):
        super().__init__(omega, amplitude, K)
        _setup(self, **CUBIC)

    def external_term(self, tau):
        return self.F0[None, :, None] * np.cos(tau)[:, None, None]


def _build_tabulated(*table):
    system = _Tabulated(*table)
    HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)
    provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    return FBSProblem(system, provider, ComplexStiffness())


@pytest.mark.parametrize("case", ["1D amplitude", "wrong K shape", "decreasing omega",
                                  "decreasing amplitude", "3 amplitude levels"])
def test_table_rejects_invalid_input(case):
    omega, amplitude, K = _table(_cubic_law)
    broken = {
        "1D amplitude": (omega, amplitude[:, 0], K[:, :, 0]),
        "wrong K shape": (omega, amplitude, K[:, :-1]),
        "decreasing omega": (omega[::-1], amplitude, K),
        "decreasing amplitude": (omega, amplitude[::-1], K),
        "3 amplitude levels": (omega, amplitude[:3], K[:, :3]),
    }[case]
    with pytest.raises(ValueError):
        FBS_SystemComplexStiffnessExperimental(*broken)


def test_table_reproduces_polynomial_law():
    """CUBIC is at most cubic in omega and quadratic in a, so the cubic splines
    reproduce it exactly: force and both Jacobians equal the analytic system."""
    analytic = _build(ComplexStiffness())
    tabulated = _build_tabulated(*_table(_cubic_law))
    coefficients = _coefficients()

    for name in ("compute_F_int", "compute_J_int_RI", "compute_dF_int_domega_RI"):
        reference = getattr(analytic.method, name)(_point(coefficients), analytic.ode)
        result = getattr(tabulated.method, name)(_point(coefficients), tabulated.ode)
        assert np.allclose(result, reference,
                           rtol=1e-7, atol=1e-9 * np.abs(reference).max()), name


def test_table_interpolation_error_decreases_with_refinement():
    """Guide A4: for a non-polynomial law the interpolation error of cubic splines
    falls with about the fourth power of the grid spacing, once the grid resolves
    the law (16 points: spacing about A_REF / 4)."""
    rng = np.random.default_rng(5)
    omega_queries = rng.uniform(OMEGA_GRID[0], OMEGA_GRID[-1], 20)
    amplitude_queries = rng.uniform(0.0, 8.0e-3, (20, 1)) * AMPLITUDE_SCALE

    errors = []
    for n in (16, 32):
        table = FBS_SystemComplexStiffnessExperimental(
            *_table(_soft_law, np.linspace(OMEGA_GRID[0], OMEGA_GRID[-1], n), _amplitude_grid(n)))
        errors.append(max(np.abs(table.complex_stiffness(w, a) - _soft_law(w, a)).max()
                          for w, a in zip(omega_queries, amplitude_queries)))

    assert errors[1] < errors[0] / 8.0


def test_table_jacobians_match_finite_differences():
    """The spline derivatives are consistent with the spline values, through ComplexStiffness."""
    problem = _build_tabulated(*_table(_soft_law))
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

    analytic = problem.method.compute_dF_int_domega_RI(_point(coefficients), problem.ode)
    step = 1e-6 * OMEGA
    finite_difference = (_force_RI(problem, coefficients, OMEGA + step)
                         - _force_RI(problem, coefficients, OMEGA - step)) / (2.0 * step)
    assert np.allclose(finite_difference, analytic,
                       rtol=1e-6, atol=1e-8 * np.abs(analytic).max())


def test_table_holds_value_below_smallest_amplitude():
    """Below the table the smallest level is held, with zero amplitude derivative and a warning."""
    amplitude_grid = _amplitude_grid(10, start=1.0e-3)
    table = FBS_SystemComplexStiffnessExperimental(*_table(_soft_law, amplitude_grid=amplitude_grid))
    amplitude = np.array([0.0, 4.0e-3, 4.0e-3])          # DOF 0 below its smallest level

    with pytest.warns(UserWarning, match="below the smallest"):
        value = table.complex_stiffness(OMEGA, amplitude)
    with pytest.warns(UserWarning, match="below the smallest"):
        slope = table.complex_stiffness_derivative_amplitude(OMEGA, amplitude)

    held = table.complex_stiffness(OMEGA, amplitude_grid[0])
    assert np.isclose(value[0], held[0])
    assert slope[0] == 0.0 and np.all(slope[1:] != 0.0)


@pytest.mark.parametrize("omega, amplitude", [(400.0, np.full(N_INT, 1.0e-3)),
                                              (OMEGA, np.array([1.0e-3, 1.0, 1.0e-3]))],
                         ids=["omega_above_table", "amplitude_above_table"])
def test_table_rejects_queries_outside(omega, amplitude):
    table = FBS_SystemComplexStiffnessExperimental(*_table(_soft_law))
    with pytest.raises(ValueError):
        table.complex_stiffness(omega, amplitude)
