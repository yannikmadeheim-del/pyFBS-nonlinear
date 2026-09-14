"""Two bar+beam elements with frictional contact -- AFT vs DLFT contact + AFT friction.

Tiago Martins' MSc thesis case study 6.4 (figure 6.13), solved as a two-substructure
FBS problem: each cantilever bar+beam element is one substructure, they are uncoupled
in the linear system, and a single contact element couples them across a 2-DOF
interface (normal + tangential).

Every run traces the SAME physics twice:

    "aft"       -- regularized unilateral spring (normal) + regularized dry friction,
                   exactly the thesis model, solved with AFT().
    "dlft_aft"  -- RIGID normal contact by DLFT prediction/correction, friction by
                   the IDENTICAL regularized tanh law, solved with DLFTContactAFT.

and does so for the modal FRF provider (always) and the experimental one (when
RUN_EXPERIMENTAL is set). The experimental provider is synthesized from the same
modal data, so its branches must reproduce the modal ones up to interpolation and
finite-difference error.

Run with::

    python -m pyfbs.nonlinearFBS.examples.two_bar_beam_friction.main
"""
import sys
import time
from copy import deepcopy

import numpy as np

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from pyfbs.nonlinearFBS import (
    AFT, ArcLengthParameterization, BiExponentialAdaptation, DLFTContactAFT,
    ExperimentalFRF, ExponentialAdaptation, FBSProblem, Fourier, Fourier_Real,
    FourierOmegaPoint, HarmonicBalanceMethod, NumericalFRF,
    OrthogonalParameterization, TangentPredictorBordered, TangentPredictorOne,
    TangentPredictorRobust, TangentPredictorTwo,
)

from .dynamical_system import (
    N_IF, L_F, L_N, BarBeamParams, SYSTEM_TYPES, modal_model,
)
from . import plotting_saving as io


# ---------------------------------------------------------------------------
# CONFIG -- system, then solver, then output. Plain JSON types only: the whole
# dict is written into every result CSV header and read back by the plotters.
# ---------------------------------------------------------------------------
CONFIG = dict(
    # --- system (thesis case study 6.4 values) ----------------------------
    params = BarBeamParams().to_dict(),
    solver = "pyfbs-nlfbs",         # names the pipeline in mixed comparisons

    # --- method ------------------------------------------------------------
    method = "aft",                 # "aft" | "dlft_aft"; study.py sweeps it
    dlft_epsilon = 1.0,             # DLFT penalty/Lagrangian factor; conditions the
                                    # iteration only, the converged force is eps-independent.
                                    # Now that the contact opens and closes every period the
                                    # conditioning decides how far the branch gets: 0.1 reaches
                                    # omega ~ 0.98, 1.0 stalls at 0.85, 100 at 0.90, and 10 and
                                    # 1000 fail at the very first point.

    # --- solver ------------------------------------------------------------
    # harmonic 0 is MANDATORY: the contact rectifies, so the response has a mean.
    harmonics = list(range(0, 16)),
    sample_number = 512,            # >= 4H+1 = 61, else the analytic Jacobian is inexact
    omega_lo = 0.85, omega_hi = 1.09, sweep = "up",     # thesis figure 6.15 window
    omega_resolution = 0.002,       # only consumed by the "experimental" provider
    parameterization = "ArcLengthParameterization",  # | OrthogonalParameterization
    predictor = "TangentPredictorBordered",          # | TangentPredictorOne
                                                     # | TangentPredictorTwo
                                                     # | TangentPredictorRobust
    step_adaptation = "ExponentialAdaptation",       # | BiExponentialAdaptation
    solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-8},
    step_kwargs = {"base": 2.0, "initial_step_length": 0.01,
                   "maximum_step_length": 0.05, "minimum_step_length": 1e-9,
                   "goal_number_of_iterations": 4},
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
)

# --- output ---------------------------------------------------------------
# main.py traces ONE configuration per method. For parameter permutations use
# study.py, which sweeps any list-valued entry of the parameter block.
METHODS = ["aft", "dlft_aft"]   # both are run, always
RUN_EXPERIMENTAL = False        # True -> also run every method on ExperimentalFRF

# solver parts addressable by name from CONFIG (and hence from a CSV header)
SOLVER_PARTS = {cls.__name__: cls for cls in (
    ArcLengthParameterization, OrthogonalParameterization,
    TangentPredictorBordered, TangentPredictorOne, TangentPredictorTwo,
    TangentPredictorRobust, ExponentialAdaptation, BiExponentialAdaptation)}


def params_of(cfg):
    return BarBeamParams(**cfg["params"])


def synthesis_omega_end(cfg):
    """End of the synthesized FRF grid [rad/s].

    The solver queries Y at every harmonic multiple n*omega, so the grid must
    reach the highest harmonic of the highest swept frequency; 2% of margin
    keeps the top of the window off the extrapolation warning.
    """
    return max(cfg["harmonics"]) * cfg["omega_hi"] * 1.02


def build_provider(cfg, frf_source):
    """The FRF provider selected by ``frf_source``, from the same modal data.

    Six DOFs give six modes, so the modal basis is COMPLETE and the "modal"
    provider is exact. The "experimental" one samples that same synthesis on a
    grid and splines it back, which is what introduces a representation error.
    """
    p = params_of(cfg)
    Omega, Phi, zeta = modal_model(p)
    modal = NumericalFRF.from_modal(Omega, Phi, zeta)
    if frf_source == "modal":
        return modal
    if frf_source == "experimental":
        # compute_FRF evaluates at harmonics*omega, so omega=1 with the grid as
        # "harmonics" synthesizes the whole grid in one vectorized call.
        grid = np.arange(0.0, synthesis_omega_end(cfg) + cfg["omega_resolution"],
                         cfg["omega_resolution"])
        dofs = np.arange(6)
        Y = modal.compute_FRF(1.0, grid, dofs, dofs)          # (N_freq, 6, 6)
        return ExperimentalFRF(grid, Y)
    raise ValueError(f"frf_source must be 'modal' or 'experimental', got {frf_source!r}")


def initial_guess(problem, omega_start):
    """Cold start from the contact-free linear response, CLIPPED AT THE GAP.

    The excitation is purely harmonic now, so there is no static contact state
    left to seed into harmonic 0. Neither obvious cold start works on its own:
    zero amplitude ignores a response that runs several gaps deep, and the
    contact-free response itself violates the constraint by just as much -- both
    leave the first Newton solve too far away to converge.

    Clipping the free response at x_N = eps is the rigid-contact KINEMATIC
    estimate: the free motion wherever the tips are apart, the gap wherever they
    would have interpenetrated. It satisfies the constraint by construction and
    only gets the contact force wrong, which is what Newton is good at fixing.
    """
    zero = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=omega_start)
    free = Fourier(problem._get_Fadm(zero).copy())   # contact-free: Q_rel = F_adm
    Fourier_Real.compute_time_series(free)
    time_series = free.time_series.copy()
    time_series[:, 0, 0] = np.minimum(time_series[:, 0, 0], problem.ode.p.eps)
    return FourierOmegaPoint(Fourier_Real.new_from_time_series(time_series),
                             omega_start)


def solve_config(cfg, provider, method=None):
    """Trace the forced-response branch of ``cfg``; ``method`` defaults to cfg's.

    The wiring order is load-bearing: ``Fourier`` state is class-level and
    ``FBSProblem.__init__`` snapshots ``Fourier.number_of_harmonics``, so
    ``update_dependencies`` must run BEFORE the problem is constructed, and
    again for every run whose harmonics or sample_number differ.

    :returns: (system, problem, solution_set, solve_time_s)
    """
    method = method or cfg["method"]
    harmonics = list(cfg["harmonics"])
    if 0 not in harmonics:
        raise ValueError("harmonic 0 is required: the contact rectifies, so the "
                         "response carries a non-zero mean")
    HarmonicBalanceMethod.update_dependencies(harmonics, cfg["sample_number"])

    p = params_of(cfg)
    system = SYSTEM_TYPES[method](p, sample_number=cfg["sample_number"])
    if method == "aft":
        nonlinear_method = AFT()
    elif method == "dlft_aft":
        nonlinear_method = DLFTContactAFT(L_N, L_F, epsilon=cfg["dlft_epsilon"],
                                          g_zero=p.eps)
    else:
        raise ValueError(f"method must be one of {list(SYSTEM_TYPES)}, got {method!r}")

    problem = FBSProblem(system, provider, nonlinear_method)
    solver = HarmonicBalanceMethod(
        harmonics=harmonics, freq_domain_ode=problem,
        corrector_parameterization=SOLVER_PARTS[cfg["parameterization"]],
        predictor=SOLVER_PARTS[cfg["predictor"]],
        step_length_adaptation=SOLVER_PARTS[cfg["step_adaptation"]])

    w_lo, w_hi = cfg["omega_lo"], cfg["omega_hi"]
    if cfg["sweep"] == "up":
        w_start, direction = w_lo, +1.0
    elif cfg["sweep"] == "down":
        w_start, direction = w_hi, -1.0
    else:
        raise ValueError(f"sweep must be 'up' or 'down', got {cfg['sweep']!r}")

    ig = initial_guess(problem, w_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex),
                                                   omega=direction)

    t0 = time.perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = cfg["maximum_number_of_solutions"],
        angular_frequency_range       = [w_lo, w_hi],
        solver_kwargs                 = dict(cfg["solver_kwargs"]),
        step_length_adaptation_kwargs = dict(cfg["step_kwargs"]),
        jacobian_update_frequency     = cfg["jacobian_update_frequency"],
        verbose                       = True,
    )
    return system, problem, ss, time.perf_counter() - t0


if __name__ == "__main__":
    frf_sources = ["modal"] + (["experimental"] if RUN_EXPERIMENTAL else [])
    print(f"Continuation window: omega in [{CONFIG['omega_lo']}, {CONFIG['omega_hi']}]"
          f" ({CONFIG['sweep']}ward sweep), harmonics {CONFIG['harmonics'][0]}"
          f"..{CONFIG['harmonics'][-1]}")

    written = []
    for frf_source in frf_sources:
        provider = build_provider(CONFIG, frf_source)
        for method in METHODS:
            print(f"\n=== {method} | {frf_source} FRF ===")
            cfg = deepcopy(CONFIG)
            cfg["method"] = method
            system, problem, ss, solve_time = solve_config(cfg, provider)
            out_csv = io.result_path(cfg, method, frf_source)
            io.save_solution(out_csv, cfg, ss, problem, system, method,
                             frf_source, solve_time)
            written.append(out_csv)

    print("\nwritten:")
    for path in written:
        print(f"  {path}")
    print("\nplot with:  python -m pyfbs.nonlinearFBS.examples"
          ".two_bar_beam_friction.plot_comparison")
