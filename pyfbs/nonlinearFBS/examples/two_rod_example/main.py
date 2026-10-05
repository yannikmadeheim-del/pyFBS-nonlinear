"""Two-rod vibro-impact -- single-system run, saved for the verification plots.

Two identical clamped-free FE rods face each other across a gap g0; the left rod
is harmonically driven at its tip, the right rod is an identical passive mirror.
Configure the system here (contact method, FRF source, parameters), solve its
nonlinear frequency response -- one branch per DLFT penalty factor in
EPSILON_REL --, save every branch to Results/<RESULT_DIR>/ and plot the driven
tip u_A and the relative approach x_r = u_A + u_B.

The saved CSV is compared against the NLvib references with plotting.py.

Run:  python main.py
"""
import sys

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS import (
    Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, ExperimentalFRF,
    DLFTContact, DLFTContactAFT, AFT, HarmonicBalanceMethod,
    ArcLengthParameterization, TangentPredictorBordered,
)

from dynamical_system import RodParams, TwoRodVibroImpact, TwoRodPenaltyContact, L_N, L_F
import plotting


# ============================ configuration =================================
# --- which model to solve ---
METHOD     = "dlft_aft"    # "dlft_aft" (rigid contact, DLFT part of DLFTContactAFT)
                           # | "dlft" (rigid contact, DLFTContact) | "aft" (regularized penalty)
FRF_SOURCE = "numerical"   # "numerical" (modal synthesis) | "experimental" (sampled + noisy)

# --- rods (Vadcard Table 1) ---
# Every physical value here, n_elem and sample_number must match the parameter
# block of NLvib_verification/two_rod_unilateral_contact/two_rod_nlvib_reference.m;
# plotting.py warns about any mismatch within a study folder.
PARAMS = RodParams(n_elem=20, F0=25e3, sample_number=1000)    # rod A
LB_REL = 1            # rod B length L_B / L_A  (1.0 = identical rods)

# --- contact ---
GAP         = 0.2e-3     # g0: initial tip-to-tip gap [m]
# DLFT penalty factors eps / k_rod (DLFT methods only), one branch per value.  eps
# conditions the iteration, and through the harmonic truncation it also shifts
# the converged contact state slightly (residual penetration, ripple).
EPSILON_REL = np.arange(5, 10, 1)
K_REL       = 100.0      # AFT penalty stiffness k_c / k_rod    (AFT only)
ALPHA       = 1e8        # AFT tanh-regularization sharpness    (np.inf = hard, nonsmooth)

# --- experimental FRF (only used when FRF_SOURCE == "experimental") ---
DENSITY    = 10.0        # measured-FRF density [samples/Hz]
NOISE      = np.inf      # measured-FRF SNR [dB]  (np.inf = clean)
NOISE_SEED = 1

# --- frequency window + solver (PHYSICAL omega [rad/s]) ---
# Window is given as fractions of the first coupled mode omega_1 and scaled to rad/s
# where the system is available (solve_branch / main).  STEP_KWARGS lives in the
# solver's scaled metric: Dscale[-1] = omega_ref makes a step advance
# ~step_length * omega_1.
# SWEEP picks the edge the branch starts from: "down" starts at OMEGA_HIGH_REL and
# runs toward OMEGA_LOW_REL, "up" the reverse.  With identical rods the contact
# branch bends far above omega_1; keep both edges clear of 3*omega_1, the second
# axial mode of a clamped-free rod.  Window and SWEEP must match P.omega_low_rel /
# P.omega_high_rel / P.sweep of the NLvib reference script.
HARMONICS       = list(range(0, 31))         # 0..20 -> H = 20
OMEGA_LOW_REL   = 0.8                         # window [OMEGA_LOW_REL, OMEGA_HIGH_REL] * omega_1
OMEGA_HIGH_REL  = 1.2
SWEEP           = "down"                      # "down" | "up"
SOLVER_KWARGS = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
STEP_KWARGS   = {"base": 2.0, "initial_step_length": 0.001, "maximum_step_length": 0.001,
                 "minimum_step_length": 1e-10, "goal_number_of_iterations": 3}
MAX_SOLUTIONS = 10000

# --- output ---
RESULT_DIR  = "LB_REL=5"   # study folder inside Results/
RESULT_NAME = None                  # None -> pyfbs_<METHOD>; DLFT branches get _eps<value>
N_WORKERS   = 1                     # branches solved in parallel processes (1 = one after another)


# response signals: driven tip u_A and relative approach x_r = u_A + u_B
SIGNALS     = ("tipA", "xr")
SIGNAL_TEX  = {"tipA": r"u_A", "xr": r"x_r"}
SIGNAL_DESC = {"tipA": "driven tip $u_A$",
               "xr":   "relative approach $x_r = u_A + u_B$"}


# ============================ helpers (self-contained) ======================

def linear_relative(system, omega):
    """Linear (no-contact) relative interface response x_r = B Y f_ext at frequency
    omega, from the system's DAMPED modal data (the same admittance the solver uses).

    Below the gap the DLFT branch carries no contact force, so x_r reduces to exactly
    this linear response -- the two curves must overlay there.  Synthesized from the
    system's modal data (Omega, Phi, zeta).
    """
    Om, Phi, z = system.angular_eig_freq, system.eig_vec, system.zeta
    Y = Phi @ np.diag(1.0 / (Om ** 2 - omega ** 2 + 2j * z * Om * omega)) @ Phi.T
    F = np.zeros((system.total_dimension, 1)); F[system.rod_tip_idx, 0] = system.F0
    return (system.B_coupling @ (Y @ F))[0, 0]


def make_experimental_provider(system):
    """Sample the system admittance on a measurement grid (+ optional noise).

    Grid spacing is 1/DENSITY Hz on the physical axis up to the top queried
    harmonic; optional pyFBS-style measurement noise at SNR=NOISE.
    """
    h_max     = int(np.max(HARMONICS))
    omega_hi  = OMEGA_HIGH_REL * system.omega_ref                                 # rad/s
    omega_max = h_max * omega_hi * 1.05
    f_max     = omega_max / (2.0 * np.pi)
    n_freq    = max(2, int(round(DENSITY * f_max)) + 1)
    grid      = np.linspace(0.0, omega_max, n_freq)                               # rad/s grid

    # physical damped admittance on the grid, synthesized from the modal data (no C
    # matrix); the modal data is already physical so no omega_ref rescaling is needed.
    prov = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    d = system.total_dimension
    all_dofs = np.arange(d)
    Y = prov.compute_FRF(1.0, grid, all_dofs, all_dofs)   # full (n_freq, d, d)

    if np.isfinite(NOISE):                       # proportional jitter + additive floor
        rng = np.random.default_rng(NOISE_SEED)
        s = 10.0 ** (-NOISE / 20.0)
        for i in range(d):
            for j in range(d):
                y_abs = np.abs(Y[:, i, j]); floor = np.median(y_abs)
                g = rng.standard_normal((n_freq, 4))
                Y[:, i, j] += (y_abs * (s * g[:, 0] + 1j * s * g[:, 1])
                               + floor * (s * g[:, 2] + 1j * s * g[:, 3]))
    return ExperimentalFRF(grid, Y, fd_step=1e-6)


def build(epsilon_rel=None):
    """Return (system, provider, method) for the configuration above.

    :param epsilon_rel: DLFT penalty factor eps / k_rod of this branch (DLFT
        methods only).
    """
    params_B = replace(PARAMS, L=LB_REL * PARAMS.L)     # rod B (length L_B = LB_REL * L_A)

    if METHOD == "dlft_aft":
        system = TwoRodVibroImpact(PARAMS, params_B)
        method = DLFTContactAFT(L_N, L_F, epsilon=epsilon_rel * system.k_rod, g_zero=GAP)
    elif METHOD == "dlft":
        system = TwoRodVibroImpact(PARAMS, params_B)
        method = DLFTContact(epsilon=epsilon_rel * system.k_rod, g_zero=GAP)
    elif METHOD == "aft":
        system = TwoRodPenaltyContact(PARAMS, params_B, k_rel=K_REL, g_zero=GAP, alpha=ALPHA)
        method = AFT()
    else:
        raise ValueError(f"METHOD must be 'dlft_aft', 'dlft' or 'aft', got {METHOD!r}")

    if FRF_SOURCE == "numerical":
        provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    elif FRF_SOURCE == "experimental":
        provider = make_experimental_provider(system)
    else:
        raise ValueError(f"FRF_SOURCE must be 'numerical' or 'experimental', got {FRF_SOURCE!r}")

    return system, provider, method


def run_config(system, epsilon_rel=None):
    """Parameter set of this run, written into the result CSV as config_json.

    The physical keys carry the same names as in the NLvib reference script, so
    that plotting.py can check that a study folder compares one and the same
    system.
    """
    config = dict(
        solver="pyfbs-nlfbs", method=METHOD, frf_source=FRF_SOURCE,
        n_elem=PARAMS.n_elem, L_A=PARAMS.L, LB_rel=LB_REL, E=PARAMS.E,
        rho=PARAMS.rho, A=PARAMS.A, xi=PARAMS.xi, F0=PARAMS.F0, gap=GAP,
        harmonics=[int(h) for h in HARMONICS], sample_number=system.sample_number,
        omega_1=float(system.omega_ref),
        omega_low_rel=OMEGA_LOW_REL, omega_high_rel=OMEGA_HIGH_REL, sweep=SWEEP,
        solver_kwargs=SOLVER_KWARGS, step_kwargs=STEP_KWARGS,
    )
    if METHOD in ("dlft_aft", "dlft"):
        config.update(law="rigid", epsilon_rel=epsilon_rel)
    else:
        config.update(law="tanh" if np.isfinite(ALPHA) else "penalty", k_rel=K_REL,
                      alpha=float(ALPHA) if np.isfinite(ALPHA) else None)
    if FRF_SOURCE == "experimental":
        config.update(density=DENSITY, noise_db=NOISE if np.isfinite(NOISE) else None,
                      noise_seed=NOISE_SEED)
    return config


def solve_branch(system, provider, method, verbose=True):
    """Run one arc-length DLFT/AFT-HBM branch.

    :param verbose: live progress line of the continuation.
    :returns: (omega_phys, {signal: peak}, solution set, problem, solve time [s])
    """
    HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)
    problem = FBSProblem(system, provider, method)
    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, freq_domain_ode=problem,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)


    # PHYSICAL sweep window: fractions of the first coupled mode -> rad/s.  The
    # branch starts at the edge SWEEP names, and the sign of the reference
    # direction orients the first tangent toward the other edge.
    w1 = system.omega_ref
    w_low, w_high = OMEGA_LOW_REL * w1, OMEGA_HIGH_REL * w1
    if SWEEP == "down":
        w_start, direction = w_high, -1.0
    elif SWEEP == "up":
        w_start, direction = w_low, +1.0
    else:
        raise ValueError(f"SWEEP must be 'down' or 'up', got {SWEEP!r}")

    Q1 = np.array([[linear_relative(system, w_start)]])
    ig = FourierOmegaPoint.new_from_first_harmonic(Q1, omega=w_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=direction)

    t0 = perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = MAX_SOLUTIONS,
        angular_frequency_range       = [w_low, w_high],
        solver_kwargs                 = SOLVER_KWARGS,
        step_length_adaptation_kwargs = STEP_KWARGS,
        jacobian_update_frequency     = 1,
        verbose                       = verbose,
    )
    solve_time = perf_counter() - t0
    omega_phys = np.array(ss.omega)
    peak = {s: np.zeros(len(omega_phys)) for s in SIGNALS}
    for i, (four, o_hat) in enumerate(zip(ss.fourier, ss.omega)):
        full = problem.compute_full_response(four, o_hat)
        Fourier_Real.compute_time_series(full)
        uA = full.time_series[:, system.rod_tip_idx, 0]
        uB = full.time_series[:, system.tipB_idx, 0]
        peak["tipA"][i] = float(np.max(np.abs(uA)))
        peak["xr"][i]   = float(np.max(np.abs(uA + uB)))
    print(f"  {len(omega_phys)} points in {solve_time:.1f} s")
    return omega_phys, peak, ss, problem, solve_time


def epsilon_values():
    """The eps / k_rod values to solve: the EPSILON_REL list for the DLFT
    methods, a single None for AFT, which has no eps."""
    return [float(e) for e in EPSILON_REL] if METHOD in ("dlft_aft", "dlft") else [None]


def result_name(epsilon_rel):
    """CSV name of one branch: RESULT_NAME (or pyfbs_<METHOD>), with _eps<value>
    appended for a DLFT branch, so the branches of one sweep do not overwrite
    each other."""
    stem = Path(RESULT_NAME).stem if RESULT_NAME else f"pyfbs_{METHOD}"
    return f"{stem}.csv" if epsilon_rel is None else f"{stem}_eps{epsilon_rel:g}.csv"


def run_branch(epsilon_rel):
    """Build, solve and save one branch; runs in a worker process when N_WORKERS > 1.

    A branch that fails (e.g. a singular Jacobian at the first point for a large
    eps) is reported and skipped, so it does not cost the other branches.

    :returns: (epsilon_rel, omega_phys, {signal: peak}), the last two None if the
        branch failed.
    """
    tag = "" if epsilon_rel is None else f", eps = {epsilon_rel:g} k_rod"
    print(f"start: two-rod {METHOD.upper()} / {FRF_SOURCE}{tag}")
    try:
        system, provider, method = build(epsilon_rel)
        omega_phys, peak, ss, problem, solve_time = solve_branch(
            system, provider, method, verbose=N_WORKERS == 1)
        plotting.save_result(plotting.RESULTS / RESULT_DIR / result_name(epsilon_rel),
                             run_config(system, epsilon_rel), ss, problem, system,
                             solve_time)
    except Exception as exc:
        print(f"FAILED: two-rod {METHOD.upper()}{tag}: {exc!r}")
        return epsilon_rel, None, None
    return epsilon_rel, omega_phys, peak


def main():
    values = epsilon_values()
    system, _, _ = build(values[0])       # omega_1 and the linear backdrop do not depend on eps
    omega_1 = system.omega_ref
    print(f"two-rod {METHOD.upper()} / {FRF_SOURCE}  |  L_B = {LB_REL:g} L_A,  "
          f"omega_1 = {omega_1:.1f} rad/s ({omega_1/2/np.pi:.1f} Hz)  |  "
          f"{len(values)} branch(es), {N_WORKERS} worker(s)")

    if N_WORKERS > 1 and len(values) > 1:
        with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(values))) as pool:
            results = list(pool.map(run_branch, values))
    else:
        results = [run_branch(value) for value in values]
    results = [result for result in results if result[1] is not None]

    # linear (no-contact) backdrop: rod B is unforced without contact (u_B = 0), so
    # the same curve backs both the u_A and the x_r window.
    om_lin   = np.linspace(OMEGA_LOW_REL * omega_1, OMEGA_HIGH_REL * omega_1, 600)
    peak_lin = np.array([abs(linear_relative(system, w)) for w in om_lin])

    SCALE = 1.0e-4
    XLIM  = (OMEGA_LOW_REL * omega_1, OMEGA_HIGH_REL * omega_1)
    YLIM  = (0.0, 5.5)
    colors = (["#E8820C"] if len(results) == 1
              else plt.cm.viridis(np.linspace(0.0, 0.9, max(len(results), 1))))

    for sig in SIGNALS:
        tex = SIGNAL_TEX[sig]
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.plot(om_lin, peak_lin / SCALE, ':', color="k", lw=1.0,
                label="Linear FRF (no contact)")
        ax.axhline(GAP / SCALE, color="red", ls="--", lw=1.2,
                   label=r"Contact threshold $g_0$")
        for (epsilon_rel, omega_phys, peak), color in zip(results, colors):
            tag = "" if epsilon_rel is None else f", eps = {epsilon_rel:g} k_rod"
            ax.plot(omega_phys, peak[sig] / SCALE, '-', color=color, lw=1.8,
                    label=f"nonlinearFBS {METHOD.upper()} ({FRF_SOURCE}){tag}")

        ax.set_title(f"NFRC: two-rod vibro-impact -- {SIGNAL_DESC[sig]}", fontsize=11)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_ylabel(rf"$\|{tex}(t)\|_\infty$  [$\times 10^{{-4}}$ m]")
        ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
