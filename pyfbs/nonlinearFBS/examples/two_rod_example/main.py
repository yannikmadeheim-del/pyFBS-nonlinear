"""Two-rod vibro-impact -- single-system quicklook (standalone).

Two identical clamped-free FE rods face each other across a gap g0; the left rod
is harmonically driven at its tip, the right rod is an identical passive mirror.
Configure ONE system here (contact method, FRF source, parameters), solve its
nonlinear frequency response, and plot the driven tip u_A and the relative
approach x_r = u_A + u_B.


External reference (off by default)
-----------------------------------
Set ``REFERENCE_CSV`` to the name of a CSV file placed in THIS folder to overlay an
external reference NFRC on the plots.  Expected columns:

    omega        physical angular frequency [rad/s]
    A_tipA       (optional) peak |u_A(t)| [m]
    A_xr         (optional) peak |x_r(t)| [m]

Only the signals whose column is present are overlaid.

Run:  python main.py
"""
import sys
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from dataclasses import replace
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, ExperimentalFRF,
    DLFTContact, AFT, HarmonicBalanceMethod,
    ArcLengthParameterization, TangentPredictorBordered,
)

from dynamical_system import RodParams, TwoRodVibroImpact, TwoRodPenaltyContact


# ============================ configuration =================================
# --- which model to solve ---
METHOD = "dlft"           # "dlft" (rigid contact) | "aft" (regularized penalty)
FRF    = "numerical"     # "numerical" (exacM,C,K) | "experimental" (sampled + noisy)

# --- rods (Vadcard Table 1) ---
PARAMS = RodParams(F0=25e3, sample_number = 1000)    # rod A
LB_REL = 1/20             # rod B length L_B / L_A  (1.0 = identical rods)

# --- contact ---
GAP         = 0.2e-3     # g0: initial tip-to-tip gap [m]
EPSILON_REL = 0.5     # DLFT penalty = EPSILON_REL * k_rod   (DLFT only; eps-independent)
K_REL       = 100.0      # AFT penalty stiffness k_c / k_rod    (AFT only)
ALPHA       = 1e8        # AFT tanh-regularization sharpness    (np.inf = hard, nonsmooth)

# --- experimental FRF (only used when FRF == "experimental") ---
DENSITY    = 0.1        # measured-FRF density [samples/Hz]
NOISE      = np.inf      # measured-FRF SNR [dB]  (np.inf = clean)
NOISE_SEED = 1

# --- frequency window + solver (PHYSICAL omega [rad/s]) ---
# Window is given as fractions of the first coupled mode omega_1 and scaled to rad/s
# where the system is available (solve_branch / main).  STEP_KWARGS is UNCHANGED: the
# solver metric scaling Dscale[-1] = omega_ref makes a step advance ~step_length * omega_1,
# so 0.002 still yields ~150 points across the window whether omega is dimensional or not.
HARMONICS       = list(range(0, 21))         # 0..20 -> H = 20
OMEGA_START_REL = 1.2                         # sweep 1.2*omega_1 ...
OMEGA_END_REL   = 0.9                         # ... down to 0.9*omega_1
SOLVER_KWARGS = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
STEP_KWARGS   = {"base": 4.0, "initial_step_length": 0.002, "maximum_step_length": 0.005,
                 "minimum_step_length": 1e-8, "goal_number_of_iterations": 3}
MAX_SOLUTIONS = 10000

# --- output ---
SAVE_PNG      = False    # set True to write the figures to disk
REFERENCE_CSV = None     # e.g. "reference_two_rod.csv" (in this folder) to overlay; None = off


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
    modal data (Omega, Phi, zeta); the system no longer carries a C matrix.
    """
    Om, Phi, z = system.angular_eig_freq, system.eig_vec, system.zeta
    Y = Phi @ np.diag(1.0 / (Om ** 2 - omega ** 2 + 2j * z * Om * omega)) @ Phi.T
    F = np.zeros((system.total_dimension, 1)); F[system.rod_tip_idx, 0] = system.F0
    return (system.B_coupling @ (Y @ F))[0, 0]


def make_experimental_provider(system):
    """Sample the system admittance on a measurement grid (+ optional noise).

    Mirrors studies/frf.py: grid spacing is 1/DENSITY Hz on the physical axis up to
    the top queried harmonic; optional pyFBS-style measurement noise at SNR=NOISE.
    """
    h_max     = int(np.max(HARMONICS))
    omega_hi  = max(abs(OMEGA_START_REL), abs(OMEGA_END_REL)) * system.omega_ref  # rad/s
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


def build():
    """Return (system, provider, method) for the configuration above."""
    params_B = replace(PARAMS, L=LB_REL * PARAMS.L)     # rod B (length L_B = LB_REL * L_A)

    if METHOD == "dlft":
        system = TwoRodVibroImpact(PARAMS, params_B)
        method = DLFTContact(epsilon=EPSILON_REL * system.k_rod, g_zero=GAP)
    elif METHOD == "aft":
        system = TwoRodPenaltyContact(PARAMS, params_B, k_rel=K_REL, g_zero=GAP, alpha=ALPHA)
        method = AFT()
    else:
        raise ValueError(f"METHOD must be 'dlft' or 'aft', got {METHOD!r}")

    if FRF == "numerical":
        provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
    elif FRF == "experimental":
        provider = make_experimental_provider(system)
    else:
        raise ValueError(f"FRF must be 'numerical' or 'experimental', got {FRF!r}")

    return system, provider, method


def solve_branch(system, provider, method):
    """Run one arc-length DLFT/AFT-HBM branch; return (omega_phys, {signal: peak})."""
    HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)
    problem = FBSProblem(system, provider, method)
    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, freq_domain_ode=problem,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)


    # PHYSICAL sweep window: fractions of the first coupled mode -> rad/s
    w1 = system.omega_ref
    w_start, w_end = OMEGA_START_REL * w1, OMEGA_END_REL * w1

    Q1 = np.array([[linear_relative(system, w_start)]])
    ig = FourierOmegaPoint.new_from_first_harmonic(Q1, omega=w_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=-1.0)

    t0 = perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = MAX_SOLUTIONS,
        angular_frequency_range       = [w_start, w_end],
        solver_kwargs                 = SOLVER_KWARGS,
        step_length_adaptation_kwargs = STEP_KWARGS,
        jacobian_update_frequency     = 1,
    )
    omega_phys = np.array(ss.omega)
    peak = {s: np.zeros(len(omega_phys)) for s in SIGNALS}
    for i, (four, o_hat) in enumerate(zip(ss.fourier, ss.omega)):
        full = problem.compute_full_response(four, o_hat)
        Fourier_Real.compute_time_series(full)
        uA = full.time_series[:, system.rod_tip_idx, 0]
        uB = full.time_series[:, system.tipB_idx, 0]
        peak["tipA"][i] = float(np.max(np.abs(uA)))
        peak["xr"][i]   = float(np.max(np.abs(uA + uB)))
    print(f"  {len(omega_phys)} points in {perf_counter() - t0:.1f} s")
    return omega_phys, peak


def load_reference():
    """External reference NFRC from a CSV in this folder, or None if off/missing."""
    if not REFERENCE_CSV:
        return None
    import pandas as pd
    path = Path(__file__).parent / REFERENCE_CSV
    if not path.exists():
        print(f"WARNING: reference CSV not found: {path}  -- overlay skipped")
        return None
    return pd.read_csv(path)


def main():
    system, provider, method = build()
    omega_1 = system.omega_ref
    print(f"two-rod {METHOD.upper()} / {FRF}  |  L_B = {LB_REL:g} L_A,  "
          f"omega_1 = {omega_1:.1f} rad/s ({omega_1/2/np.pi:.1f} Hz)")

    omega_phys, peak = solve_branch(system, provider, method)

    # linear (no-contact) backdrop: rod B is unforced without contact (u_B = 0), so
    # the same curve backs both the u_A and the x_r window.
    om_lin   = np.linspace(OMEGA_START_REL * omega_1, OMEGA_END_REL * omega_1, 600)
    peak_lin = np.array([abs(linear_relative(system, w)) for w in om_lin])


    ref = load_reference()

    SCALE = 1.0e-4
    XLIM  = tuple(sorted((OMEGA_START_REL * omega_1, OMEGA_END_REL * omega_1)))
    YLIM  = (0.0, 5.5)

    for sig in SIGNALS:
        tex = SIGNAL_TEX[sig]
        fig, ax = plt.subplots(figsize=(8.0, 5.5))
        ax.plot(om_lin, peak_lin / SCALE, ':', color="k", lw=1.0,
                label="Linear FRF (no contact)")
        ax.axhline(GAP / SCALE, color="red", ls="--", lw=1.2,
                   label=r"Contact threshold $g_0$")
        if ref is not None and f"A_{sig}" in ref:
            ax.plot(ref["omega"], ref[f"A_{sig}"] / SCALE, '-', color="0.6", lw=3.0,
                    zorder=1, label=f"reference ({REFERENCE_CSV})")
        ax.plot(omega_phys, peak[sig] / SCALE, '-', color="#E8820C", lw=1.8,
                label=f"pyhbm {METHOD.upper()}-FBS ({FRF})")

        ax.set_title(f"NFRC: two-rod vibro-impact -- {SIGNAL_DESC[sig]}", fontsize=11)
        ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_ylabel(rf"$\|{tex}(t)\|_\infty$  [$\times 10^{{-4}}$ m]")
        ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")
        fig.tight_layout()

        if SAVE_PNG:
            out = Path(__file__).parent / f"two_rod_frc_{sig}.png"
            fig.savefig(out, dpi=150)
            print(f"Figure saved: {out}")

    plt.show()


if __name__ == "__main__":
    main()
