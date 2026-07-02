"""
Clamped-free FE rod impacting a FLEXIBLE obstacle (Vadcard, Batailly & Thouverez,
J. Sound Vib. 531 (2022) 116950, Fig. 6/17 + Table 1).

The flexible wall is modelled as a SECOND SUBSTRUCTURE -- a grounded linear spring
k_obs -- coupled to the rod through Frequency-Based Substructuring (FBS).  The
unmodified, *rigid* DLFT contact then acts on the RELATIVE interface DOF

        x_r(t) = u_B(t) - u_w(t),        contact when  x_r > g0,

where u_B is the rod free-end displacement and u_w the obstacle-spring node.  The
obstacle compliance 1/k_obs enters the interface admittance Y_r = B Y B^T
automatically, so at convergence the contact force satisfies the obstacle law
        lambda = k_obs * (u_B - g0)     (penetration resisted by finite stiffness)
and the rigid wall is recovered as k_obs -> infinity.  No change to the DLFT
nonlinear-method code is needed: the flexibility lives entirely in the FRF.

Model (Vadcard Table 1):
    A = 15.6 cm^2,  E = 210 GPa,  rho = 7800 kg/m^3,  L = 13 cm,
    g0 = 0.2 mm,  n = 20 two-node bar elements (left end A clamped, right end B free).
    element matrices  M_e = rho*A*l/6 [[2,1],[1,2]],  K_e = E*A/l [[1,-1],[-1,1]].
    harmonic forcing f_ex = 25e3 N at node B;  k_rod = E*A/L ~ 2.5e9 N/m.

Obstacle (flexible wall):  k_obs = k_rel * k_rod,  k_rel = k_obs / k_rod the
relative obstacle stiffness.  This script reproduces Vadcard Fig. 17: the NFRC for
k_rel = 0.4, 4, 20, 40 (weakly -> strongly nonlinear), as a 2x2 panel grid.
"""
import sys
from pathlib import Path

# Live progress (no block buffering) + UTF-8 so the continuation's "Δω" print
# does not crash under Windows' default cp1252 console encoding.
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

from time import time
import numpy as np
from numpy import zeros

import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod,
    ArcLengthParameterization, TangentPredictorBordered,
)


from dynamical_system import RodVibroImpactFlexible


# ============================ parameters ====================================
GAP       = 0.2e-3       # g0: wall offset [m]
HARMONICS = list(range(0, 21))   # 0..20  -> H = 20
POLY_DEG  = 30                   # N = (POLY_DEG+1)*Highest_Harmonic + 1
F0        = 25e3                 # f_ex: harmonic forcing at node B [N]
SAVE_PNG  = False                # set True to write the figure to disk

# Obstacle stiffnesses to sweep, as multiples of k_rod = E*A/L
# (Vadcard Fig. 17 a-d:  k_obs = 0.4, 4, 20, 40 x k_rod).
K_REL_VALUES = [0.4, 4.0, 20.0, 40.0]

# default system (k_rel = 1) used for the omega_1 reference, FD checks and the
# linear (no-contact) FRF -- all of which are independent of k_obs.
rod = RodVibroImpactFlexible(k_rel=1.0, F0=F0, poly_deg=POLY_DEG)
OMEGA_1 = rod.omega_modes[0]
# DLFT penalty: stiffness units; large vs the interface dynamic stiffness ~ k_rod.
# Converged solution is epsilon-independent (Vadcard 2022), so any large value works.
EPSILON   = 2.0e1 * rod.k_rod

print(f"first axial mode: omega_1 = {OMEGA_1:.1f} rad/s  ({OMEGA_1/2/np.pi:.1f} Hz)")
print(f"rod static stiffness k_rod = E*A/L = {rod.k_rod:.3e} N/m")
print(f"sweeping k_rel = k_obs/k_rod in {K_REL_VALUES}")
print(f"DLFT penalty epsilon = {EPSILON:.3e}")

# continuation runs in the nondimensional frequency w_hat = w / omega_1.
# Plot window matches Vadcard Fig. 17.
OMEGA_START = 1.20              # w_hat
OMEGA_END   = 0.90               # w_hat  (room for the contact stiffening overhang)


# ============================ build problem (default) =======================

HarmonicBalanceMethod.update_dependencies(HARMONICS, rod.polynomial_degree)
provider = NumericalFRF(rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(rod, provider, contact)
print(f"DOFs: total = {rod.total_dimension} (rod {rod.total_dimension-1} + obstacle 1), "
      f"interface = {rod.dimension}, "
      f"N_time = {Fourier.number_of_time_samples}, N_h = {Fourier.number_of_harmonics}")


# ============================ initial guess (linear) ========================

def linear_relative(omega):
    """Linear (no-contact) relative interface response x_r = B u at frequency omega.

    In the linear regime the obstacle carries no force (u_w = 0), so x_r = u_B; the
    result is therefore independent of k_obs.
    """
    M, C, K = rod.mass_matrix, rod.damping_matrix, rod.stiffness_matrix
    Z = -omega**2 * M + 1j * omega * C + K
    F = zeros((rod.total_dimension, 1)); F[rod.rod_tip_idx, 0] = rod.F0
    u = np.linalg.solve(Z, F)
    return (rod.B_coupling @ u)[0, 0]    # complex relative amplitude (1st harmonic)

Q1_start = abs(linear_relative(OMEGA_START))
print(f"linear free-end amplitude at omega_start: {Q1_start:.3e} m  (gap g0 = {GAP:.1e} m)")


# ============================ continuation ==================================

solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-6}
step_kwargs = {
    "base":                      4.0,
    "initial_step_length":       0.002,
    "maximum_step_length":       0.001,   # narrow w_hat window -> keep steps small
    "minimum_step_length":       1e-7,
    "goal_number_of_iterations": 3,
}

def run_frc(k_rel_value,
            parameterization=ArcLengthParameterization,
            predictor=TangentPredictorBordered):
    """Continuation sweep for one obstacle stiffness  k_obs = k_rel_value * k_rod.

    Rebuilds the rod + grounded-spring substructure system for this k_obs, runs the
    DLFT-HBM continuation, and returns (omega_phys, peak_uB) along the branch.
    peak_uB = ||u_B(t)||_inf (max over a period of the rod free-end displacement).
    Branch order is preserved -- do NOT sort, that would break the fold.
    """
    sys_k  = RodVibroImpactFlexible(k_rel=k_rel_value, F0=F0, poly_deg=POLY_DEG)
    prov_k = NumericalFRF(sys_k.mass_matrix, sys_k.damping_matrix, sys_k.stiffness_matrix)
    cont_k = DLFTContact(epsilon=EPSILON, g_zero=GAP)
    prob_k = FBSProblem(sys_k, prov_k, cont_k)
    solver_k = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=prob_k,
                                     corrector_parameterization=parameterization,
                                     predictor=predictor)

    Q1_l = np.array([[linear_relative(OMEGA_START)]])
    ig   = FourierOmegaPoint.new_from_first_harmonic(Q1_l, omega=OMEGA_START)
    rd   = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=-1.0)

    print(f"\nContinuation: k_obs = {k_rel_value:g} k_rod  (k_obs = {sys_k.k_obs:.3e} N/m)")
    t0 = time()
    ss = solver_k.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = 5000,
        angular_frequency_range       = [OMEGA_START, OMEGA_END],
        solver_kwargs                 = solver_kwargs,
        step_length_adaptation_kwargs = step_kwargs,
        jacobian_update_frequency     = 1,
    )
    omega_hat  = np.array(ss.omega)
    omega_phys = omega_hat * OMEGA_1
    peak = np.zeros_like(omega_phys)
    for i, (four, o_hat) in enumerate(zip(ss.fourier, omega_hat)):
        full = prob_k.compute_full_response(four, o_hat)
        Fourier_Real.compute_time_series(full)
        peak[i] = float(np.max(np.abs(full.time_series[:, sys_k.rod_tip_idx, 0])))
    print(f"-> {len(omega_phys)} points, omega in "
          f"[{omega_phys.min():.1f}, {omega_phys.max():.1f}] rad/s, "
          f"peak/1e-4 in [{peak.min()/1e-4:.2f}, {peak.max()/1e-4:.2f}], {time()-t0:.1f} s")
    # return branch + raw solution set (needed for warm-starting the rigid case)
    return omega_phys, peak, ss, sys_k


# one continuation branch per obstacle stiffness
raw = {kv: run_frc(kv) for kv in K_REL_VALUES}
results = {kv: (raw[kv][0], raw[kv][1]) for kv in K_REL_VALUES}


# ============================ linear FRF (reference) ========================
# Pure-harmonic linear (no-contact) relative response; same for every k_obs (the
# obstacle is decoupled until contact). Diverges at omega_1 -> clipped by y-limit.
wh_lin   = np.linspace(OMEGA_START, OMEGA_END, 600)
peak_lin = np.array([abs(linear_relative(w)) for w in wh_lin])
om_lin   = wh_lin * OMEGA_1


# ============================ plot (Vadcard Fig. 17 style) ==================
# 2x2 grid, one panel per k_obs.  Each panel: linear FRF (dotted), contact
# threshold g0 (red dashed), and the DLFT-HBM flexible-obstacle NFRC (orange).

SCALE = 1.0e-4              # y-axis in units of 1e-4 m
XLIM  = (5.8e4, 7.4e4)      # Vadcard Fig. 17 frequency window [rad/s]
YLIM  = (0.0, 5.5)
PANEL = ["(a)", "(b)", "(c)", "(d)"]

import pandas as pd
# CSVs are saved by MATLAB next to this script (MATLAB's cwd when run from editor)
NLVIB_DIR = Path(__file__).parent

def _nlvib_csv(kv):
    """Return the NLvib CSV path for a given k_rel value, tolerating int/float names."""
    # MATLAB writes e.g. 4 not 4.0, but 0.4 stays as 0.4
    stem = f"nlvib_rod_flexible_kobs_{kv:g}_krod.csv"
    return NLVIB_DIR / stem

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
for ax, lbl, kv in zip(axes.ravel(), PANEL, K_REL_VALUES):
    om_k, peak_k = results[kv]

    # --- data sources ---
    ax.plot(om_lin, peak_lin / SCALE, ':',  color="k", lw=1.0,
            label="Linear FRF (no contact)")
    ax.axhline(GAP / SCALE, color="red", ls="--", lw=1.2,
               label=r"Contact threshold $g_0$")
    ax.plot(om_k, peak_k / SCALE, '-', color="#E8820C", lw=1.8,
            label="pyhbm DLFT-FBS\n(2-substructure, flexible wall)")

    # NLvib independent reference (green dashed)
    csv = _nlvib_csv(kv)
    if csv.exists():
        df = pd.read_csv(csv)
        ax.plot(df["omega"], df["A_peak"] / SCALE, '--', color="#2CA02C", lw=1.4,
                zorder=5, label="NLvib HBM reference\n(penalty unilateral spring, MATLAB)")
    else:
        print(f"  [overlay] CSV not found: {csv.name}")

    ax.set_title(rf"{lbl}  $k_\mathrm{{obs}} = {kv:g}\,k_\mathrm{{rod}}$", fontsize=11)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\|x(t)\|_\infty$  [$\times 10^{-4}$ m]")
for ax in axes[1, :]:
    ax.set_xlabel(r"$\omega$  [rad$\cdot$s$^{-1}$]")

fig.suptitle(
    "NFRC: clamped-free FE rod vs. flexible obstacle  (Vadcard JSV 531, 2022, Fig. 17)\n"
    r"Orange $-$: pyhbm DLFT-FBS (2-substructure, rod + grounded spring, Python)   "
    r"Black $\cdot$: NLvib HBM (penalty unilateral spring, MATLAB)",
    fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])

if SAVE_PNG:
    out = Path(__file__).parent / "rod_vibroimpact_frc.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure saved: {out}")


plt.show()
