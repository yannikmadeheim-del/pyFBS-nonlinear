"""
SDOF vibro-impact validation: pyhbm DLFT vs. NLvib shooting (rigid-wall reference).

Solves the same SDOF mass-spring-damper-wall as the NLvib shooting script
using the new FBSProblem + NumericalFRF + DLFTContact API, then overlays the
two FRCs.
"""
from pathlib import Path

from pyfbs.nonlinearFBS.numerical_continuation.corrector_step import ArcLengthParameterization

from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    FBSProblem, NumericalFRF, DLFTContact,
    HarmonicBalanceMethod,
)


from dynamical_system import SDOFVibroImpact


# ============================ parameters ====================================
EPSILON   = 1.0   # DLFT penalty (stiffness units); converged solution is eps-independent
PARAMS = dict(m=1.0, c=0.05, k=1.0, F0=0.02)   # c=0.1 -> linear amp at res ~2*g0
GAP       = 0.1
HARMONICS = list(range(0, 30))
SAVE_PNG  = False   # set True to write the figure to disk

OMEGA_START = 0.5
OMEGA_END   = 2.5


# ============================ build problem (NEW API) ======================

system   = SDOFVibroImpact(**PARAMS)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.polynomial_degree)

provider = NumericalFRF(system.mass_matrix, system.damping_matrix, system.stiffness_matrix)
contact  = DLFTContact(epsilon=EPSILON, g_zero=GAP)
problem  = FBSProblem(system, provider, contact)


# ============================ initial guess =================================

initial_guess               = FourierOmegaPoint.zero_amplitude(dimension=system.dimension, omega=OMEGA_START)
initial_reference_direction = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=1.0)


# ============================ continuation ==================================

solver_kwargs = {
    "maximum_iterations": 300,
    "absolute_tolerance": 1e-6,
}
step_kwargs = {
    "base":                      2.0,
    "initial_step_length":       0.005,
    "maximum_step_length":       1.0,
    "minimum_step_length":       1e-6,
    "goal_number_of_iterations": 4,
}

print("\n" + "=" * 70)
print(f"DLFT continuation at ε = {EPSILON:.1e}")
print("=" * 70)

solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization)
t0 = time()
solution_set = solver.solve_and_continue(
    initial_guess                 = initial_guess,
    initial_reference_direction   = initial_reference_direction,
    maximum_number_of_solutions   = 10000,
    angular_frequency_range       = [OMEGA_START, OMEGA_END],
    solver_kwargs                 = solver_kwargs,
    step_length_adaptation_kwargs = step_kwargs,
    jacobian_update_frequency     = 1,    # full Newton -- no Jacobian reuse
)
om_now = np.array(solution_set.omega)
print(f"  -> {len(om_now)} points, "
      f"ω range [{om_now.min():.4f}, {om_now.max():.4f}]  in {time()-t0:.1f} s")

# ============================ post-process amplitudes =====================

Nt     = Fourier.number_of_time_samples
omegas = np.array(solution_set.omega)
A1     = np.zeros_like(omegas)
Apeak  = np.zeros_like(omegas)

for i, (four, om) in enumerate(zip(solution_set.fourier, omegas)):
    full = problem.compute_full_response(four, om)
    A1[i] = 2.0 / Nt * np.abs(full.coefficients[1, 0, 0])
    Fourier_Real.compute_time_series(full)
    Apeak[i] = float(np.max(np.abs(full.time_series[:, 0, 0])))


# ============================ NLvib reference =============================

ref_csv = Path(__file__).parent / "nlvib_sdof_vibroimpact_shooting_frc.csv"
df = pd.read_csv(ref_csv) if ref_csv.exists() else None
if df is None:
    print(f"\nWARNING: NLvib CSV not found at {ref_csv}. Overlay skipped.")


# ============================ plot =========================================

fig, ax = plt.subplots(figsize=(8.5, 5.5))
if df is not None:
    ax.plot(df["omega"], df["A1"],     'o-', color="C1", ms=3, lw=1.0,
            label="NLvib shooting -- 1st harmonic")
    ax.plot(df["omega"], df["A_peak"], 'o-', color="C3", ms=3, lw=1.0,
            label="NLvib shooting -- peak |q(t)|")
ax.plot(omegas, A1,    '-',  color="C0", lw=1.6, label="pyhbm DLFT -- 1st harmonic")
ax.plot(omegas, Apeak, '--', color="C2", lw=1.4, label="pyhbm DLFT -- peak |q(t)|")
ax.axhline(GAP, color="k", ls=":", lw=0.8, label=f"gap $g_0$={GAP}")
ax.set_xlabel(r"$\omega$")
ax.set_ylabel("amplitude")
ax.set_title(f"SDOF vibro-impact -- DLFT (ε={EPSILON:.0e}) vs NLvib shooting")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.4)
plt.tight_layout()

if SAVE_PNG:
    out = Path(__file__).parent / "comparison_dlft_vs_nlvib.png"
    fig.savefig(out, dpi=130)
    print(f"\nFigure saved: {out}")
plt.show()
