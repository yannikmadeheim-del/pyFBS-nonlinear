"""
Two rods under unilateral contact (sketch: system.png).

Two identical clamped-free FE rods face each other across an initial gap g0. Rod A is
driven harmonically at its tip, rod B is passive and only moves through contact. The
rigid unilateral contact on the relative approach x_r = u_A + u_B is solved with the
DLFT part of DLFTContactAFT (no friction DoFs).

Plot: forced response max_t |x_r(t)| together with the linear response without contact.
"""

import numpy as np
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier_Real, FourierOmegaPoint, FBSProblem, NumericalFRF, DLFTContactAFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Parameters ------------------------------------------------------------------
PARAMS = ds.RodParams(n_elem=20, F0=25e3, sample_number=1000)
GAP = 0.2e-3                         # initial gap g0 [m]
EPSILON_REL = 15.0                   # DLFT penalty factor eps / k_rod
HARMONICS = list(range(0, 31))
OMEGA_LOW_REL, OMEGA_HIGH_REL = 0.8, 1.2     # window as fraction of the first mode omega_1

# --- System and solver -----------------------------------------------------------
system = ds.TwoRodVibroImpact(PARAMS)
provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)

HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)   # before FBSProblem
problem = FBSProblem(system, provider,
                     DLFTContactAFT(ds.L_N, ds.L_F, epsilon=EPSILON_REL * system.k_rod,
                                    g_zero=GAP))
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

w_lo, w_hi = OMEGA_LOW_REL * system.omega_ref, OMEGA_HIGH_REL * system.omega_ref


def linear_relative(omega):
    """Complex amplitude of x_r without contact, from the modal data."""
    Om, Phi, z = system.angular_eig_freq, system.eig_vec, system.zeta
    Y = Phi @ np.diag(1.0 / (Om ** 2 - omega ** 2 + 2j * z * Om * omega)) @ Phi.T
    return (system.B_coupling @ Y[:, system.tipA_idx])[0] * system.F0


# downward sweep: start from the linear response at w_hi, reference direction omega = -1
ig = FourierOmegaPoint.new_from_first_harmonic(np.array([[linear_relative(w_hi)]]), omega=w_lo)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((1, 1), complex), omega=1.0)

ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=10000,
    angular_frequency_range=[w_lo, w_hi],
    solver_kwargs={"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs={"base": 3.0, "initial_step_length": 0.001,
                                   "maximum_step_length": 0.001,
                                   "minimum_step_length": 1e-10,
                                   "goal_number_of_iterations": 3},
    jacobian_update_frequency=1, verbose=True)

# --- Relative approach -----------------------------------------------------------
# x_r is the Newton unknown itself
xr_max = []
for four in ss.fourier:
    Fourier_Real.compute_time_series(four)
    xr_max.append(np.abs(four.time_series[:, 0, 0]).max())

omega_lin = np.linspace(w_lo, w_hi, 600)
xr_lin = np.abs([linear_relative(w) for w in omega_lin])

# --- Plot ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(omega_lin, xr_lin, "-.", color="0.45", label="linear, no contact")
ax.plot(ss.omega, xr_max, "k", label="DLFT")
ax.set_xlim(w_lo, w_hi)
ax.set_ylim(0.0, 1.1 * max(xr_lin.max(), max(xr_max)))
ax.axhspan(GAP, ax.get_ylim()[1], color="red", alpha=0.12, label=r"contact regime $x_r \geq g_0$")
ax.set_title("Forced response")
ax.set_xlabel(r"$\omega$ [rad/s]")
ax.set_ylabel(r"$\max_t |x_r(t)|$ [m]")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
