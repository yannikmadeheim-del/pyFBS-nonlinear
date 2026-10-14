"""
Two bar+beam elements with normal contact and dry friction (sketch: system.png).

Each cantilever bar+beam element is one substructure. They are coupled at their tips
by rigid unilateral normal contact (DLFT) and regularized Coulomb friction (AFT),
solved together with DLFTContactAFT. The static clamping force P5 keeps the contact
closed over the whole period.

Plot: (a) minimum gap min_t h(t) and (b) tangential sliding max_t |x_T(t)| over the
excitation frequency; (c) normal force and (d) friction force over one period at the
resonance.
"""

import numpy as np
from numpy.fft import irfft
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, Fourier_Real, FourierOmegaPoint, FBSProblem, NumericalFRF, DLFTContactAFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Parameters ------------------------------------------------------------------
PARAMS = ds.BarBeamParams(P1=0.02, P2=0.02, P5=0.05)
EPSILON = 5.0                        # DLFT penalty factor
HARMONICS = list(range(0, 26))       # harmonic 0 required: the contact force has a mean
SAMPLE_NUMBER = 1000
W_LO, W_HI = 0.5, 2.0                # continuation window [rad/s]

# --- System and solver -----------------------------------------------------------
HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # before FBSProblem
system = ds.BarBeamDLFT(PARAMS, sample_number=SAMPLE_NUMBER)
provider = NumericalFRF.from_modal(system.angular_eig_freq, system.eig_vec, system.zeta)
problem = FBSProblem(system, provider,
                     DLFTContactAFT(ds.L_N, ds.L_F, epsilon=EPSILON, g_zero=PARAMS.eps))
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

# initial guess: contact-free response at W_LO, clipped at the gap
zero = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=W_LO)
free = Fourier(problem._get_Fadm(zero).copy())
Fourier_Real.compute_time_series(free)
x_free = free.time_series.copy()
x_free[:, 0, 0] = np.minimum(x_free[:, 0, 0], PARAMS.eps)
ig = FourierOmegaPoint(Fourier_Real.new_from_time_series(x_free), W_LO)

# upward sweep: reference direction omega = +1
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=1.0)

ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=50000,
    angular_frequency_range=[W_LO, W_HI],
    solver_kwargs={"maximum_iterations": 300, "absolute_tolerance": 1e-8},
    step_length_adaptation_kwargs={"base": 2.0, "initial_step_length": 0.01,
                                   "maximum_step_length": 0.05,
                                   "minimum_step_length": 1e-9,
                                   "goal_number_of_iterations": 4},
    jacobian_update_frequency=1, verbose=True)

# --- Time signals ----------------------------------------------------------------
hlist = [int(h) for h in Fourier.harmonics]
Nt = Fourier.number_of_time_samples
omega = np.asarray(ss.omega)


def period(coefficients):
    """One period of a signal from its rFFT coefficients of the retained harmonics."""
    padded = np.zeros(Nt // 2 + 1, dtype=complex)
    padded[hlist] = coefficients
    return irfft(padded, n=Nt)


h_min, slide_max, q1_max = [], [], []
for four, w in zip(ss.fourier, ss.omega):
    x_N = period(four.coefficients[:, 0, 0])               # x_N = q5 - q2
    x_T = period(four.coefficients[:, 1, 0])               # x_T = q1 - q4
    q1 = period(problem.compute_full_response(four, w).coefficients[:, ds.Q1, 0])
    h_min.append(PARAMS.eps - x_N.max())                   # gap function h = eps - x_N
    slide_max.append(np.abs(x_T).max())
    q1_max.append(np.abs(q1).max())

# contact forces at the resonance (peak of the axial tip response q1), as converged by DLFT
i_star = int(np.argmax(q1_max))
point = FourierOmegaPoint(ss.fourier[i_star], ss.omega[i_star])
F_int = problem.method.compute_F_int(point, problem.ode).reshape(len(hlist), ds.N_IF)
f_N, f_T = period(F_int[:, 0]), period(F_int[:, 1])
t = np.arange(Nt) / Nt

# --- Plot ------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(7, 9))
for ax, y, title, ylabel in [(axes[0], h_min, "(a) Gap function", r"$\min_t h(t)$"),
                             (axes[1], slide_max, "(b) Tangential sliding",
                              r"$\max_t |x_T(t)|$")]:
    ax.plot(omega, y, "k")
    ax.plot(omega[i_star], y[i_star], "ko", markersize=4)
    ax.axvline(omega[i_star], color="0.45", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_xlim(W_LO, W_HI)

for ax, y, title, ylabel in [(axes[2], f_N, rf"(c) Normal force at $\omega$ = {omega[i_star]:.4f}",
                              r"$f^N(t)$"),
                             (axes[3], f_T, "(d) Friction force", r"$f^F(t)$")]:
    ax.plot(t, y, "k")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r"$t / T$")
    ax.set_xlim(0.0, 1.0)

for ax in axes:
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
