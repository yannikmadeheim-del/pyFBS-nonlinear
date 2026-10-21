"""
Nonlinear FBS of the lab testbench A + B with DLFTContactAFT.

Excitation at the first B reference DoF. The result is |H1| of the interface gap
q_rel, rebuilt from the recovered physical DoFs.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ModalVPFRF, DLFTContactAFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)
# from pyfbs.nonlinearFBS import ExperimentalFRF

# --- Parameters ---------------------------------------------------------------
F_LO, F_HI = 1.0, 500.0      # continuation window [Hz]
HARMONICS = [1, 3, 5, 7]
SAMPLE_NUMBER = 256

K, ALPHA = 1.0e6, 1.0e8      # N/m, N/m^3
C, BETA = 0.5, 0.0           # N*s/m, N*s^3/m^3
F0 = 200.0                   # N
EPSILON = 1.0                # DLFT penalty, no effect without localised DoFs

# --- FRF provider -------------------------------------------------------------
provider = ModalVPFRF.from_substructures(
    [(ds.MK_A, ds.vpt_A, ds.df_chn_A, ds.df_imp_A),
     (ds.MK_B, ds.vpt_B, ds.df_chn_B, ds.df_imp_B)],
    modal_damping=ds.modal_damping)
# provider = ExperimentalFRF(ds.omega, ds.Y)

# --- Problem and solver -------------------------------------------------------
HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # before FBSProblem
system = ds.TestbenchJoint(K, ALPHA, C, BETA, F0=F0, sample_number=SAMPLE_NUMBER)
# Residual DLFTContactAFT: R_n = Q_n^rel + B Y(n omega) B^T ( (L^N)^T F_n^N + (L^F)^T F_n^F + F_n^nl,A ) - B Y(n omega) F_n^ext = 0
problem = FBSProblem(system, provider, DLFTContactAFT(ds.L_N, ds.L_F, epsilon=EPSILON))
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

# downward sweep: zero start at w_hi, reference direction omega = -1
w_lo, w_hi = 2 * np.pi * F_LO, 2 * np.pi * F_HI
ig = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=-1.0)

t0 = time.perf_counter()
ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=50000,
    angular_frequency_range=[w_lo, w_hi],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-8},
    step_length_adaptation_kwargs={"base": 2.0, "initial_step_length": 0.01,
                                   "maximum_step_length": 5.0,
                                   "minimum_step_length": 1e-6,
                                   "goal_number_of_iterations": 3},
    jacobian_update_frequency=1, verbose=True)
print(f"{len(ss)} points in {time.perf_counter() - t0:.1f} s")

# --- Physical DoFs ------------------------------------------------------------
Nt = Fourier.number_of_time_samples
i_h1 = [int(h) for h in Fourier.harmonics].index(1)

f_sol = np.asarray(ss.omega) / (2 * np.pi)
q_rel = np.empty((len(f_sol), ds.N_IF), dtype=complex)
for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
    q_full = problem.compute_full_response(four, w).coefficients    # (Nh, N, 1)
    q_rel[i] = 2.0 / Nt * (ds.Bc @ q_full[i_h1])[:, 0]              # physical H1

# --- Plot ---------------------------------------------------------------------
fig, (ax_t, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for j, name in enumerate(["ux", "uy", "uz"]):
    ax_t.semilogy(f_sol, np.abs(q_rel[:, j]), label=name)
for j, name in enumerate(["rx", "ry", "rz"], start=3):
    ax_r.semilogy(f_sol, np.abs(q_rel[:, j]), label=name)
ax_t.set_ylabel(r"$|Q^{\mathrm{rel}}_{1,j}|$ [m]")
ax_r.set_ylabel(r"$|Q^{\mathrm{rel}}_{1,j}|$ [rad]")
ax_r.set_xlabel("f [Hz]")
ax_r.set_xlim(F_LO, F_HI)
for ax in (ax_t, ax_r):
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
fig.tight_layout()
plt.show()
