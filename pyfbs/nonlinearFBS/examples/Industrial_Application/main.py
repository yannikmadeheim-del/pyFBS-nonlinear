"""
Nonlinear FBS: Subframe + ControlArm, coupled by two rubber bushings.

Both substructures come from Ansys (.full). Each bushing is condensed to one virtual
point with 6 DoF (VPT), so the bushing law acts on 12 interface DoFs:

    f = k u + c v + alpha u^3 + beta v^3        (diagonal, per DoF)

Excitation on the ControlArm (impact group 20), response at the sensor on the
Subframe (group 10). Result: first-harmonic accelerance for the sensor channels x, y, z.
Units as in the FE model: mm, N.
"""

import numpy as np
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Parameters ------------------------------------------------------------------
F_LO, F_HI = 1.0, 1000.0             # continuation window [Hz]
HARMONICS = [1, 3, 5, 7, 9]
SAMPLE_NUMBER = 37

# bushing parameters in SI; ds.bushing_parameters_from_si converts them to mm
K_TRANS, K_ROT = 2e5, 1e2                # N/m, N*m/rad
C_TRANS, C_ROT = 4e2, 2e-1               # N*s/m, N*m*s/rad
ALPHA_TRANS, ALPHA_ROT = -5e9, -5e2      # N/m^3, N*m/rad^3
BETA_TRANS, BETA_ROT = 1e3, 1e-3         # N*s^3/m^3, N*m*s^3/rad^3

F0 = 1000.0                          # excitation amplitude [N]
OUT_CH = {"x": 12, "y": 13, "z": 14}     # output sensor channels (Subframe, group 10)

# --- System and solver -----------------------------------------------------------
bushing = ds.bushing_parameters_from_si(K_TRANS, K_ROT, C_TRANS, C_ROT,
                                        ALPHA_TRANS, ALPHA_ROT, BETA_TRANS, BETA_ROT)
w_lo, w_hi = 2 * np.pi * F_LO, 2 * np.pi * F_HI

provider = ExperimentalFRF(ds.omega, ds.Y)
system = ds.Subframe_ControlArm(**bushing, F0=F0, sample_number=SAMPLE_NUMBER, omega_ref=w_hi)

HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # before FBSProblem
problem = FBSProblem(system, provider, AFT())
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

# downward sweep: start at w_hi, reference direction omega = -1
ig = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=-1.0)

ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=10000,
    angular_frequency_range=[w_lo, w_hi],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-8},
    step_length_adaptation_kwargs={"base": 2.0, "initial_step_length": 0.01,
                                   "maximum_step_length": 5.0,
                                   "minimum_step_length": 1e-6,
                                   "goal_number_of_iterations": 3},
    jacobian_update_frequency=1, verbose=True)

# --- Output sensor ---------------------------------------------------------------
# compute_full_response recovers all channels from the interface gap (Newton unknown)
Nt = Fourier.number_of_time_samples
i_h1 = [int(h) for h in Fourier.harmonics].index(1)

omega = np.asarray(ss.omega)
u1 = np.array([problem.compute_full_response(four, w).coefficients[i_h1, list(OUT_CH.values()), 0]
               for four, w in zip(ss.fourier, ss.omega)]) * 2.0 / Nt     # 1st harmonic [mm]
accelerance = omega[:, None] ** 2 * u1 * 1e-3 / F0                         # [m/s^2/N]

# --- Plot ------------------------------------------------------------------------
# continuation order, not sorted: a nonlinear response can fold over
fig, ax = plt.subplots(figsize=(8, 5))
for j, name in enumerate(OUT_CH):
    ax.semilogy(omega / (2 * np.pi), np.abs(accelerance[:, j]), label=name)
ax.set_title("Accelerance: excitation ControlArm -> sensor Subframe")
ax.set_xlabel("f [Hz]")
ax.set_ylabel(r"$|\omega^2 Q_{1}\, /\, F_0|$  [m s$^{-2}$ N$^{-1}$]")
ax.set_xlim(F_LO, F_HI)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
