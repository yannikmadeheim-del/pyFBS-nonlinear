"""
Nonlinear FBS: Subframe + ControlArm, coupled by two rubber bushings with a measured
stiffness map.

Both substructures come from Ansys (.full). Each bushing is condensed to one virtual
point with 6 DoF (VPT); its measured complex dynamic stiffness k*_j(f, a_j) acts on
the 12 interface DoFs as a diagonal, amplitude-dependent first-harmonic law:

    F_j = k*_j(omega, a_j) Q_rel,j

interpolated by FBS_SystemComplexStiffnessExperimental and solved with ComplexStiffness.

Excitation on the ControlArm (impact group 20), response at the sensor on the
Subframe (group 10). Result: first-harmonic accelerance for the sensor channels x, y, z.
Units as in the FE model: mm, N, rad.
"""

import numpy as np
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, ComplexStiffness,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Bushing stiffness maps (measured data) -----------------------------------
# DoF order [ux, uy, uz, rx, ry, rz] in the VP frame; both bushings share freq and
# the number of amplitude levels M.
#   freq       (P,)        measurement frequencies [Hz]
#   amplitude  (M, 6)      amplitude levels per DoF [mm or rad]
#   K          (P, M, 6)   complex dynamic stiffness k' + i k'' [N/mm or N*mm/rad]
BUSHING_1 = dict(freq=np.array([]), amplitude=np.empty((0, 6)), K=np.empty((0, 0, 6), dtype=complex))
BUSHING_2 = dict(freq=np.array([]), amplitude=np.empty((0, 6)), K=np.empty((0, 0, 6), dtype=complex))

# --- Parameters ---------------------------------------------------------------
F_LO, F_HI = 5.0, 1000.0             # continuation window [Hz]
HARMONICS = [1]                      # ComplexStiffness is a first-harmonic law
SAMPLE_NUMBER = 37

F0 = 1000.0                          # excitation amplitude [N]
OUT_CH = {"x": 12, "y": 13, "z": 14}     # output sensor channels (Subframe, group 10)

# --- Stiffness table ----------------------------------------------------------
omega_grid, amplitude_grid, K_table = ds.stiffness_map(BUSHING_1, BUSHING_2)
w_lo, w_hi = 2 * np.pi * F_LO, 2 * np.pi * F_HI

# the last continuation point lies just outside the window, so the table needs a margin
if not (omega_grid[0] <= 0.98 * w_lo and 1.02 * w_hi <= omega_grid[-1]):
    raise ValueError(f"the window [{F_LO}, {F_HI}] Hz plus a 2 % margin must lie inside the "
                     f"measured range [{omega_grid[0] / (2 * np.pi):.4g}, "
                     f"{omega_grid[-1] / (2 * np.pi):.4g}] Hz")

# --- System and solver --------------------------------------------------------
provider = ExperimentalFRF(ds.omega, ds.Y)
system = ds.Subframe_ControlArmStiffnessMap(omega_grid, amplitude_grid, K_table,
                                            F0=F0, sample_number=SAMPLE_NUMBER)

HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # before FBSProblem
problem = FBSProblem(system, provider, ComplexStiffness())
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

# downward sweep: start at w_hi, reference direction omega = -1
ig = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=-1.0)

ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=50000,
    angular_frequency_range=[w_lo, w_hi],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-8},
    step_length_adaptation_kwargs={"base": 2.0, "initial_step_length": 0.01,
                                   "maximum_step_length": 1.0,
                                   "minimum_step_length": 1e-6,
                                   "goal_number_of_iterations": 3},
    jacobian_update_frequency=1, verbose=True)

# --- Interface amplitudes -----------------------------------------------------
# amplitudes above the largest tabulated level stop the continuation, so compare them
Nt = Fourier.number_of_time_samples
a_max = np.max([2.0 / Nt * np.abs(four.coefficients[0, :, 0]) for four in ss.fourier], axis=0)
print("\nDoF   max amplitude   largest tabulated level   [mm or rad]")
for j in range(ds.N_IF):
    print(f"{j:3d}   {a_max[j]:13.4g}   {amplitude_grid[-1, j]:23.4g}")

# --- Output sensor ------------------------------------------------------------
# compute_full_response recovers all channels from the interface gap (Newton unknown)
omega = np.asarray(ss.omega)
u1 = np.array([problem.compute_full_response(four, w).coefficients[0, list(OUT_CH.values()), 0]
               for four, w in zip(ss.fourier, ss.omega)]) * 2.0 / Nt     # 1st harmonic [mm]
accelerance = omega[:, None] ** 2 * u1 * 1e-3 / F0                         # [m/s^2/N]

# --- Plot ---------------------------------------------------------------------
# continuation order, not sorted: a nonlinear response can fold over
fig, ax = plt.subplots(figsize=(8, 5))
for j, name in enumerate(OUT_CH):
    ax.semilogy(omega / (2 * np.pi), np.abs(accelerance[:, j]), label=name)
ax.set_title("Accelerance: excitation ControlArm -> sensor Subframe (bushing stiffness map)")
ax.set_xlabel("f [Hz]")
ax.set_ylabel(r"$|\omega^2 Q_{1}\, /\, F_0|$  [m s$^{-2}$ N$^{-1}$]")
ax.set_xlim(F_LO, F_HI)
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
