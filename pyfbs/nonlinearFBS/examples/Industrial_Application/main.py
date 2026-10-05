"""
Nichtlineare FBS-Analyse: Subframe + ControlArm ueber zwei Gummibuchsen.

Anregung am ControlArm, Antwort am Subframe. Ergebnis ist die FRF uy/F0 [mm/N].
Einheiten wie im FE-Modell: mm, N.
"""

import time

import numpy as np

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Parameter -------------------------------------------------------------------
F_LO, F_HI = 10.0, 2000.0    # Kontinuationsfenster [Hz]
HARMONICS = [1]              # Buchsengesetz ist linear -> hoehere Harmonische loesen zu 0
SAMPLE_NUMBER = 128

# Buchsenparameter in SI; ds.bushing_parameters_from_si rechnet ins mm-Modell um
K_TRANS, K_ROT = 1e8, 1e2             # N/m, N*m/rad   -- Platzhalter
Alpha_TRANS, Alpha_ROT = 1e14, 1e2    # N/m^3, N*m/rad^3
C_TRANS, C_ROT = 1.0, 1e-6            # N*s/m, N*m*s/rad
Beta_TRANS, Beta_ROT = 1e6, 1e-6      # N*s^3/m^3, N*m*s^3/rad^3

F0 = 100.0                      # N am Anregungspunkt

OUT_UY = 13                  # Antwort-Sensor Gruppe 10: x,y,z auf 12,13,14

# --- System und Solver -----------------------------------------------------------
bushing = ds.bushing_parameters_from_si(K_TRANS, K_ROT, C_TRANS, C_ROT,
                                        Alpha_TRANS, Alpha_ROT, Beta_TRANS, Beta_ROT)
w_lo, w_hi = 2 * np.pi * F_LO, 2 * np.pi * F_HI

provider = ExperimentalFRF(ds.omega, ds.Y)
system = ds.Subframe_ControlArm(**bushing, F0=F0, sample_number=SAMPLE_NUMBER, omega_ref=w_hi)

HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # VOR FBSProblem
problem = FBSProblem(system, provider, AFT())
solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                               corrector_parameterization=ArcLengthParameterization,
                               predictor=TangentPredictorBordered)

# Abwaerts sweepen: Start bei w_hi, Referenzrichtung mit omega = -1
ig = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=-1.0)

t0 = time.perf_counter()
ss = solver.solve_and_continue(
    initial_guess=ig, initial_reference_direction=rd,
    maximum_number_of_solutions=10000,
    angular_frequency_range=[w_lo, w_hi],
    solver_kwargs={"maximum_iterations": 200, "absolute_tolerance": 1e-8},
    step_length_adaptation_kwargs={"base": 2.0, "initial_step_length": 0.01,
                                   "maximum_step_length": 0.05,
                                   "minimum_step_length": 1e-6,
                                   "goal_number_of_iterations": 3},
    jacobian_update_frequency=1, verbose=True)
print(f"{len(ss)} Punkte in {time.perf_counter() - t0:.1f} s")

# --- Physikalische DoFs zurueckgewinnen ------------------------------------------
# Der Newton-Unbekannte ist nur der Interface-Spalt Q_rel (12 DoF). compute_full_response
# rechnet daraus die Antwort aller N reduzierten DoFs beider Substrukturen zurueck.
Nt = Fourier.number_of_time_samples
i_h1 = [int(h) for h in Fourier.harmonics].index(1)

f_sol = np.asarray(ss.omega) / (2 * np.pi)
uy = np.empty(len(f_sol), dtype=complex)
for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
    full = problem.compute_full_response(four, w)          # (Nh, N, 1)
    uy[i] = 2.0 / Nt * full.coefficients[i_h1, OUT_UY, 0]  # 1. Harmonische, physikalisch

frf = uy / F0                                              # Rezeptanz [mm/N]

# --- Ergebnis --------------------------------------------------------------------
order = np.argsort(f_sol)
np.savetxt(ds.HERE / "frf_uy.csv",
           np.column_stack([f_sol[order], np.abs(frf)[order],
                            np.angle(frf)[order], np.abs(uy)[order]])[:, :4],
           delimiter=",", header="freq_hz,frf_abs_mm_per_N,frf_phase_rad,uy_abs_mm",
           comments="")

peak = order[np.argmax(np.abs(frf)[order])]
print(f"FRF uy/F0: max |H| = {np.abs(frf)[peak]:.3e} mm/N bei {f_sol[peak]:.1f} Hz")
print(f"gespeichert: frf_uy.csv")
