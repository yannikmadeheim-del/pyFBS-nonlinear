"""
Extension A on the lab testbench: ComplexStiffness against AFT with the same Duffing law;
first-harmonic equivalence, plus the error of the first-harmonic approximation against AFT with more harmonics.
The same law as a table (FBS_SystemComplexStiffnessExperimental) and sampled FRFs (ExperimentalFRF)
check the tabulated stiffness and the FRF-provider independence.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ModalVPFRF, ExperimentalFRF, AFT, ComplexStiffness,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Parameters ---------------------------------------------------------------
F_LO, F_HI = 1.0, 300.0      # continuation window [Hz]
SAMPLE_NUMBER = 256

k = 1.0e6                    # N/m
alpha = 1.0e8                # N/m^3
c = 0.5                      # N*s/m
F0 = 200.0                   # N
A_MAX = 1.0                  # largest tabulated amplitude [m or rad]; the splines reproduce
                             # the quadratic law exactly, so it only has to enclose the response

# --- Problem and solver -------------------------------------------------------
def run(system, method, harmonics, provider):
    """Downward continuation over [F_LO, F_HI] for one system, method, harmonic set and
    FRF provider; returns f [Hz] and the physical first harmonic at the output channel."""
    HarmonicBalanceMethod.update_dependencies(harmonics, SAMPLE_NUMBER)   # before FBSProblem
    problem = FBSProblem(system, provider, method)
    solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=problem,
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
                                       "maximum_step_length": 1.0,
                                       "minimum_step_length": 1e-6,
                                       "goal_number_of_iterations": 3},
        jacobian_update_frequency=1, verbose=True)
    print(f"{len(ss)} points in {time.perf_counter() - t0:.1f} s")

    # --- Physical DoFs ------------------------------------------------------------
    Nt = Fourier.number_of_time_samples
    i_h1 = [int(h) for h in Fourier.harmonics].index(1)

    f_sol = np.asarray(ss.omega) / (2 * np.pi)
    q_out = np.empty(len(f_sol), dtype=complex)
    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        q_full = problem.compute_full_response(four, w).coefficients    # (Nh, N, 1)
        q_out[i] = 2.0 / Nt * q_full[i_h1, system.out_full, 0]          # physical H1 at the output channel

    return f_sol, q_out

# --- FRF provider -------------------------------------------------------------
provider_num = ModalVPFRF.from_substructures(
    [(ds.MK_A, ds.vpt_A, ds.df_chn_A, ds.df_imp_A),
     (ds.MK_B, ds.vpt_B, ds.df_chn_B, ds.df_imp_B)],
    modal_damping=ds.modal_damping)
provider_exp = ExperimentalFRF(ds.omega, ds.Y)

# --- Runs ---------------------------------------------------------------------
system_cs = ds.TestbenchJointDuffingComplexStiffness(k, alpha, c, F0=F0, sample_number=SAMPLE_NUMBER)
system_aft = ds.TestbenchJointDuffingAFT(k, alpha, c, F0=F0, sample_number=SAMPLE_NUMBER)
omega_grid = np.linspace(0.0, 2 * np.pi * 1.02 * F_HI, 400)    # margin beyond the window
amplitude_grid = np.linspace(0.0, A_MAX, 30)[:, None] * np.ones(ds.N_IF)
K_table = ds.duffing_table(k, alpha, c, omega_grid, amplitude_grid)
system_table = ds.TestbenchJointDuffingTable(omega_grid, amplitude_grid, K_table, F0=F0, sample_number=SAMPLE_NUMBER)

f_cs,  q_cs  = run(system_cs,  ComplexStiffness(), [1], provider_num)
f_aft, q_aft = run(system_aft, AFT(),              [1], provider_num)
f_h7,  q_h7  = run(system_aft, AFT(),              [1, 3, 5, 7], provider_num)
f_cs_exp, q_cs_exp = run(system_cs,  ComplexStiffness(), [1], provider_exp)
f_tab,    q_tab    = run(system_table, ComplexStiffness(), [1], provider_num)

# --- Plot ---------------------------------------------------------------------
# First-harmonic magnitude at the output channel (first A reference DoF) for the
# excitation F0 cos(omega t) at the input channel (first B reference DoF).
# ComplexStiffness and AFT [1] must coincide; AFT [1, 3, 5, 7] shows the error of
# the first-harmonic approximation; ComplexStiffness with the sampled (experimental)
# FRFs shows the provider independence. Line styles separate the curves where they overlap.
# The tabulated law is drawn as markers: they must sit on the violet ComplexStiffness line.
curves = [
    (f_cs,     q_cs,     "ComplexStiffness, H = [1], ModalVPFRF",      dict(color="#6250d6", linestyle="-",  linewidth=3)),
    (f_aft,    q_aft,    "AFT, H = [1], ModalVPFRF",                   dict(color="#eb6834", linestyle="--", linewidth=2)),
    (f_h7,     q_h7,     "AFT, H = [1, 3, 5, 7], ModalVPFRF",          dict(color="#1baf7a", linestyle=":",  linewidth=2)),
    (f_cs_exp, q_cs_exp, "ComplexStiffness, H = [1], ExperimentalFRF", dict(color="#0b0b0b", linestyle="-.", linewidth=1.5)),
    (f_tab,    q_tab,    "ComplexStiffness, H = [1], ModalVPFRF, table", dict(color="#6250d6", linestyle="none",
                                                                             marker="o", fillstyle="none",
                                                                             markersize=6, markevery=0.03)),
]

fig, ax = plt.subplots(figsize=(8, 4.5))
for f, q, label, style in curves:
    ax.semilogy(f, np.abs(q), label=label, **style)
ax.set_xlabel("f [Hz]")
ax.set_ylabel(r"$|Q_{\mathrm{out},1}|$ [m]")
ax.set_xlim(F_LO, F_HI)
ax.grid(True, which="both", color="0.9", linewidth=0.6)
ax.legend(frameon=False)
fig.tight_layout()
plt.show()
