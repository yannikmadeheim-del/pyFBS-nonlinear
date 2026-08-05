"""
Linear-spring (bushing) FBS coupling of the pyFBS lab testbench -- LM-FBS reference
PLUS a verification against the nonlinearFBS solver.

Substructures A and B are coupled through a linear spring acting on all six
virtual-point interface DoFs (3 translations + 3 rotations). The reference uses the
LM-FBS "bushing as compatibility relaxation" formulation -- compatibility  B u = 0
is relaxed to  B u = Gamma * lambda, giving

    Y_assembled = Y - Y B^T (B Y B^T + Gamma)^-1 B Y

The SAME spring is then applied through the nonlinearFBS solver as a linear interface
force  F_nl = K_spring x_r  (Gamma = K_spring^-1), solved with AFT + arc-length HBM
continuation in physical rad/s. Overlaying the two frequency responses verifies the
solver on this known linear case. The admittance enters through the FRF provider
selected by FRF_SOURCE ("modal" = ModalVPFRF, "experimental" = ExperimentalFRF).
"""

import sys

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib.pyplot as plt

import pyfbs
from pyfbs.nonlinearFBS import (
    Fourier_Real, FourierOmegaPoint, FBSProblem, ExperimentalFRF, ModalVPFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

from dynamical_system import build_testbench_data, TestbenchLinearSpring, N_IF

# ---------------------------------------------------------------------------
# Linear spring parameters (one spring per interface DoF) -- tune these.
#   k_trans : translational stiffness [N/m]    for ux, uy, uz
#   k_rot   : rotational    stiffness [Nm/rad]  for rx, ry, rz
#   c_trans : translational viscous damping [Ns/m]    (0.0 = pure spring)
#   c_rot   : rotational    viscous damping [Nms/rad]
# ---------------------------------------------------------------------------
k_trans, k_rot = 1.0e6, 1.0e3
c_trans, c_rot = 1e-2, 1e-2

# FRF frequency resolution Delta f [Hz]: SMALLER = finer (density = 1/F_RESOLUTION
# samples per Hz). The pyFBS default 1 Hz renders the lightly-damped resonances
# (Q ~ 167 -> ~1.4 Hz wide) as 1-point spikes; 0.1 Hz puts ~14 samples across each.
# Grid size ~ f_end / F_RESOLUTION points (Y is npts x N x N complex):  0.1 -> 20k
# (~0.3 GB, ~5 s);  0.01 -> 200k (~3 GB);  0.001 -> 2M (~29 GB, will OOM). Use ~0.1.
F_RESOLUTION = 0.1

FRF_SOURCE = "modal"    # "modal":        ModalVPFRF -- VPT folded into the FE modes, exact at n*omega
                        # "experimental": ExperimentalFRF -- presampled admittance grid + spline

# --- full-mesh response animation -------------------------------------------
# Animate the full periodic motion of the coupled A+B meshes at one excitation
# frequency, exactly like the mode-shape display (here linear, so only harmonic 1).
ANIMATE_RESPONSE = True      # set False to skip the 3D animation
ANIMATE_FREQ_HZ  = None      # [Hz] animate the converged point nearest this frequency
                             # (e.g. 710.1 = a strongly responding resonance);
                             # None -> the peak-amplitude point of the branch
R_SCALE          = 0.08      # max animated displacement as a fraction of the model diagonal

# ---------------------------------------------------------------------------
# 1) Data: ANSYS lab testbench -> block-diagonal admittance Y = diag(Y_A, Y_B).
#    (load once; shared by the LM-FBS reference and the solver verification)
# ---------------------------------------------------------------------------
data = build_testbench_data(k_trans, k_rot, c_trans, c_rot, f_resolution=F_RESOLUTION)
freq, omega, Y = data["freq"], data["omega"], data["Y"]
nA, nB, N      = data["nA"], data["nB"], data["N"]
Bc             = data["Bc"]
k_diag, c_diag = data["k_diag"], data["c_diag"]

# ---------------------------------------------------------------------------
# 2) Spring dynamic flexibility Gamma(omega) on the 6 interface DoFs.
#    Gamma_ii = 1 / (k_i + i*omega*c_i),  VP order [ux, uy, uz, rx, ry, rz].
# ---------------------------------------------------------------------------
n_if = N_IF
Z_spring = k_diag[None, :] + 1j * omega[:, None] * c_diag[None, :]   # (N_freq, 6)
Gamma = np.zeros((len(freq), n_if, n_if), dtype=complex)
d = np.arange(n_if)
Gamma[:, d, d] = 1.0 / Z_spring

# ---------------------------------------------------------------------------
# 3) LM-FBS reference assembly (dual form; interface inverse via pyfbs.tpinv).
# ---------------------------------------------------------------------------
BY   = Bc @ Y        # (N_freq, 6, N)
YBt  = Y @ Bc.T      # (N_freq, N, 6)
Yint = Bc @ YBt      # (N_freq, 6, 6)  == B Y B^T

# trunc=0 -> full pseudo-inverse (tpinv's trunc=None path is broken upstream).
Y_spring = Y - YBt @ pyfbs.tpinv(Yint + Gamma, trunc=0) @ BY

# drop the 12 interface DoFs and read one cross-interface FRF.
ref = np.r_[np.arange(0, nA - n_if), np.arange(nA + n_if, N)]  # A ref + B ref
Y_spring_ref = Y_spring[:, ref][:, :, ref]

out = 0             # response DoF on substructure A
inp = nA - n_if     # force DoF on substructure B (first B reference DoF) -> cross-interface FRF
mag_ref = np.abs(Y_spring_ref[:, out, inp])             # reference cross-FRF magnitude

# ===========================================================================
# 4) VERIFICATION: same coupling through the nonlinearFBS solver.
#    F_nl = K_spring x_r reproduces Y_spring exactly, so the solver's forced
#    response must overlay the LM-FBS spring receptance.
# ===========================================================================
f_lo, f_hi = 0.1, 1999
w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
print(f"Verification window: {f_lo:.1f}..{f_hi:.1f} Hz")

system    = TestbenchLinearSpring(data, F0=1.0)         # F0=1 -> response curve IS the receptance

HARMONICS = [1]                                        # linear, cos forcing -> only 1st harmonic
                                                       # (incl. harmonic 0 makes the AFT DC-Jacobian
                                                       #  inexact -> Newton stalls near resonance)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)

if FRF_SOURCE == "modal":
    provider = ModalVPFRF.from_substructures(
        [(data["MK_A"], data["vpt_A"], data["df_chn_A"], data["df_imp_A"]),
         (data["MK_B"], data["vpt_B"], data["df_chn_B"], data["df_imp_B"])],
        modal_damping=data["modal_damping"])
elif FRF_SOURCE == "experimental":
    provider = ExperimentalFRF(omega, Y)
else:
    raise ValueError(f"FRF_SOURCE must be 'modal' or 'experimental', got {FRF_SOURCE!r}")

problem = FBSProblem(system, provider, AFT())      # asserts B cols == provider.n_dofs (= N)
solver  = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=problem,
    corrector_parameterization=ArcLengthParameterization,
    predictor=TangentPredictorBordered)

# cold zero start at the top of the window; reference direction sweeps omega downward.
ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex), omega=-1.0)

print(f"\n[{FRF_SOURCE} FRF] continuation:")
ss = solver.solve_and_continue(
    initial_guess                 = ig,
    initial_reference_direction   = rd,
    maximum_number_of_solutions   = 10000,
    angular_frequency_range       = [w_lo, w_hi],
    solver_kwargs                 = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs = {"base": 2.0,
                                     "initial_step_length": 1.0,
                                     "maximum_step_length": 5.0*2*np.pi,
                                     "minimum_step_length": 1e-4,
                                     "goal_number_of_iterations": 3},
    jacobian_update_frequency     = 3,
)

# per converged point: full physical response -> peak |u_out(t)| = |Y_coupled[out, inp]| (F0=1)
f_sol  = np.array(ss.omega) / (2.0 * np.pi)         # rad/s -> Hz
recept = np.zeros(len(f_sol))
for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
    full = problem.compute_full_response(four, w)
    Fourier_Real.compute_time_series(full)
    recept[i] = float(np.max(np.abs(full.time_series[:, system.out_full, 0])))

# ---------------------------------------------------------------------------
# 5) Verification figure: solver NFRC overlaid on the LM-FBS spring receptance.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(freq, mag_ref, '-', color="#1f77b4", lw=1.6,
            label="LM-FBS spring coupling (reference)")
ax.semilogy(f_sol, recept, 'o', color="#E8820C", ms=4.0, mfc="none",
            label=f"nonlinearFBS solver ({FRF_SOURCE} FRF)")
ax.set_xlim(f_lo, f_hi)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude |q|  [m/N]")
ax.set_title("Verification: nonlinearFBS vs LM-FBS spring coupling")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 6) Full-mesh response animation: the full periodic motion of the coupled A+B
#    meshes at one excitation frequency (linear coupling -> 1st-harmonic ODS);
#    excitation (red) and output (blue) DoFs are marked.
# ---------------------------------------------------------------------------
if ANIMATE_RESPONSE:
    from pyfbs.nonlinearFBS.examples.response_visualization import animate_response_at_frequency

    target = ANIMATE_FREQ_HZ if ANIMATE_FREQ_HZ is not None else float(f_sol[np.argmax(recept)])
    substructures = [
        {"model": data["MK_A"], "vpt": data["vpt_A"], "df_imp": data["df_imp_A"], "n_reduced": nA},
        {"model": data["MK_B"], "vpt": data["vpt_B"], "df_imp": data["df_imp_B"], "n_reduced": nB},
    ]
    # show_origin=False: View3D's origin CSYS is sized for millimetres (arrow
    # size 10), but the FE mesh is in metres (~0.15 m), so the default triad
    # dwarfs the model and the camera frames the arrows instead of the response.
    view = pyfbs.display.View3D(title="Linear-spring coupling -- response animation",
                                show_origin=False)
    animate_response_at_frequency(
        view, problem, ss, substructures,
        target_frequency=target, modal_damping=data["modal_damping"],
        output_dof=system.out_full,
        r_scale=R_SCALE, run_animation=True)

plt.show()

# keep the 3D window open after the matplotlib figures are closed
if ANIMATE_RESPONSE:
    view.plot.app.exec_()
