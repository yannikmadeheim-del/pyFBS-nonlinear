"""
Linear-spring (bushing) FBS coupling of the pyFBS lab testbench -- LM-FBS reference
PLUS a verification against the ported nonlinearFBS solver.

Substructures A and B are coupled through a linear spring acting on all six
virtual-point interface DoFs (3 translations + 3 rotations). The reference uses the
LM-FBS "bushing as compatibility relaxation" formulation -- compatibility  B u = 0
is relaxed to  B u = Gamma * lambda, giving

    Y_assembled = Y - Y B^T (B Y B^T + Gamma)^-1 B Y

The SAME spring is then applied through the nonlinearFBS solver as a linear interface
force  F_nl = K_spring x_r  (Gamma = K_spring^-1), solved with AFT + arc-length HBM
continuation in physical rad/s. Overlaying the two frequency responses verifies the
new solver on this known linear case (the milestone before the nonlinear couplings).
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
    Fourier_Real, FourierOmegaPoint, FBSProblem, ExperimentalFRF, AFT,
    HarmonicBalanceMethod,
)
from pyfbs.nonlinearFBS.numerical_continuation.corrector_step import ArcLengthParameterization
from pyfbs.nonlinearFBS.numerical_continuation.predictor_step import TangentPredictorBordered

from dynamical_system import build_testbench_data, TestbenchLinearSpring, N_IF

# ---------------------------------------------------------------------------
# Linear spring parameters (one spring per interface DoF) -- tune these.
#   k_trans : translational stiffness [N/m]    for ux, uy, uz
#   k_rot   : rotational    stiffness [Nm/rad]  for rx, ry, rz
#   c_trans : translational viscous damping [Ns/m]    (0.0 = pure spring)
#   c_rot   : rotational    viscous damping [Nms/rad]
# ---------------------------------------------------------------------------
k_trans, k_rot = 1.0e6, 1.0e3
c_trans, c_rot = 0.0, 0.0

# FRF frequency resolution Delta f [Hz]: SMALLER = finer (density = 1/F_RESOLUTION
# samples per Hz). The pyFBS default 1 Hz renders the lightly-damped resonances
# (Q ~ 167 -> ~1.4 Hz wide) as 1-point spikes; 0.1 Hz puts ~14 samples across each.
# Grid size ~ f_end / F_RESOLUTION points (Y is npts x N x N complex):  0.1 -> 20k
# (~0.3 GB, ~5 s);  0.01 -> 200k (~3 GB);  0.001 -> 2M (~29 GB, will OOM). Use ~0.1.
F_RESOLUTION = 0.01

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
Y_rigid  = Y - YBt @ pyfbs.tpinv(Yint,         trunc=0) @ BY

# ---------------------------------------------------------------------------
# 4) Reference figure: drop the 12 interface DoFs, compare rigid vs spring.
# ---------------------------------------------------------------------------
ref = np.r_[np.arange(0, nA - n_if), np.arange(nA + n_if, N)]  # A ref + B ref
Y_spring_ref = Y_spring[:, ref][:, :, ref]
Y_rigid_ref  = Y_rigid[:,  ref][:, :, ref]

out = 0             # response DoF on substructure A
inp = nA - n_if     # force DoF on substructure B (first B reference DoF) -> cross-interface FRF

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
for label, Yc in (("rigid coupling", Y_rigid_ref), ("spring coupling", Y_spring_ref)):
    ax_mag.semilogy(freq, np.abs(Yc[:, out, inp]), label=label)
    ax_ph.plot(freq, np.angle(Yc[:, out, inp]), label=label)
ax_mag.set_ylabel("Receptance |Y|  [m/N]")
ax_mag.set_title("Linear-spring vs rigid FBS coupling of testbench A + B")
ax_mag.grid(True, which="both", alpha=0.3)
ax_mag.legend()
ax_ph.set_ylabel("Phase [rad]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.grid(True, alpha=0.3)
fig.tight_layout()

# ===========================================================================
# 5) VERIFICATION: same coupling through the nonlinearFBS solver.
#    F_nl = K_spring x_r reproduces Y_spring exactly, so the solver's forced
#    response must overlay the LM-FBS spring receptance.
# ===========================================================================
mag_ref = np.abs(Y_spring_ref[:, out, inp])             # reference cross-FRF magnitude

# centre the continuation window on the dominant coupled resonance; the sweep runs
# directly in physical rad/s (no omega_ref / nondimensionalization).
SPAN = 0.06                                             # window half-width (+/- 6%)
band  = freq > 50.0                                     # skip rigid-body / DC
f_ref = float(freq[np.argmax(np.where(band, mag_ref, 0.0))])
f_lo, f_hi = (1.0 - SPAN) * f_ref, (1.0 + SPAN) * f_ref
w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
print(f"Verification window: {f_lo:.1f}..{f_hi:.1f} Hz around f_ref = {f_ref:.1f} Hz")

system   = TestbenchLinearSpring(data, F0=1.0)          # F0=1 -> response curve IS the receptance
provider = ExperimentalFRF(omega, Y)                   # physical rad/s grid, physical Y

HARMONICS = [1]                                        # linear, cos forcing -> only 1st harmonic
                                                       # (incl. harmonic 0 makes the AFT DC-Jacobian
                                                       #  inexact -> Newton stalls near resonance)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)
problem = FBSProblem(system, provider, AFT())          # asserts B cols == provider.n_dofs (= N)
solver  = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=problem,
    corrector_parameterization=ArcLengthParameterization,
    predictor=TangentPredictorBordered)

# cold zero start at the top of the window; reference direction sweeps omega downward.
# step lengths are window-relative (rad/s) so sampling is independent of f_ref.
width = w_hi - w_lo
ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex), omega=-1.0)

ss = solver.solve_and_continue(
    initial_guess                 = ig,
    initial_reference_direction   = rd,
    maximum_number_of_solutions   = 2000,
    angular_frequency_range       = [w_lo, w_hi],
    solver_kwargs                 = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs = {"base": 4.0,
                                     "initial_step_length": width / 80.0,
                                     "maximum_step_length": width / 40.0,
                                     "minimum_step_length": width * 1e-6,
                                     "goal_number_of_iterations": 3},
    jacobian_update_frequency     = 1,
)

# per converged point: full physical response -> peak |u_out(t)| = |Y_coupled[out, inp]| (F0=1)
f_solver = np.array(ss.omega) / (2.0 * np.pi)           # rad/s -> Hz
recept   = np.zeros(len(f_solver))
for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
    full = problem.compute_full_response(four, w)
    Fourier_Real.compute_time_series(full)
    recept[i] = float(np.max(np.abs(full.time_series[:, system.out_full, 0])))

# ---------------------------------------------------------------------------
# 6) Verification figure: solver NFRC overlaid on the LM-FBS spring receptance.
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.semilogy(freq, mag_ref, '-', color="#1f77b4", lw=1.6,
             label="LM-FBS spring coupling (reference)")
ax2.semilogy(f_solver, recept, 'o', color="#E8820C", ms=4.0,
             label="nonlinearFBS solver (AFT + HBM)")
ax2.set_xlim(f_lo, f_hi)
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Amplitude |q|  [m/N]")
ax2.set_title("Verification: nonlinearFBS vs LM-FBS linear-spring coupling")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()
fig2.tight_layout()

plt.show()
