"""
Cubic-spring FBS coupling of the pyFBS lab testbench through the nonlinearFBS solver.

Same testbench and same FBS structure as the linear-spring example, but substructures
A and B are now coupled by a CUBIC hardening spring on the six virtual-point interface
DoFs:  f_nl = k x_r + alpha x_r^3,  x_r = B u.  Because the force is nonlinear
there is no closed-form LM-FBS assembly: the forced response is traced by AFT +
arc-length HBM continuation directly in physical rad/s.

The SAME cubic coupling is solved twice, through both FRF providers, to verify the
pipeline against itself:
  - ExperimentalFRF : measured-style admittance grid (prior VPT) + spline interpolation,
  - ModalVPFRF      : VPT folded into the FE mode shapes, synthesized at the exact n*omega.
Their forced-response curves must overlay. The linear (alpha=0) LM-FBS receptance is
drawn faintly as a backbone so the hardening bend is visible.
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

from dynamical_system import build_testbench_data, TestbenchCubicSpring, N_IF

# ---------------------------------------------------------------------------
# Coupling parameters -- tune these.
#   k_trans / k_rot         : linear stiffness [N/m] / [Nm/rad]
#   alpha_trans / alpha_rot : cubic stiffness coeff [N/m^3] / [Nm/rad^3] in f = k*x + alpha*x^3
#   beta_trans / beta_rot   : cubic damping coeff [N s^3/m^3] / [Nm s^3/rad^3] adding
#                             a velocity term beta * xdot^3 (0 -> no nonlinear damping;
#                             raise it to clip the hardening peak amplitude)
#   F0                      : harmonic force amplitude [N] (drives the gap, hence the
#                             cubic strength -- raise F0 and/or alpha to bend harder)
# ---------------------------------------------------------------------------
k_trans, k_rot         = 1.0e3, 1.0e3
alpha_trans, alpha_rot = 1.0e4, 1.0e04
beta_trans, beta_rot   = 1.0e0, 1.0e0
F0                     = 50.0
F_RESOLUTION           = 0.1           # FRF resolution Delta f [Hz] (see linear example)
HARMONICS              = [1, 3, 5, 7]     # cubic forcing generates odd harmonics

# --- full-mesh response animation (Route B) ---------------------------------
# Animate the full nonlinear periodic motion (all harmonics) of the coupled A+B
# meshes at one excitation frequency, exactly like the mode-shape display.
ANIMATE_RESPONSE = True      # set False to skip the 3D animation
TARGET_FREQ_HZ   = 1000      # [Hz] visualize the converged point nearest this;
                             # None -> use the peak-amplitude point of the branch
R_SCALE          = 0.08      # max animated displacement as a fraction of the model diagonal

# ---------------------------------------------------------------------------
# 1) Data: ANSYS lab testbench -> block-diagonal admittance Y = diag(Y_A, Y_B).
# ---------------------------------------------------------------------------
data = build_testbench_data(k_trans, k_rot, alpha_trans, alpha_rot,
                            beta_trans, beta_rot, f_resolution=F_RESOLUTION)
freq, omega, Y = data["freq"], data["omega"], data["Y"]
nA, nB, N      = data["nA"], data["nB"], data["N"]
Bc, k_diag     = data["Bc"], data["k_diag"]

# ---------------------------------------------------------------------------
# 2) Linear (alpha=0) LM-FBS spring receptance -- backbone for the window + plot.
#    Pure spring, so Gamma = K_spring^-1 (real, diagonal).
# ---------------------------------------------------------------------------
Gamma = np.zeros((len(freq), N_IF, N_IF))
d = np.arange(N_IF)
Gamma[:, d, d] = 1.0 / k_diag

BY   = Bc @ Y
YBt  = Y @ Bc.T
Yint = Bc @ YBt
Y_spring = Y - YBt @ pyfbs.tpinv(Yint + Gamma, trunc=0) @ BY

ref = np.r_[np.arange(0, nA - N_IF), np.arange(nA + N_IF, N)]
Y_spring_ref = Y_spring[:, ref][:, :, ref]
out, inp = 0, nA - N_IF
mag_lin  = np.abs(Y_spring_ref[:, out, inp])           # linear cross-FRF magnitude [m/N]

# ---------------------------------------------------------------------------
# 3) Continuation window: centre on the linear resonance, extend upward so the
#    hardening peak (which shifts to higher frequency) stays inside the band.
# ---------------------------------------------------------------------------
             # skip rigid-body / DC

f_lo, f_hi = 20, 1000
w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
width = w_hi - w_lo
f_ref = float(freq[np.argmax(np.where(freq>20, mag_lin, 0.0))])
print(f"Continuation window: {f_lo:.1f}..{f_hi:.1f} Hz around f_ref = {f_ref:.1f} Hz")

system = TestbenchCubicSpring(data, F0=F0)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)


def run_nfrc(provider, label):
    """AFT + arc-length HBM continuation for one FRF provider; returns (f_Hz, amplitude)
    along the converged branch. Identical solver settings for every provider, so any
    difference in the curve is the FRF pipeline alone. Amplitude = peak |u_out(t)| over
    the period (the nonlinear forced response at this F0)."""
    problem = FBSProblem(system, provider, AFT())      # asserts B cols == provider.n_dofs (= N)
    solver  = HarmonicBalanceMethod(
        harmonics=HARMONICS, freq_domain_ode=problem,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)

    # cold zero start at the top of the window; reference direction sweeps omega downward.
    ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=w_hi)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex), omega=-1.0)

    print(f"\n[{label}] continuation:")
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = 10000,
        angular_frequency_range       = [w_lo, w_hi],
        solver_kwargs                 = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
        step_length_adaptation_kwargs = {"base": 4.0,
                                         "initial_step_length": 0.1*2*np.pi,
                                         "maximum_step_length": 5*2*np.pi,
                                         "minimum_step_length": 1e-7,
                                         "goal_number_of_iterations": 3},
        jacobian_update_frequency     = 1,
    )

    f_solver = np.array(ss.omega) / (2.0 * np.pi)       # rad/s -> Hz
    amp      = np.zeros(len(f_solver))
    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        full = problem.compute_full_response(four, w)
        Fourier_Real.compute_time_series(full)
        amp[i] = float(np.max(np.abs(full.time_series[:, system.out_full, 0])))
    return f_solver, amp, problem, ss


# (a) ExperimentalFRF: prior VPT pre-applied to the whole grid, then splined.
f_exp, amp_exp, _, _ = run_nfrc(ExperimentalFRF(omega, Y), "ExperimentalFRF + prior VPT")

# (b) ModalVPFRF: VPT folded into the FE mode shapes, synthesized at the EXACT n*omega.
provider_num = ModalVPFRF.from_substructures(
    [(data["MK_A"], data["vpt_A"], data["df_chn_A"], data["df_imp_A"]),
     (data["MK_B"], data["vpt_B"], data["df_chn_B"], data["df_imp_B"])],
    modal_damping=data["modal_damping"])
f_num, amp_num, problem_num, ss_num = run_nfrc(provider_num, "ModalVPFRF (numerical fold)")

# ---------------------------------------------------------------------------
# 4) Figure: both solver curves overlaid, with the linear receptance (* F0) backbone.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(freq, mag_lin * F0, '-', color="#999999", lw=1.2,
            label=f"linear spring (alpha=0) x F0={F0:g} N")
ax.semilogy(f_exp, amp_exp, 'o', color="#E8820C", ms=4.0, mfc="none",
            label="ExperimentalFRF + prior VPT (solver)")
ax.semilogy(f_num, amp_num, '-', color="#2ca02c", ms=5.0,
            label="ModalVPFRF numerical fold (solver)")
ax.set_xlim(f_lo, f_hi)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude |q|  [m]")
ax.set_title("Cubic-spring FBS coupling of testbench A + B (experimental vs numerical VPT)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 5) Full-mesh response animation (Route B): the full nonlinear periodic motion
#    of the coupled A+B meshes at one excitation frequency, exactly like the
#    mode-shape display but reconstructed from all retained harmonics.
# ---------------------------------------------------------------------------
if ANIMATE_RESPONSE:
    import pyfbs
    from pyfbs.nonlinearFBS.examples.response_visualization import animate_response_at_frequency

    # default target: the peak-amplitude point of the numerical branch
    target = TARGET_FREQ_HZ if TARGET_FREQ_HZ is not None else float(f_num[np.argmax(amp_num)])
    substructures = [
        {"model": data["MK_A"], "vpt": data["vpt_A"], "df_imp": data["df_imp_A"], "n_reduced": nA},
        {"model": data["MK_B"], "vpt": data["vpt_B"], "df_imp": data["df_imp_B"], "n_reduced": nB},
    ]
    # show_origin=False: View3D's origin CSYS is sized for millimetres (arrow
    # size 10), but the FE mesh is in metres (~0.15 m), so the default triad
    # dwarfs the model and the camera frames the arrows instead of the response.
    view = pyfbs.display.View3D(title="Cubic-spring coupling -- response animation",
                                show_origin=False)
    animate_response_at_frequency(
        view, problem_num, ss_num, substructures,
        target_frequency=target, modal_damping=data["modal_damping"],
        r_scale=R_SCALE, run_animation=True)

plt.show()

# keep the 3D window open after the matplotlib figures are closed
if ANIMATE_RESPONSE:
    view.plot.app.exec_()
