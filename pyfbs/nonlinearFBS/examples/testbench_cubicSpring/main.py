"""
Cubic-spring FBS coupling of the pyFBS lab testbench through the nonlinearFBS solver.

Same testbench and same FBS structure as the linear-spring example, but substructures
A and B are now coupled by a CUBIC hardening spring on the six virtual-point interface
DoFs:  f_nl = k x_r + alpha x_r^3,  x_r = B u.  Because the force is nonlinear
there is no closed-form LM-FBS assembly: the forced response is traced by AFT +
arc-length HBM continuation directly in physical rad/s.

The admittance enters through the FRF provider selected by FRF_SOURCE:
  - "modal"        : ModalVPFRF -- VPT folded into the FE mode shapes, synthesized at the exact n*omega,
  - "experimental" : ExperimentalFRF -- measured-style admittance grid (prior VPT) + spline interpolation.
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
alpha_trans, alpha_rot = 1.0e8, 1.0e6
beta_trans, beta_rot   = 0, 0
F0                     = 50.0
F_RESOLUTION           = 0.1           # FRF resolution Delta f [Hz] (see linear example)
HARMONICS              = [1, 3, 5, 7]     # cubic forcing generates odd harmonics

FRF_SOURCE = "modal"    # "modal":        ModalVPFRF -- VPT folded into the FE modes, exact at n*omega
                        # "experimental": ExperimentalFRF -- presampled admittance grid + spline

# --- full-mesh response animation -------------------------------------------
# Animate the full nonlinear periodic motion (all harmonics) of the coupled A+B
# meshes at one excitation frequency, exactly like the mode-shape display.
ANIMATE_RESPONSE = True      # set False to skip the 3D animation
ANIMATE_FREQ_HZ  = None      # [Hz] animate the converged point nearest this frequency;
                             # None -> the peak-amplitude point of the branch
R_SCALE          = 0.08      # max animated displacement as a fraction of the model diagonal

# ---------------------------------------------------------------------------
# 1) Data: ANSYS lab testbench -> block-diagonal admittance Y = diag(Y_A, Y_B).
# ---------------------------------------------------------------------------
data = build_testbench_data(k_trans, k_rot, alpha_trans, alpha_rot,
                            beta_trans, beta_rot, f_resolution=F_RESOLUTION)
freq, omega, Y = data["freq"], data["omega"], data["Y"]
nA, nB, N      = data["nA"], data["nB"], data["N"]

# ---------------------------------------------------------------------------
# 2) Continuation window: wide enough that the hardening peak (which shifts to
#    higher frequency) stays inside the band.
# ---------------------------------------------------------------------------
f_lo, f_hi = 15, 1000
w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
print(f"Continuation window: {f_lo:.1f}..{f_hi:.1f} Hz")

system = TestbenchCubicSpring(data, F0=F0)
HarmonicBalanceMethod.update_dependencies(HARMONICS, system.sample_number)

# ---------------------------------------------------------------------------
# 3) FRF provider (FRF_SOURCE) and AFT + arc-length HBM continuation.
# ---------------------------------------------------------------------------
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
    maximum_number_of_solutions   = 30000,
    angular_frequency_range       = [w_lo, w_hi],
    solver_kwargs                 = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_length_adaptation_kwargs = {"base": 3.0,
                                     "initial_step_length": 0.01*2*np.pi,
                                     "maximum_step_length": 0.1*2*np.pi,
                                     "minimum_step_length": 1e-7,
                                     "goal_number_of_iterations": 3},
    jacobian_update_frequency     = 1,
)

# per converged point: peak |u_out(t)| over one period (the nonlinear forced
# response at this F0)
f_sol = np.array(ss.omega) / (2.0 * np.pi)          # rad/s -> Hz
amp   = np.zeros(len(f_sol))
for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
    full = problem.compute_full_response(four, w)
    Fourier_Real.compute_time_series(full)
    amp[i] = float(np.max(np.abs(full.time_series[:, system.out_full, 0])))

# ---------------------------------------------------------------------------
# 4) Figure: nonlinear forced-response curve.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogy(f_sol, amp, '-', color="#2ca02c",
            label=f"nonlinearFBS solver ({FRF_SOURCE} FRF)")
ax.set_xlim(f_lo, f_hi)
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude |q|  [m]")
ax.set_title("Cubic-spring FBS coupling of testbench A + B")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 5) Full-mesh response animation: the full nonlinear periodic motion of the
#    coupled A+B meshes at one excitation frequency, reconstructed from all
#    retained harmonics; excitation (red) and output (blue) DoFs are marked.
# ---------------------------------------------------------------------------
if ANIMATE_RESPONSE:
    from pyfbs.nonlinearFBS.examples.response_visualization import animate_response_at_frequency

    target = ANIMATE_FREQ_HZ if ANIMATE_FREQ_HZ is not None else float(f_sol[np.argmax(amp)])
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
        view, problem, ss, substructures,
        target_frequency=target, modal_damping=data["modal_damping"],
        output_dof=system.out_full,
        r_scale=R_SCALE, run_animation=True)

plt.show()

# keep the 3D window open after the matplotlib figures are closed
if ANIMATE_RESPONSE:
    view.plot.app.exec_()
