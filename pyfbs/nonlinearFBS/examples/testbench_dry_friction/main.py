"""
Dry-friction FBS coupling of the pyFBS lab testbench through the nonlinearFBS solver.

Same testbench and same FBS structure as the spring examples, but substructures A
and B are now coupled by a DRY-FRICTION joint on the virtual-point interface gap
x_r = B u: isotropic regularized Coulomb friction on the x-y gap velocity plus a
penalty spring on the z gap, mirroring the physical joint (B clamps A's plate,
contact normal along z, slip in the x-y plane):

    f_T   = 2 mu N tanh(alpha ||v_T||) v_T/||v_T||,   f_z  = k_trans z_r,
    M_rz  = 2 mu N G tanh(alpha G rzdot),             M_rxy = k_rot theta_r,

with N the bolt clamping force (independent of the excitation amplitude F0)
and G the effective contact radius of the torsional friction. Every DoF's
joint term is decoupled from the others. Because
the force is nonlinear there is no closed-form LM-FBS assembly: the forced
response is traced by AFT + arc-length HBM continuation directly in physical rad/s.

The admittance enters through the FRF provider selected by FRF_SOURCE:
  - "modal"        : ModalVPFRF -- VPT folded into the FE mode shapes, synthesized at the exact n*omega,
  - "experimental" : ExperimentalFRF -- measured-style admittance grid (prior VPT) + spline interpolation.

The converged branch is exported to reference_<FRF_SOURCE>_fbs_hbm.csv in
PHYSICAL coordinates so the solve is done once and the solution recovered from
the CSV alone (same idea as the pyhbm Craig-Bampton reference): per continuation
point omega, the corrector diagnostics and, for every harmonic, the complex
amplitudes of the 6-DoF interface gap x_rel (the Newton unknown = VP_A - VP_B,
which drives the joint force) and of the FULL virtual-point response
(compute_full_response) -- every VPT channel of A and B, named as in the pyFBS
workbook (S1..S12) with the interface DoFs as A_vp_*/B_vp_*. See
save_physical_solution; the comment header documents the reconstruction formula
and every column.
"""

import sys
import time
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib.pyplot as plt

import pyfbs
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, ModalVPFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

from dynamical_system import build_testbench_data, TestbenchDryFriction, N_IF

# ---------------------------------------------------------------------------
# Coupling parameters -- tune these.
#   mu_trans : friction coefficient [-]
#   N        : bolt clamping force [N]; both clamp faces carry mu_trans*N, so
#              the x-y friction force saturates at the slip force 2*mu_trans*N
#   k_trans  : penalty stiffness [N/m] tying the normal (z) interface gap
#   G        : geometry factor [m] of the torsional friction about z = effective
#              contact radius; the spin moment saturates at 2*mu_trans*N*G and
#              G*rzdot is the sliding speed fed to the same tanh regularization
#   k_rot    : rotational penalty stiffness [Nm/rad] tying the tilt gap DoFs
#              rx, ry (scale ~ k_trans*G^2)
#   alpha    : tanh regularization sharpness [s/m] in f_T = 2*mu*N*tanh(alpha*||v_T||);
#              near sticking the joint acts like a viscous damper
#              c_eff = 2*mu_trans*N*alpha (raise -> closer to ideal Coulomb,
#              but a harder Newton problem)
#   F0       : harmonic excitation amplitude [N]; the transmitted tangential
#              force vs. the slip force 2*mu_trans*N decides stick or slip
# ---------------------------------------------------------------------------
mu_trans      = 0.3
N             = 200.0
k_trans       = 1.0e8
G             = 0    # geometry factor --> effective friction radius
k_rot         = 0
alpha         = 1.0e4
modal_damping = 0.005
F0            = 100
F_RESOLUTION  = 0.1            # FRF resolution Delta f [Hz] (see linear example)
HARMONICS     = np.arange(1,20, 2)   # friction force is odd in v -> odd harmonics only
sample_number = 400

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
# Physical-solution CSV export (see module docstring).
# ---------------------------------------------------------------------------
VP_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")


def full_response_labels(vpt, prefix):
    """
    Clean, prefixed name + metadata per VP-space channel DoF of ``vpt``, in the
    row order of ``vpt.df_chn`` -- which is exactly the DoF order that
    :meth:`FBSProblem.compute_full_response` returns for this substructure
    (both FRF providers stack the response as [vpt_A.df_chn | vpt_B.df_chn]).

    Interface DoFs (the 6 virtual-point rows, Name "VP1_ux"..) are renamed
    ``<prefix>_vp_<dof>``; every other sensor channel keeps its workbook Name
    with the blank removed ("S1 X" -> "<prefix>_S1X").

    :return: (labels, meta) with meta[i] = (label, workbook_name, position(3),
        direction(3), grouping) for the CSV header.
    """
    labels, meta = [], []
    for _, row in vpt.df_chn.iterrows():
        name = str(row["Name"])
        if name.startswith("VP"):
            label = f"{prefix}_vp_{row['Description']}"
        else:
            label = f"{prefix}_{name.replace(' ', '')}"
        pos = np.array([row[f"Position_{i}"] for i in (1, 2, 3)], float)
        dvec = np.array([row[f"Direction_{i}"] for i in (1, 2, 3)], float)
        labels.append(label)
        meta.append((label, name, pos, dvec, row.get("Grouping")))
    return labels, meta


def export_header(system, data, solve_time, n_points, harmonics, frf_source,
                  labels, meta_A, meta_B, out_label, in_label):
    """Self-contained comment header: run metadata, reconstruction formula,
    joint law, excitation DoF and the id/position/direction of every exported
    physical channel (see :func:`full_response_labels`)."""
    lines = [
        f"testbench_dry_friction FBS physical forced-response reference -- "
        f"{frf_source} FRF provider",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(harmonics)}",
        f"joint friction: mu_trans = {system.mu_trans}, clamping force N = "
        f"{system.N} N, slip force 2*mu_trans*N = "
        f"{2.0 * system.mu_trans * system.N} N, alpha = {system.alpha} s/m",
        f"joint torsional friction (rz): geometry factor G = {system.G} m, "
        f"slip moment 2*mu_trans*N*G = {2.0 * system.mu_trans * system.N * system.G} Nm",
        f"joint penalty stiffness: k_trans (z) = {system.k_trans} N/m, "
        f"k_rot (rx, ry) = {system.k_rot} Nm/rad",
        f"excitation: f(t) = F0 cos(omega t) at '{in_label}', F0 = {system.F0} N",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the PHYSICAL solution:",
        "  x_rel_*: the 6-DoF interface gap (Newton unknown) = VP_A - VP_B;"
        " the joint force is f_T = 2*mu_trans*N*tanh(alpha*||v_T||)*v_T/||v_T||"
        " on the [ux, uy] gap velocity, f_z = k_trans*x on uz, M_rxy = k_rot*x"
        " on rx/ry and M_rz = 2*mu_trans*N*G*tanh(alpha*G*xdot) on rz",
        "  A_*/B_* : full virtual-point response (compute_full_response) at every"
        " VPT channel; A_vp_*/B_vp_* are the 6 interface DoFs, the rest are the"
        " workbook sensor channels S1..S12",
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) );  channel"
        " acceleration: acc_h = -(h omega)^2 a_h",
        "units: displacement amplitudes m, vp rotations rad, freq_hz Hz,"
        " omega_rad_s rad/s",
        f"columns: freq_hz, omega_rad_s | iterations, step_length (corrector"
        f" diagnostics) | uout_h1_abs_m = |a_1({out_label})|, uout_time_max_m ="
        f" max_t |{out_label}(t)| (the plotted curves) | re/im of a_h,"
        f" harmonic-major, then DoF order as listed",
        f"uout (plotted output channel) = {out_label}",
    ]
    for name, meta in (("A", meta_A), ("B", meta_B)):
        lines.append(f"channels {name}: {len(meta)} VPT DoFs (order = "
                     f"compute_full_response block for {name})")
        lines += [f"  {label}: {wname!r} grouping {grp} at "
                  f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m, direction "
                  f"[{d[0]:.7f}, {d[1]:.7f}, {d[2]:.7f}]"
                  for label, wname, p, d, grp in meta]
    return lines


def save_physical_solution(ss, problem, system, data, csv_out, solve_time,
                           harmonics, frf_source):
    """
    Export the converged branch in PHYSICAL coordinates, one row per point:
    freq/omega, corrector diagnostics, the two plotted output-channel curves
    and, per harmonic h and physical DoF, the complex amplitude a_h as a re/im
    pair, normalized so that u(t) = Re(sum_h a_h e^{1j h w t}) -- |a_h| is the
    physical amplitude, NOT the raw rFFT solver coefficient. DoFs: the 6-DoF
    interface gap x_rel (ss.fourier) followed by the full VP-space response
    (compute_full_response) named via :func:`full_response_labels`. Read back
    with pandas.read_csv(csv_out, comment="#").
    Returns (freq_hz, uout_h1_abs, uout_time_max) for the plot.
    """
    from numpy.fft import irfft

    hlist = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples

    labels_A, meta_A = full_response_labels(data["vpt_A"], "A")
    labels_B, meta_B = full_response_labels(data["vpt_B"], "B")
    full_labels = labels_A + labels_B
    assert len(full_labels) == problem.d_total, (len(full_labels), problem.d_total)
    labels = [f"x_rel_{d}" for d in VP_DOFS] + full_labels

    # raw rFFT-convention coefficients: interface gap (Newton unknown) + full
    # VP response (one provider call per converged point, as the plot already did)
    x_rel = np.array([f.coefficients[:, :, 0] for f in ss.fourier])   # (n, Nh, 6)
    n = x_rel.shape[0]
    q_full = np.empty((n, len(hlist), problem.d_total), dtype=complex)
    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        q_full[i] = problem.compute_full_response(four, w).coefficients[:, :, 0]
    raw = np.concatenate([x_rel, q_full], axis=2)          # (n, Nh, 6 + d_total)

    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in hlist])
    amp = raw * scale[None, :, None]                       # physical amplitudes

    # plotted output channel = compute_full_response DoF system.out_full
    out_label = full_labels[system.out_full]
    in_label = full_labels[system.inp_full]
    i_out, i_h1 = labels.index(out_label), hlist.index(1)
    padded = np.zeros((n, max(hlist) + 1), dtype=complex)
    padded[:, hlist] = raw[:, :, i_out]
    uout_time_max = np.abs(irfft(padded, n=n_t, axis=1)).max(axis=1)
    uout_h1_abs = np.abs(amp[:, i_h1, i_out])

    omega = np.asarray(ss.omega, dtype=float)
    freq = omega / (2.0 * np.pi)
    reim = np.stack([amp.real, amp.imag], axis=-1)         # re/im adjacent per DoF
    table = np.column_stack([
        freq, omega,
        np.asarray(ss.iterations, dtype=float),
        np.asarray(ss.step_length, dtype=float),
        uout_h1_abs, uout_time_max,
        reim.reshape(n, -1),                               # harmonic-major, then DoF
    ])
    cols = (["freq_hz", "omega_rad_s", "iterations", "step_length",
             "uout_h1_abs_m", "uout_time_max_m"]
            + [f"{p}_h{h}_{lab}" for h in hlist for lab in labels
               for p in ("re", "im")])
    assert table.shape[1] == len(cols)

    header = export_header(system, data, solve_time, n, harmonics,
                           frf_source, labels,
                           meta_A, meta_B, out_label, in_label)
    with open(csv_out, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write(f"# uout_time_max_m: max |{out_label}(t)| over the {n_t} time"
                 f" samples of one period\n")
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, table, delimiter=",", fmt="%.10e")
    print(f"physical solution written: {csv_out}  ({n} points, {len(labels)}"
          f" DoFs x {len(hlist)} harmonics, {solve_time:.1f} s solve)")
    return freq, uout_h1_abs, uout_time_max


def plot_joint_diagnostics(ss, system, data, f_lo, f_hi):
    """
    Joint-level diagnostic figure, everything in terms of the VP relative
    displacement x_rel = VP_A - VP_B (the Newton unknown):

      top:    NFRC -- 1st-harmonic amplitude ||a_1||_2 of the tangential gap
              [ux, uy] (the friction plane) and |a_1| of the normal (z) gap,
              with the frictionless linear reference (mu = 0, z spring only)
              and a peak marker. In gap terms the STUCK limit is x_rel -> 0,
              so the meaningful linear reference is the frictionless one.
      middle: hysteresis loop f_T vs gap and the force-velocity law f_T vs
              gap velocity at the peak point, both projected on the dominant
              slip direction e (major axis of the 1st-harmonic gap ellipse;
              the tangential motion is 2D, a scalar loop needs a direction).
      bottom: one period of gap, gap velocity and friction force along e.

    The friction force is evaluated through system.interface_force, i.e. the
    exact law the solver used. Returns the figure.
    """
    from numpy.fft import irfft

    hlist = [int(h) for h in Fourier.harmonics]
    n_t   = Fourier.number_of_time_samples
    i_h1  = hlist.index(1)
    H     = max(hlist)

    raw   = np.array([f.coefficients[:, :, 0] for f in ss.fourier])  # (n, Nh, 6)
    omega = np.asarray(ss.omega, dtype=float)
    f_sol = omega / (2.0 * np.pi)
    a1    = raw[:, i_h1, :] * (2.0 / n_t)              # physical h1 amplitudes
    gap_T = np.sqrt(np.abs(a1[:, 0])**2 + np.abs(a1[:, 1])**2)
    gap_z = np.abs(a1[:, 2])
    i_pk  = int(np.argmax(gap_T))
    w_pk, f_pk = omega[i_pk], f_sol[i_pk]

    # frictionless linear reference on the presampled admittance grid: only the
    # penalty springs couple -> x_rel = (I + (B Y B^T) K)^-1 B Y f_ext, the same
    # residual convention the solver uses (K = diag(0, 0, k_trans, k_rot, k_rot, 0)).
    freq, Y, Bc = data["freq"], data["Y"], data["Bc"]
    i0, i1 = np.searchsorted(freq, f_lo), np.searchsorted(freq, f_hi) + 1
    BYB  = np.einsum("ij,fjk->fik", Bc, Y[i0:i1] @ Bc.T)             # (Nf, 6, 6)
    gext = (Y[i0:i1, :, system.inp_full] @ Bc.T) * system.F0         # (Nf, 6)
    K = np.diag([0.0, 0.0, system.k_trans, system.k_rot, system.k_rot, 0.0])
    x_lin = np.linalg.solve(np.eye(6)[None] + BYB @ K, gext[:, :, None])[:, :, 0]
    ref_T = np.sqrt(np.abs(x_lin[:, 0])**2 + np.abs(x_lin[:, 1])**2)

    # one period at the peak: physical time signals per DoF (rFFT convention,
    # same reconstruction as save_physical_solution), force via the system law.
    padded = np.zeros((6, H + 1), dtype=complex)
    padded[:, hlist] = raw[i_pk].T
    x_t = irfft(padded, n=n_t, axis=1)                               # (6, n_t) [m]
    v_t = irfft(padded * (1j * w_pk * np.arange(H + 1))[None, :], n=n_t, axis=1)
    tau = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    f_j = system.interface_force(x_t.T[:, :, None], v_t.T[:, :, None], tau)[:, :, 0]

    # dominant slip direction e = major axis of the h1 tangential gap ellipse
    M = np.stack([a1[i_pk, :2].real, a1[i_pk, :2].imag], axis=1)
    e = np.linalg.svd(M)[0][:, 0]
    xs, vs, fs = x_t[:2].T @ e, v_t[:2].T @ e, f_j[:, :2] @ e
    Fs = 2.0 * system.mu_trans * system.N

    fig = plt.figure(figsize=(10, 11))
    gs  = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 0.9],
                           hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])

    ax0.semilogy(f_sol, 1e6 * gap_T, color="#1f77b4",
                 label="||a_1|| tangential gap [ux, uy]")
    ax0.semilogy(f_sol, 1e6 * gap_z, color="#7f7f7f", lw=0.9,
                 label="|a_1| normal gap uz")
    ax0.semilogy(freq[i0:i1], 1e6 * ref_T, "k:", lw=1.0,
                 label="no friction (mu = 0, z spring only)")
    ax0.plot(f_pk, 1e6 * gap_T[i_pk], "rv", ms=7)
    ax0.annotate("peak", (f_pk, 1e6 * gap_T[i_pk]), textcoords="offset points",
                 xytext=(6, 6), color="red", fontsize=9)
    ax0.set_xlim(f_lo, f_hi)
    ax0.set_xlabel("frequency [Hz]")
    ax0.set_ylabel("|1st harmonic| of x_rel  [um]")
    ax0.set_title(f"VP relative displacement NFRC  (N = {system.N:g} N, "
                  f"F0 = {system.F0:g} N, mu = {system.mu_trans:g}, "
                  f"k_trans = {system.k_trans:g} N/m)")
    ax0.grid(True, which="both", alpha=0.3)
    ax0.legend()

    ax1.plot(1e6 * xs, fs, color="purple", lw=1.0)
    ax2.plot(1e3 * vs, fs, color="teal", lw=1.2)
    for ax in (ax1, ax2):
        ax.axhline(+Fs, color="gray", ls="--", lw=0.8)
        ax.axhline(-Fs, color="gray", ls="--", lw=0.8)
        ax.grid(True, alpha=0.3)
    ax1.set_xlabel("gap along e  [um]")
    ax1.set_ylabel("friction force along e  [N]")
    ax1.set_title(f"hysteresis loop at peak ({f_pk:.1f} Hz)")
    ax2.set_xlabel("gap velocity along e  [mm/s]")
    ax2.set_ylabel("friction force along e  [N]")
    ax2.set_title("force-velocity law  2 mu N tanh(alpha ||v_T||)")

    t_ms = 1e3 * tau / w_pk
    ax3.plot(t_ms, 1e6 * xs, "k-", label="gap . e  [um]")
    ax3.plot(t_ms, 1e3 * vs, "g--", label="gap velocity . e  [mm/s]")
    ax3r = ax3.twinx()
    ax3r.plot(t_ms, fs, color="deepskyblue", label="friction force . e  [N]")
    ax3.set_xlabel(f"time over one period [ms]  (f = {f_pk:.1f} Hz)")
    ax3.set_ylabel("gap [um],  gap velocity [mm/s]")
    ax3r.set_ylabel("friction force [N]")
    lines = ax3.get_lines() + ax3r.get_lines()
    ax3.legend(lines, [l.get_label() for l in lines], loc="upper right",
               fontsize=8)
    ax3.grid(True, alpha=0.3)
    return fig


# ---------------------------------------------------------------------------
# 1) Data: ANSYS lab testbench -> block-diagonal admittance Y = diag(Y_A, Y_B).
# ---------------------------------------------------------------------------
data = build_testbench_data(mu_trans, k_trans, alpha, G=G, k_rot=k_rot,
                            f_resolution=F_RESOLUTION, modal_damping=modal_damping)
freq, omega, Y = data["freq"], data["omega"], data["Y"]
nA, nB         = data["nA"], data["nB"]    # (data["N"] = total DoFs; the name N
                                           # is taken by the clamping force)

# ---------------------------------------------------------------------------
# 2) Continuation window: covers the first coupled resonances. Friction does
#    not bend the branches like the cubic spring; it saturates the transmitted
#    tangential force between stick (low amplitude, near-rigid x-y coupling)
#    and slip.
# ---------------------------------------------------------------------------
f_lo, f_hi = 1, 1000
w_lo, w_hi = 2.0 * np.pi * f_lo, 2.0 * np.pi * f_hi
print(f"Continuation window: {f_lo:.1f}..{f_hi:.1f} Hz")

system = TestbenchDryFriction(data, F0=F0, N=N, sample_number=sample_number)
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

problem = FBSProblem(system, provider, AFT())      # asserts B cols == provider.n_dofs (= total DoFs)
solver  = HarmonicBalanceMethod(
    harmonics=HARMONICS, freq_domain_ode=problem,
    corrector_parameterization=ArcLengthParameterization,
    predictor=TangentPredictorBordered)

# cold zero start at the top of the window; reference direction sweeps omega downward.
ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=w_hi)
rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex), omega=-1.0)

print(f"\n[{FRF_SOURCE} FRF] continuation:")
t0 = time.perf_counter()
ss = solver.solve_and_continue(
    initial_guess                 = ig,
    initial_reference_direction   = rd,
    maximum_number_of_solutions   = 50000,
    angular_frequency_range       = [w_lo, w_hi],
    solver_kwargs                 = {"maximum_iterations": 500, "absolute_tolerance": F0 * 1e-7},
    step_length_adaptation_kwargs = {"base": 2.5,
                                     "initial_step_length": 1,
                                     "maximum_step_length": 10,
                                     "minimum_step_length": 1e-6,
                                     "goal_number_of_iterations": 8},
    jacobian_update_frequency     = 1,
    verbose=True,
)
solve_time = time.perf_counter() - t0

# Export the branch in physical coordinates (recoverable from the CSV alone) and
# reuse its output-channel curves for the plot. uout_time_max == the previous
# per-point peak |u_out(t)|; uout_h1_abs == |1st-harmonic amplitude|.
csv_out = Path(__file__).resolve().parent / f"reference_{FRF_SOURCE}_fbs_hbm.csv"
f_sol, uout_h1_abs, amp = save_physical_solution(
    ss, problem, system, data, csv_out, solve_time, HARMONICS, FRF_SOURCE)

# ---------------------------------------------------------------------------
# 4) Figure: nonlinear forced-response curve.
# ---------------------------------------------------------------------------
fig, (ax_max, ax_h1) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax_max.semilogy(f_sol, amp, '-', color="#2ca02c",
                label=f"nonlinearFBS solver ({FRF_SOURCE} FRF)")
ax_max.set_ylabel("max|u_out(t)|  [m]")
ax_max.set_title("Dry-friction FBS coupling of testbench A + B")
ax_max.grid(True, which="both", alpha=0.3)
ax_max.legend()

ax_h1.semilogy(f_sol, uout_h1_abs, '-', color="#2ca02c",
               label="|1st harmonic amplitude|")
ax_h1.set_xlim(f_lo, f_hi)
ax_h1.set_xlabel("Frequency [Hz]")
ax_h1.set_ylabel("|1st harmonic|  [m]")
ax_h1.grid(True, which="both", alpha=0.3)
ax_h1.legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 4b) Joint diagnostics in terms of the VP relative displacement: NFRC of the
#     gap's 1st harmonic, hysteresis loop, force-velocity law and one period
#     at the peak (see plot_joint_diagnostics).
# ---------------------------------------------------------------------------
plot_joint_diagnostics(ss, system, data, f_lo, f_hi)

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
    view = pyfbs.display.View3D(title="Dry-friction coupling -- response animation",
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
