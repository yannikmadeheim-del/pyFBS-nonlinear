"""
One configurable nonlinearFBS solve of the lab testbench with a composable joint.

Edit CONFIG at the top -- it holds the joint (one entry per element, each with
the interface DoFs it acts on), the excitation, the FRF source, the continuation
window and the solver parts -- then run this file. It builds the testbench data,
traces the forced-response branch by AFT + arc-length HBM continuation, exports
the branch in PHYSICAL coordinates to results/<name>.csv and plots it.

The CSV follows the shared export convention of the other examples (complex
harmonic amplitudes a_h as re_h<h>_<dof> / im_h<h>_<dof> with
u(t) = Re(sum_h a_h e^{i h omega t})), so its curves overlay with the existing
reference CSVs. One extra header line, ``# config_json: {...}``, records the
exact CONFIG that produced it -- that is what makes a run reproducible and what
plot_comparison.py turns into automatic legend labels.

``build_data_and_provider`` and ``solve_config`` are importable, so study.py
runs parameter permutations without duplicating any of this.
"""

import json
import sys
import time
import warnings
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, ModalVPFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, OrthogonalParameterization,
    TangentPredictorBordered, TangentPredictorOne, TangentPredictorTwo,
    TangentPredictorRobust, ExponentialAdaptation, BiExponentialAdaptation,
)

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import plotting
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison.dynamical_system import (
    build_testbench_data, make_joints, TestbenchJoint, N_IF, VP_DOFS,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# CONFIG -- system, then solver, then output. Plain JSON types only: the whole
# dict is written into the result CSV header and read back by the plotters.
# ---------------------------------------------------------------------------
CONFIG = dict(
    # --- system -----------------------------------------------------------
    # one entry per joint element; "dofs" says which interface DoFs it acts on,
    # named out of ("ux", "uy", "uz", "rx", "ry", "rz"). Elements are summed, so
    # several may share a DoF (e.g. a linear and a cubic spring on uz).
    joints = [dict(type="linear",   k=1.0e6, c=0.5, dofs=("ux", "uy", "uz")),
              dict(type="linear",   k=1.0e3, c=0.1, dofs=("rx", "ry", "rz")),
              dict(type="cubic",    alpha=1.0e8,    dofs=("uz",)),
              dict(type="friction", mu=0.3, N=200.0, alpha_reg=1.0e4,
                   dofs=("ux", "uy"))],
    F0 = 200.0,                    # harmonic excitation amplitude [N]
    modal_damping = 0.005,         # modal damping ratio of both substructures
    solver = "pyfbs-nlfbs",        # names the pipeline in mixed comparisons
    # measurement table, by file name without extension in measurements/
    workbook = measurements.DEFAULT,

    # --- solver -----------------------------------------------------------
    frf_source = "experimental",   # "modal" | "experimental"
    static_correction = True,      # residual flexibility; "experimental" path only
    limit_modes = None,            # free-interface modes per substructure in Y (None = all)
    no_modes = 100,                # modes the eigensolve computes (>= limit_modes)
    f_resolution = 0.1,            # only consumed by the "experimental" provider
    harmonics = [1, 3, 5, 7], sample_number = 256,
    f_lo = 1.0, f_hi = 2500.0, sweep = "down",       # "down" | "up"
    parameterization = "ArcLengthParameterization",  # | OrthogonalParameterization
    predictor = "TangentPredictorBordered",          # | TangentPredictorOne
                                                     # | TangentPredictorTwo
                                                     # | TangentPredictorRobust
    step_adaptation = "ExponentialAdaptation",       # | BiExponentialAdaptation
    solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_kwargs = {"base": 2.0, "initial_step_length": 0.01,
                   "maximum_step_length": 1.0, "minimum_step_length": 1e-6,
                   "goal_number_of_iterations": 3},
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
)

# --- output ---------------------------------------------------------------
SAVE_FULL_RESPONSE = True   # False -> export only the 6 x_rel DoFs + the output channel
RESULT_NAME = None          # None -> auto-generated from the config

# solver parts addressable by name from CONFIG (and hence from a CSV header)
SOLVER_PARTS = {cls.__name__: cls for cls in (
    ArcLengthParameterization, OrthogonalParameterization,
    TangentPredictorBordered, TangentPredictorOne, TangentPredictorTwo,
    TangentPredictorRobust, ExponentialAdaptation, BiExponentialAdaptation)}


def synthesis_f_end(cfg):
    """End frequency [Hz] of the synthesized FRF grid.

    The ExperimentalFRF spline is evaluated at n*omega, so the grid has to reach
    max(harmonics) * f_hi. frf_synth samples np.arange(f_start, f_end, df), whose
    end is EXCLUSIVE, so one resolution step is added -- without it the highest
    harmonic of the first continuation point falls just outside the data and the
    provider extrapolates. ModalVPFRF synthesizes at the exact n*omega and never
    touches the grid, so a window just above f_hi is enough there -- which saves
    a lot of build time and memory.
    """
    if cfg["frf_source"] == "experimental":
        return max(cfg["harmonics"]) * cfg["f_hi"] * 1.02 + cfg["f_resolution"]
    return 1.5 * cfg["f_hi"]


def build_data_and_provider(cfg):
    """Testbench data + the FRF provider selected by ``cfg["frf_source"]``.

    This is the expensive part and depends on the joint only through the
    frequency window and the mode count, so study.py builds it once per group
    of runs -- ``group_key`` there lists both.
    """
    data = build_testbench_data(f_resolution=cfg["f_resolution"],
                                modal_damping=cfg["modal_damping"],
                                f_end=synthesis_f_end(cfg),
                                limit_modes=cfg.get("limit_modes"),
                                no_modes=cfg.get("no_modes", 100),
                                workbook=cfg.get("workbook",
                                                 measurements.DEFAULT),
                                static_correction=cfg.get("static_correction", True))
    if cfg["frf_source"] == "modal":
        if cfg.get("static_correction", True):
            # the correction sits in data["Y"], which ModalVPFRF never reads
            warnings.warn("static_correction is set but the 'modal' provider "
                          "ignores it -- use frf_source='experimental' for the "
                          "residual-flexibility corrected admittance")
        provider = ModalVPFRF.from_substructures(
            [(data["MK_A"], data["vpt_A"], data["df_chn_A"], data["df_imp_A"]),
             (data["MK_B"], data["vpt_B"], data["df_chn_B"], data["df_imp_B"])],
            modal_damping=data["modal_damping"],
            limit_modes=cfg.get("limit_modes"))
    elif cfg["frf_source"] == "experimental":
        # accuracy is bounded by f_resolution here: the branch is only as sharp
        # as the spline through the sampled admittance grid.
        provider = ExperimentalFRF(data["omega"], data["Y"])
    else:
        raise ValueError(f"frf_source must be 'modal' or 'experimental', "
                         f"got {cfg['frf_source']!r}")
    return data, provider


def solve_config(cfg, data, provider):
    """
    Trace the forced-response branch of ``cfg``.

    The wiring order is load-bearing: ``Fourier`` state is class-level and
    ``FBSProblem.__init__`` snapshots ``Fourier.number_of_harmonics``, so
    ``update_dependencies`` must run BEFORE the problem is constructed, and
    again for every run whose harmonics or sample_number differ.

    :returns: (system, problem, solution_set, solve_time_s)
    """
    harmonics = list(cfg["harmonics"])
    HarmonicBalanceMethod.update_dependencies(harmonics, cfg["sample_number"])

    system = TestbenchJoint(data, make_joints(cfg["joints"]), F0=cfg["F0"],
                            sample_number=cfg["sample_number"])
    problem = FBSProblem(system, provider, AFT())   # asserts B cols == provider.n_dofs
    solver = HarmonicBalanceMethod(
        harmonics=harmonics, freq_domain_ode=problem,
        corrector_parameterization=SOLVER_PARTS[cfg["parameterization"]],
        predictor=SOLVER_PARTS[cfg["predictor"]],
        step_length_adaptation=SOLVER_PARTS[cfg["step_adaptation"]])

    w_lo, w_hi = 2.0 * np.pi * cfg["f_lo"], 2.0 * np.pi * cfg["f_hi"]
    # cold zero start at the end of the window the sweep starts from; the
    # reference direction orients the first tangent.
    if cfg["sweep"] == "down":
        w_start, direction = w_hi, -1.0
    elif cfg["sweep"] == "up":
        w_start, direction = w_lo, +1.0
    else:
        raise ValueError(f"sweep must be 'down' or 'up', got {cfg['sweep']!r}")
    ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=w_start)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((N_IF, 1), complex),
                                                   omega=direction)

    t0 = time.perf_counter()
    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = cfg["maximum_number_of_solutions"],
        angular_frequency_range       = [w_lo, w_hi],
        solver_kwargs                 = dict(cfg["solver_kwargs"]),
        step_length_adaptation_kwargs = dict(cfg["step_kwargs"]),
        jacobian_update_frequency     = cfg["jacobian_update_frequency"],
        verbose                       = True,
    )
    return system, problem, ss, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Physical-solution CSV export
# ---------------------------------------------------------------------------

def full_response_labels(vpt, prefix):
    """
    Clean, prefixed name + metadata per VP-space channel DoF of ``vpt``, in the
    row order of ``vpt.df_chn`` -- exactly the DoF order that
    :meth:`FBSProblem.compute_full_response` returns for this substructure
    (both providers stack the response as [vpt_A.df_chn | vpt_B.df_chn]).

    Interface DoFs (Name "VP1_ux"..) become ``<prefix>_vp_<dof>``; every other
    sensor channel keeps its workbook Name without the blank ("S1 X" -> "A_S1X").

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


def export_header(cfg, system, solve_time, n_points, labels, meta, out_label,
                  in_label, n_t, full):
    """Compact, self-contained comment header: run metadata, the joint element
    list, the excitation, the reconstruction formula, units and the
    id/position/direction of every exported channel. The machine-readable
    ``config_json`` line is written separately by :func:`save_solution`."""
    lines = [
        f"testbench_joint_comparison FBS physical forced-response -- "
        f"{cfg['frf_source']} FRF provider",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(cfg['harmonics'])}",
        f"measurement table: {cfg.get('workbook', measurements.DEFAULT)}",
        f"joint: {len(system.joints)} element(s) on the 6-DoF VP gap "
        f"x_rel = VP_A - VP_B, forces summed",
    ]
    for spec in cfg["joints"]:
        spec = dict(spec)
        kind = spec.pop("type")
        args = ", ".join(f"{k}={v}" for k, v in spec.items())
        lines.append(f"  {kind}: {args}")
    lines += [
        f"excitation: f(t) = F0 cos(omega t) at '{in_label}', F0 = {system.F0} N",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the PHYSICAL solution:",
        "  x_rel_*: the 6-DoF interface gap (Newton unknown) = VP_A - VP_B; the"
        " joint elements listed above act on it and on its velocity",
        ("  A_*/B_* : full virtual-point response (compute_full_response) at every"
         " VPT channel; A_vp_*/B_vp_* are the 6 interface DoFs, the rest are the"
         " workbook sensor channels S1..S12"
         if full else
         "  only the output channel is exported besides x_rel_*"
         " (SAVE_FULL_RESPONSE = False)"),
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
        f"uout_time_max_m: max |{out_label}(t)| over the {n_t} time samples of"
        f" one period",
    ]
    for label, wname, p, d, grp in meta:
        if label in labels:
            lines.append(f"  {label}: {wname!r} grouping {grp} at "
                         f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m, direction "
                         f"[{d[0]:.7f}, {d[1]:.7f}, {d[2]:.7f}]")
    return lines


def save_solution(path, cfg, ss, problem, system, data, solve_time, full=True):
    """
    Write the converged branch in PHYSICAL coordinates, one row per point:
    freq/omega, corrector diagnostics, the two plotted output-channel curves
    and, per harmonic h and DoF, the complex amplitude a_h as a re/im pair,
    scaled so that u(t) = Re(sum_h a_h e^{i h omega t}) -- |a_h| is the physical
    amplitude, not the raw rFFT solver coefficient. Read back with
    ``pandas.read_csv(path, comment="#")``.

    :param full: True -> the 6-DoF gap plus the full virtual-point response;
        False -> the 6-DoF gap plus the output channel only. Every other column
        and the config_json line are identical either way.
    :returns: (freq_hz, uout_h1_abs, uout_time_max) for the plot.
    """
    from numpy.fft import irfft

    hlist = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples

    labels_A, meta_A = full_response_labels(data["vpt_A"], "A")
    labels_B, meta_B = full_response_labels(data["vpt_B"], "B")
    full_labels = labels_A + labels_B
    assert len(full_labels) == problem.d_total, (len(full_labels), problem.d_total)
    out_label = full_labels[system.out_full]
    in_label = full_labels[system.inp_full]

    # raw rFFT-convention coefficients: interface gap (Newton unknown) + full
    # VP response (one provider call per converged point)
    x_rel = np.array([f.coefficients[:, :, 0] for f in ss.fourier])   # (n, Nh, 6)
    n = x_rel.shape[0]
    q_full = np.empty((n, len(hlist), problem.d_total), dtype=complex)
    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        q_full[i] = problem.compute_full_response(four, w).coefficients[:, :, 0]

    gap_labels = [f"x_rel_{d}" for d in VP_DOFS]
    if full:
        raw = np.concatenate([x_rel, q_full], axis=2)
        labels = gap_labels + full_labels
    else:
        raw = np.concatenate([x_rel, q_full[:, :, [system.out_full]]], axis=2)
        labels = gap_labels + [out_label]

    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in hlist])
    amp = raw * scale[None, :, None]                       # physical amplitudes

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

    header = export_header(cfg, system, solve_time, n, labels, meta_A + meta_B,
                           out_label, in_label, n_t, full)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write(f"# config_json: {json.dumps(cfg, sort_keys=True)}\n")
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, table, delimiter=",", fmt="%.10e")
    print(f"physical solution written: {path}  ({n} points, {len(labels)}"
          f" DoFs x {len(hlist)} harmonics, {solve_time:.1f} s solve)")
    return freq, uout_h1_abs, uout_time_max


def result_path(cfg, name=None):
    """results/<name>.csv; the auto name is the joint types plus the FRF source
    (e.g. linear+cubic_modal.csv)."""
    if name is None:
        types = list(dict.fromkeys(spec["type"] for spec in cfg["joints"]))
        name = f"{'+'.join(types)}_{cfg['frf_source']}.csv"
    return HERE / "results" / name


if __name__ == "__main__":
    print(f"Continuation window: {CONFIG['f_lo']:.1f}..{CONFIG['f_hi']:.1f} Hz"
          f" ({CONFIG['sweep']}ward sweep)")
    data, provider = build_data_and_provider(CONFIG)

    print(f"\n[{CONFIG['frf_source']} FRF] continuation:")
    system, problem, ss, solve_time = solve_config(CONFIG, data, provider)

    out_csv = result_path(CONFIG, RESULT_NAME)
    save_solution(out_csv, CONFIG, ss, problem, system, data, solve_time,
                  full=SAVE_FULL_RESPONSE)

    curve = plotting.read_result(out_csv)
    labels = [out_csv.stem]
    styles = [dict(color="#2ca02c", linestyle="-", linewidth=1.6)]
    # one curve -> every field of its config is "shared", so the note becomes
    # the parameter list of this run
    note = plotting.shared_note([curve["config"]])
    plotting.nfrc_figure([curve], labels, styles, note=note,
                         title="Testbench A + B -- assembled joint, output channel",
                         xlim=(CONFIG["f_lo"], CONFIG["f_hi"]))
    plotting.gap_figure([curve], labels, styles, note=note,
                        title="Testbench A + B -- interface gap x_rel (translations)",
                        xlim=(CONFIG["f_lo"], CONFIG["f_hi"]))
    plt.show()
