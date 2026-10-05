"""
Linear LM-FBS reference for the four DoE workbooks -- no continuation, no HBM.

The pyFBS runs in results/doe_static_correction_linear/ trace the forced
response with the nonlinear solver, but their joint is linear (a linear spring
and damper on all six virtual-point DoFs, the cubic term at alpha = 0). For a
linear joint the coupled admittance follows in closed form from the LM-FBS
coupling with relaxed compatibility (thesis eq. 2.11):

    Y_tilde = Y - Y B^T (B Y B^T + Gamma)^-1 B Y,   Gamma = diag(1 / (k + i omega c))

so the nonlinear solver has to reproduce this curve when it runs on the same
admittance data. Everything that defines a run -- workbook, joint, F0, modal
damping, mode count, static correction, frequency window -- is read from the
``config_json`` header of the pyFBS CSV it is compared against, so the
reference cannot drift away from the runs it verifies.

The results are written in the physical-solution CSV format of main.py
(harmonic 1 only), so plotting.read_result and plot_comparison.py read them
unchanged::

    python -m pyfbs.nonlinearFBS.examples.testbench_joint_comparison.lm_fbs_linear_reference
"""

import json
import sys
import time
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import pandas as pd

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import main as pyfbs_main
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import plotting
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison.dynamical_system import (
    build_testbench_data, resolve_dofs, N_IF, VP_DOFS,
)

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "results" / "collocated_impacts_static_correction_linear"
SOURCE_FILES = ["pyFBS_collocated_all.csv",
                "pyFBS_collocated_sensors.csv",
                "pyFBS_collocated_impacts.csv",
                "pyFBS_non_collocated.csv"]
OUT_DIR = HERE / "results" / "LM-FBS_newResults"

# settings that only steer the AFT + continuation solver; the closed-form
# reference has none of them, so they are not carried into its config_json
CONTINUATION_KEYS = ("parameterization", "predictor", "step_adaptation",
                     "solver_kwargs", "step_kwargs", "maximum_number_of_solutions",
                     "jacobian_update_frequency", "sweep", "sample_number")


def joint_impedance(joints, omega):
    """
    Dynamic stiffness of the joint on the six interface DoFs, k + i omega c
    summed over the linear elements that act on each DoF.

    :param joints: joint element specs as stored in ``config_json``.
    :param omega: angular frequencies [rad/s], shape (n_freq,).
    :returns: diagonal of the joint impedance Z_j, shape (n_freq, 6).
    """
    Z = np.zeros((len(omega), N_IF), dtype=complex)
    for spec in joints:
        kind = spec["type"]
        idx = resolve_dofs(spec["dofs"])
        if kind == "linear":
            Z[:, idx] += (spec.get("k", 0.0)
                          + 1j * omega * spec.get("c", 0.0))[:, None]
        elif kind == "cubic" and spec.get("alpha", 0.0) == 0.0 \
                and spec.get("beta", 0.0) == 0.0:
            continue                    # a cubic element at zero adds no force
        else:
            raise ValueError(f"joint element {spec} is not linear -- the LM-FBS "
                             f"reference only exists for a linear joint")
    # A DoF without a spring or damper is not connected at all; its Gamma entry
    # would be infinite and eq. (2.11) does not apply there.
    free = [VP_DOFS[i] for i in range(N_IF) if np.any(Z[:, i] == 0.0)]
    if free:
        raise ValueError(f"interface DoF(s) {', '.join(free)} carry no joint "
                         f"stiffness or damping -- Gamma is not defined there")
    return Z


def lm_fbs_coupling(Y, B, Gamma):
    """
    Coupled admittance with relaxed compatibility, thesis eq. (2.11):

        Y_tilde = Y - Y B^T (B Y B^T + Gamma)^-1 B Y

    evaluated for every frequency line at once.

    :param Y: uncoupled block-diagonal admittance, shape (n_freq, N, N).
    :param B: signed-Boolean matrix of the interface gap, shape (6, N).
    :param Gamma: dynamic flexibility of the joint, shape (n_freq, 6, 6).
    :returns: Y_tilde, shape (n_freq, N, N).
    """
    BY = B @ Y                                          # (n_freq, 6, N)
    Y_int = BY @ B.T                                    # interface flexibility B Y B^T
    return Y - (Y @ B.T) @ np.linalg.solve(Y_int + Gamma, BY)


def reference_config(source_cfg, source_name):
    """The source run's config, reduced to what the closed-form solve used."""
    cfg = {k: v for k, v in source_cfg.items() if k not in CONTINUATION_KEYS}
    cfg["solver"] = "lm-fbs-linear"
    cfg["harmonics"] = [1]                  # a linear system answers at 1*omega only
    cfg.setdefault("static_correction", True)
    cfg.setdefault("no_modes", 100)
    cfg["source_csv"] = source_name
    return cfg


def solve_reference(cfg):
    """
    Build the admittance of ``cfg`` and couple it with the linear joint.

    :returns: (freq_hz, u, x_rel, data) with u the physical harmonic-1
        amplitude of every VP-space DoF, shape (n_freq, N), and x_rel = B u the
        interface gap, shape (n_freq, 6).
    """
    # The window only has to reach f_hi: a linear response has no higher
    # harmonics to evaluate. Two extra lines because the synthesis grid is an
    # arange whose end is exclusive.
    data = build_testbench_data(f_resolution=0.01,
                                modal_damping=cfg["modal_damping"],
                                f_end=cfg["f_hi"] + 2 * cfg["f_resolution"],
                                limit_modes=cfg.get("limit_modes"),
                                no_modes=cfg["no_modes"],
                                workbook=cfg.get("workbook", measurements.DEFAULT),
                                static_correction=cfg["static_correction"])

    # the grid is an arange, so a line meant to sit exactly on f_lo or f_hi can
    # carry a rounding error of a few ulp; the tolerance keeps it in the window
    tol = 1e-6 * cfg["f_resolution"]
    keep = (data["freq"] >= cfg["f_lo"] - tol) & (data["freq"] <= cfg["f_hi"] + tol)
    freq = data["freq"][keep]
    omega = 2.0 * np.pi * freq
    Y = data["Y"][keep]
    B = data["Bc"]

    Gamma = np.zeros((len(freq), N_IF, N_IF), dtype=complex)
    Gamma[:, np.arange(N_IF), np.arange(N_IF)] = 1.0 / joint_impedance(cfg["joints"], omega)

    Y_tilde = lm_fbs_coupling(Y, B, Gamma)
    # f(t) = F0 cos(omega t) = Re(F0 e^{i omega t}), so the response amplitude
    # a with u(t) = Re(a e^{i omega t}) is the excited column times F0
    u = Y_tilde[:, :, data["inp_full"]] * cfg["F0"]
    x_rel = u @ B.T
    return freq, u, x_rel, data


def save_reference(path, cfg, freq, u, x_rel, data, solve_time):
    """
    Write the reference in the physical-solution CSV format of main.py:
    freq/omega, the two plotted output-channel curves and the re/im pair of the
    harmonic-1 amplitude of the gap x_rel_* and of every VP-space channel.

    :returns: |a_1| of the output channel, shape (n_freq,).
    """
    labels_A, meta_A = pyfbs_main.full_response_labels(data["vpt_A"], "A")
    labels_B, meta_B = pyfbs_main.full_response_labels(data["vpt_B"], "B")
    full_labels = labels_A + labels_B
    assert len(full_labels) == u.shape[1], (len(full_labels), u.shape[1])
    out_label = full_labels[data["out_full"]]
    in_label = full_labels[data["inp_full"]]

    labels = [f"x_rel_{d}" for d in VP_DOFS] + full_labels
    amp = np.concatenate([x_rel, u], axis=1)
    # with one harmonic the period maximum of Re(a e^{i theta}) is |a| itself
    uout_abs = np.abs(u[:, data["out_full"]])

    table = np.column_stack([freq, 2.0 * np.pi * freq, uout_abs, uout_abs,
                             np.stack([amp.real, amp.imag], axis=-1)
                             .reshape(len(freq), -1)])
    cols = (["freq_hz", "omega_rad_s", "uout_h1_abs_m", "uout_time_max_m"]
            + [f"{p}_h1_{lab}" for lab in labels for p in ("re", "im")])
    assert table.shape[1] == len(cols)

    header = [
        "testbench_joint_comparison linear LM-FBS reference -- closed-form "
        "coupling, thesis eq. (2.11)",
        "Y_tilde = Y - Y B^T (B Y B^T + Gamma)^-1 B Y, Gamma = diag(1/(k + i omega c))",
        f"source run: {SOURCE_DIR.name}/{cfg['source_csv']} (same workbook, "
        f"joint and admittance settings)",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {len(freq)} (every synthesis line in "
        f"{cfg['f_lo']:g}..{cfg['f_hi']:g} Hz, df = {cfg['f_resolution']:g} Hz)",
        f"measurement table: {cfg.get('workbook', measurements.DEFAULT)}",
        f"admittance: limit_modes = {cfg.get('limit_modes')}, static_correction"
        f" = {cfg['static_correction']}, modal_damping = {cfg['modal_damping']}",
        "joint on the 6-DoF VP gap x_rel = VP_A - VP_B:",
    ]
    for spec in cfg["joints"]:
        spec = dict(spec)
        kind = spec.pop("type")
        header.append(f"  {kind}: " + ", ".join(f"{k}={v}" for k, v in spec.items()))
    header += [
        f"excitation: f(t) = F0 cos(omega t) at '{in_label}', F0 = {cfg['F0']} N",
        "content: complex harmonic-1 amplitudes a_1 = re_h1_<dof> + 1j im_h1_<dof>",
        "reconstruction: u(t) = Re( a_1 exp(1j omega t) )",
        "units: displacement amplitudes m, vp rotations rad, freq_hz Hz,"
        " omega_rad_s rad/s",
        f"uout (plotted output channel) = {out_label}",
    ]
    for label, wname, p, d, grp in meta_A + meta_B:
        header.append(f"  {label}: {wname!r} grouping {grp} at "
                      f"({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) m, direction "
                      f"[{d[0]:.7f}, {d[1]:.7f}, {d[2]:.7f}]")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write(f"# config_json: {json.dumps(cfg, sort_keys=True)}\n")
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, table, delimiter=",", fmt="%.10e")
    print(f"reference written: {path}  ({len(freq)} lines)")
    return uout_abs


def peak(freq, amplitude):
    """Frequency and value of the largest amplitude."""
    i = int(np.argmax(amplitude))
    return freq[i], amplitude[i]


def main():
    summary = []
    for name in SOURCE_FILES:
        source = SOURCE_DIR / name
        source_cfg = plotting.read_config(source)
        if source_cfg is None:
            raise ValueError(f"{source} has no config_json header")
        cfg = reference_config(source_cfg, name)
        print(f"\n{name} -> workbook {cfg['workbook']}")

        t0 = time.perf_counter()
        freq, u, x_rel, data = solve_reference(cfg)
        solve_time = time.perf_counter() - t0
        out_csv = OUT_DIR / name.replace("pyFBS_", "LM-FBS_", 1)
        uout_abs = save_reference(out_csv, cfg, freq, u, x_rel, data, solve_time)

        run = pd.read_csv(source, comment="#", usecols=["freq_hz", "uout_h1_abs_m"])
        summary.append((cfg["workbook"], peak(freq, uout_abs),
                        peak(run["freq_hz"].to_numpy(),
                             run["uout_h1_abs_m"].to_numpy())))

    print("\nlargest |a_1(u_out)| in the window: LM-FBS reference vs pyFBS run")
    for workbook, (f_ref, a_ref), (f_run, a_run) in summary:
        print(f"  {workbook:<36s} LM-FBS {f_ref:8.2f} Hz {a_ref:.4e} m   "
              f"pyFBS {f_run:8.2f} Hz {a_run:.4e} m   "
              f"(amplitude ratio {a_run / a_ref:.4f})")


if __name__ == "__main__":
    main()
