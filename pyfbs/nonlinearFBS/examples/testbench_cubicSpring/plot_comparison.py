"""
Overlay of cubic-spring testbench forced-response references straight from
their physical-solution CSVs -- no solver, no ANSYS, no continuation.

Type the CSVs to compare into CSV_FILES below (or drag & drop them onto this
script / pass them as command-line arguments). Bare names are resolved
relative to this folder, so files dragged in here need no path. Every file
must follow the shared export convention (see the CSV comment headers):
complex harmonic amplitudes a_h stored as re_h<h>_<dof> / im_h<h>_<dof>
columns with u(t) = Re( sum_h a_h exp(1j h omega t) ), as written by both the
pyFBS nonlinearFBS example and the pyhbm testbench_cubicSpring_CB example
(RBE2 / RBE3 / virtual-point interface variants alike).

Figure 1 -- output channel A_S1X (sensor S1, X direction, on substructure A):
  top    max_t |u_out(t)| over one period, recomputed from the exported
         harmonics on a common dense time grid (the stored uout_time_max_m
         columns use each exporter's own AFT grid: 29 vs 256 samples),
  bottom |a_1|, the 1st-harmonic amplitude.

Figure 2 -- the relative virtual-point displacement x_rel = VP_A - VP_B, the
gap the cubic spring acts on (rotations carry no joint stiffness -> omitted):
  top    max_t ||x_rel(t)||_2 over the three translational DoFs,
  bottom ||a_1||_2 of the three translational 1st-harmonic amplitudes.
  pyFBS exports the gap directly (x_rel_*); pyhbm exports the two virtual
  points, so the gap is built as A_vp_* - B_vp_* (detected per file).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CSVs to overlay: type the file names here. Bare names are looked up in this
# folder (drag the files in beside the script); absolute paths work too.
# Dragging CSVs onto the script (or passing them as command-line arguments)
# overrides this list.
# ---------------------------------------------------------------------------
CSV_FILES = [
    "reference_modal_fbs_hbm.csv",
    "reference_RBE_average_cb_hbm.csv"
]

if len(sys.argv) > 1:
    CSV_FILES = sys.argv[1:]

HERE      = os.path.dirname(os.path.abspath(__file__))
HARMONICS = [1, 3, 5, 7]        # exported harmonics (identical in all CSVs)
N_T       = 1024                # common dense time grid for the period maxima
OUT       = "A_S1X"             # plotted output channel (uout of every export)
TRANS     = ("ux", "uy", "uz")  # translational gap DoFs (k_rot = 0)
XLIM      = (1, 500)            # displayed frequency range [Hz]
I_H1      = HARMONICS.index(1)  # position of the 1st harmonic in the CSV order

# line styles cycled over the listed files, in order.
STYLES = [
    ("-",  dict(color="#2ca02c", lw=1.6)),
    ("--", dict(color="black",   lw=1.0)),
    ("-.", dict(color="#d62728", lw=1.2)),
    (":",  dict(color="#1f77b4", lw=1.8)),
    ("-",  dict(color="#9467bd", lw=1.0)),
    ("--", dict(color="#ff7f0e", lw=1.2)),
]


def reim_cols(dof):
    """re/im column names of DoF `dof`, all harmonics, in CSV order."""
    return [f"{p}_h{h}_{dof}" for h in HARMONICS for p in ("re", "im")]


def amplitudes(df, dof):
    """Complex physical amplitudes a_h of `dof`, shape (n_points, n_harmonics)."""
    re = df[[f"re_h{h}_{dof}" for h in HARMONICS]].to_numpy()
    im = df[[f"im_h{h}_{dof}" for h in HARMONICS]].to_numpy()
    return re + 1j * im


def period_signal(amps):
    """u(theta) = Re( sum_h a_h e^{1j h theta} ) on N_T samples of one period.
    Matmul contracts the trailing harmonic axis: (..., n_h) -> (..., N_T)."""
    theta = np.linspace(0.0, 2.0 * np.pi, N_T, endpoint=False)
    return np.real(amps @ np.exp(1j * np.outer(HARMONICS, theta)))


def load_csv(path):
    """Read one reference CSV (only the needed columns) into the four curves."""
    head = pd.read_csv(path, comment="#", nrows=0)        # column names only
    if f"re_h{HARMONICS[0]}_x_rel_{TRANS[0]}" in head.columns:
        gap_dofs = [f"x_rel_{d}" for d in TRANS]          # gap exported directly
    elif f"re_h{HARMONICS[0]}_A_vp_{TRANS[0]}" in head.columns:
        gap_dofs = [f"{s}_vp_{d}" for s in ("A", "B") for d in TRANS]
    else:
        raise KeyError(f"{os.path.basename(path)}: neither x_rel_* nor "
                       "A_vp_*/B_vp_* harmonic columns found -- not a "
                       "physical-solution reference CSV?")

    wanted = set(["freq_hz"] + reim_cols(OUT)
                 + [c for d in gap_dofs for c in reim_cols(d)])
    df = pd.read_csv(path, comment="#", usecols=lambda c: c in wanted)
    missing = wanted - set(df.columns)
    if missing:
        raise KeyError(f"{os.path.basename(path)} lacks expected columns, "
                       f"e.g. {sorted(missing)[:3]}")

    a_out = amplitudes(df, OUT)                                   # (n, n_h)
    if gap_dofs[0].startswith("x_rel"):
        a_gap = np.stack([amplitudes(df, f"x_rel_{d}") for d in TRANS], axis=1)
    else:                                   # pyhbm: gap = VP_A - VP_B per DoF
        a_gap = np.stack([amplitudes(df, f"A_vp_{d}") - amplitudes(df, f"B_vp_{d}")
                          for d in TRANS], axis=1)                # (n, 3, n_h)

    sig_out = period_signal(a_out)                                # (n, N_T)
    sig_gap = period_signal(a_gap)                                # (n, 3, N_T)
    return dict(
        f       = df["freq_hz"].to_numpy(),
        out_max = np.abs(sig_out).max(axis=1),
        out_h1  = np.abs(a_out[:, I_H1]),
        gap_max = np.linalg.norm(sig_gap, axis=1).max(axis=1),
        gap_h1  = np.linalg.norm(a_gap[:, :, I_H1], axis=1),
    )


def overlay_figure(curves, labels, key_max, key_h1, title, label_max, label_h1):
    """One window: period maximum on top, 1st-harmonic amplitude below."""
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for (fmt, kw), label, c in zip(
            (STYLES[i % len(STYLES)] for i in range(len(curves))), labels, curves):
        ax_top.semilogy(c["f"], c[key_max], fmt, label=label, **kw)
        ax_bot.semilogy(c["f"], c[key_h1],  fmt, label=label, **kw)
    ax_top.set_title(title)
    ax_top.set_ylabel(label_max)
    ax_top.legend()
    ax_bot.set_ylabel(label_h1)
    ax_bot.set_xlabel("Frequency [Hz]")
    ax_bot.set_xlim(*XLIM)
    for ax in (ax_top, ax_bot):
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()


curves, labels = [], []
for name in CSV_FILES:
    path = name if os.path.isabs(name) else os.path.join(HERE, name)
    print(f"loading {os.path.basename(path)} "
          f"({os.path.getsize(path) / 1e6:.0f} MB) ...")
    c = load_csv(path)
    print(f"  {len(c['f'])} points, {c['f'].min():.1f}..{c['f'].max():.1f} Hz")
    curves.append(c)
    labels.append(os.path.splitext(os.path.basename(path))[0])

overlay_figure(curves, labels, "out_max", "out_h1",
               f"Cubic-spring testbench A + B -- output channel {OUT}",
               "max_t |u_out(t)|  [m]", f"|a_1({OUT})|  [m]")
overlay_figure(curves, labels, "gap_max", "gap_h1",
               "Cubic-spring testbench A + B -- relative VP displacement"
               " x_rel = VP_A - VP_B (translations)",
               "max_t ||x_rel(t)||_2  [m]", "||a_1(x_rel)||_2  [m]")
plt.show()
