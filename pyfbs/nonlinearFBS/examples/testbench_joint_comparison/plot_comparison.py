"""
Plot the result CSVs of one or more comparison folders -- no solver, no ANSYS.

Set COMPARE_DIRS below; that is the only thing to change when switching between
studies. The comparison folders live side by side in results/ (cubic_800N/,
friction_80N/, ...): create a new one, copy or move the CSVs to compare into it,
and name it in COMPARE_DIRS. Naming several draws them all in one run, each with
its own plot_styles.csv and its own windows. On the first run a
plot_styles.csv is written from the folder contents with automatic labels; edit
its rows afterwards to control legend text, colour, line style, line width and
-- through the row order -- the plot and legend order. CSVs that the spec does
not list are appended at the end with a printed note.

Two figures per layout: the output channel and the translational interface gap
x_rel. Which layout a curve belongs to is decided row by row in the spec's
``panel`` column: rows that leave it empty share one pair of axes and show the
period maximum over the 1st-harmonic amplitude, rows that name a panel are spread
over one subplot per name and show the period maximum alone -- the layout for
curves that differ in two parameters at once, such as the resolution sweeps. A
spec that uses both -- the solver runs once overlaid, and once beside their
references, one panel per parameter value -- lists those files twice and yields
four windows. A filled ``normalize`` column divides every curve by its own
excitation amplitude, so an amplitude sweep is compared as |x| / F0 in m/N.

The folders can also be given on the command line::

    python plot_comparison.py friction_80N friction_800N
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import plotting

COMPARE_DIRS = ["cubic_resolution_80N",
                "cubic_resolution_800N",
                ] # folder name(s) inside results/
STYLE_FILE   = "plot_styles.csv"     # style spec inside each of them
XLIM         = (1, 1000)             # displayed frequency range [Hz]

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
folders = [RESULTS / name for name in (sys.argv[1:] or COMPARE_DIRS)]
cache = {}          # a file two rows share -- one folder or two -- is read once


def unpack(rows, spec_path):
    """One layout's spec rows into the arguments its figure pair takes."""
    curves, labels, styles, panels, scales = map(list, zip(*rows))
    extra = dict(panels=panels) if any(panels) else {}
    # Normalizing stays all-or-nothing within a layout: a normalized and a raw
    # curve in one pair of axes would share a y label wrong for one of them.
    if all(s is not None for s in scales):
        extra["scales"] = scales
    elif any(s is not None for s in scales):
        raise ValueError(f"{spec_path}: the normalize column is filled for "
                         f"{sum(s is not None for s in scales)} of "
                         f"{len(scales)} rows of one layout -- fill it for all "
                         f"of them or for none")
    return curves, labels, styles, extra


def draw(folder):
    """Every figure of one comparison folder: one pair per layout its spec uses."""
    print(f"\n{folder.name}:")
    overlaid, panelled = [], []
    spec = plotting.read_style_spec(folder, STYLE_FILE)
    for path, label, style, panel, scale in spec:
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            cache[path] = plotting.read_result(path)
        curve = cache[path]
        print(f"  {len(curve['f'])} points, "
              f"{curve['f'].min():.1f}..{curve['f'].max():.1f} Hz  ->  {label}")
        row = (curve, label, style, panel, scale)
        (panelled if panel else overlaid).append(row)

    if overlaid:
        curves, labels, styles, extra = unpack(overlaid, folder / STYLE_FILE)
        extra["note"] = plotting.shared_note([c["config"] for c in curves])
        plotting.nfrc_figure(
            curves, labels, styles, xlim=XLIM, **extra,
            title=f"{folder.name} -- output channel")
        plotting.gap_figure(
            curves, labels, styles, xlim=XLIM, **extra,
            title=f"{folder.name} -- interface gap x_rel (translations)")
    if panelled:
        curves, labels, styles, extra = unpack(panelled, folder / STYLE_FILE)
        extra["note"] = plotting.shared_note([c["config"] for c in curves])
        plotting.nfrc_grid(
            curves, labels, styles, xlim=XLIM, **extra,
            title=f"{folder.name} -- output channel, per panel")
        plotting.gap_grid(
            curves, labels, styles, xlim=XLIM, **extra,
            title=f"{folder.name} -- interface gap x_rel, per panel")


for folder in folders:
    draw(folder)
plt.show()
