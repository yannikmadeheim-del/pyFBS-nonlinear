"""Select what to plot and which result files to plot it from -- no solver.

Two selections, both at the top of this file:

  * COMPARE_DIRS -- folder name(s) inside results/. ``"."`` is results/ itself,
    where a plain ``main.py`` run writes; a study writes to results/<STUDY_NAME>/.
    Naming several draws them all, each with its own plot_styles.csv.
  * PLOT_METHODS_TO_DRAW -- which registered plot methods to draw per folder.

The plot methods themselves live in ``plotting_saving.PLOT_METHODS``. Currently
registered:

    rel_disp_forces  max_t |x_rel(t)| over the sweep, plus the normal force N(t)
                     and the friction force f_T(t) over one period at FORCE_OMEGA
                     as subplots  [default]
    rel_disp_max     max_t |x_rel(t)| alone, one row per interface DOF
    rel_disp_h1      first-harmonic amplitude of the same
    tip_response     axial tip response q1 and q4

Adding one takes no change here: write a function taking
``(curves, labels, styles, title, xlim, force_omega)`` in plotting_saving.py,
decorate it with ``@plot_method("name")``, and name it in PLOT_METHODS_TO_DRAW.

On the first run a plot_styles.csv is written into the folder, labelled from
whatever actually differs between the runs; edit its rows afterwards to control
legend text, colour, line style, line width and -- through the row order -- the
plot and legend order. CSVs the spec does not list are appended with a note.

The settings below are the DEFAULTS. A folder may override which figures it
wants and at which frequency in its own plot_styles.csv, through the ``#`` lines
above the header, so a comparison keeps its settings next to its curves::

    # methods: rel_disp_forces, tip_response
    # force_omega: 0.9985
    # xlim: 0.85, 1.09

Folders and methods can also be given on the command line::

    python plot_comparison.py solver_verification
    python plot_comparison.py contact_regularization --methods rel_disp_max,tip_response
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import plotting_saving as io

COMPARE_DIRS = ["DLFT"]      # folder name(s) inside results/
PLOT_METHODS_TO_DRAW = ["rel_disp_forces"]  # names from plotting_saving.PLOT_METHODS
STYLE_FILE = "plot_styles.csv"              # style spec inside each folder
XLIM = (0.85, 1.09)                         # displayed omega range [rad/s]
FORCE_OMEGA = None                          # force panels at this omega;
                                            # None -> the DLFT branch's resonance


def load(folder):
    """The folder's style spec, resolved into (curves, labels, styles)."""
    curves, labels, styles = [], [], []
    for path, label, style in io.read_style_spec(folder, STYLE_FILE):
        curve = io.read_result(path)
        print(f"  {len(curve['omega'])} points, "
              f"omega {curve['omega'].min():.4f}..{curve['omega'].max():.4f}"
              f"  ->  {label}")
        curves.append(curve); labels.append(label); styles.append(style)
    return curves, labels, styles


def draw(folder, methods=None):
    """Every selected plot method for one comparison folder.

    Precedence: the command line beats the folder's own plot_styles.csv, which
    beats the defaults at the top of this file.
    """
    folder = Path(folder).resolve()
    print(f"\n{folder.name}:")
    curves, labels, styles = load(folder)          # writes the style spec if absent
    settings = io.read_plot_settings(folder, STYLE_FILE)
    names = methods or settings.get("methods") or PLOT_METHODS_TO_DRAW
    for name in names:
        if name not in io.PLOT_METHODS:
            raise KeyError(f"unknown plot method {name!r}; registered: "
                           f"{', '.join(sorted(io.PLOT_METHODS))}")
        io.PLOT_METHODS[name](curves, labels, styles, title=folder.name,
                              xlim=settings.get("xlim", XLIM),
                              force_omega=settings.get("force_omega", FORCE_OMEGA))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="*", default=None,
                        help="folder name(s) inside results/ (default: COMPARE_DIRS)")
    parser.add_argument("--methods", default=None,
                        help="comma-separated plot method names (default: the "
                             "folder's plot_styles.csv, else PLOT_METHODS_TO_DRAW)")
    args = parser.parse_args()

    names = args.folders or COMPARE_DIRS
    methods = [s.strip() for s in args.methods.split(",")] if args.methods else None
    for name in names:
        draw(io.RESULTS / name, methods)
    plt.show()
