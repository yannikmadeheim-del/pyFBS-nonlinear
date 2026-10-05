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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thesis_plot_style as thesis_style
THESIS_FONT = thesis_style.configure()

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import plotting

COMPARE_DIRS = ["."]  # folder name(s), or paths relative to results/
FRF_FOLDER = "pyhbm_CB_RBE23_pyFBS_nonlinear_verification"
# Folders drawn as one plain FRF plot: output channel over excitation, one axes.
SIMPLE_FRF_FOLDERS = {"pyhbm_CB_RBE23_pyFBS_nonlinear_verification"}
# Folders drawn as the thesis FRF figure with its panels stacked top to bottom.
STACKED_FRF_FOLDERS = {"pyhbm_pyFBS_verification_sensorCOLL_nonlinear"}
SINGLE_OUTPUT_STUDIES = {
    "cubic_study_18.09.26",
    "collocated_sensors_cubic_STUDY_highResolution",
    "collocated_sensors_forcingAmp_STUDY_highResolution",
}
RESOLUTION_STUDY_FOLDER = "non_collocated_resolution_study"
THESIS_FOLDERS = {"pyhbm_CB_RBE23_pyFBS_nonlinear_verification", FRF_FOLDER,
                  *STACKED_FRF_FOLDERS, *SINGLE_OUTPUT_STUDIES,
                  RESOLUTION_STUDY_FOLDER}

STYLE_FILE   = "plot_styles.csv"     # style spec inside each of them
XLIM         = (1, 500)              # displayed frequency range [Hz]
SHOW_LEGEND  = False                  # False: omit legends (e.g. add one in LaTeX)
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8
PANEL_TITLE_SIZE = 10

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
folders = [RESULTS / name for name in (sys.argv[1:] or COMPARE_DIRS)]
cache = {}          # a file two rows share -- one folder or two -- is read once


def thesis_layout(fig, quantity, frf=False, label_by_grid=False):
    """Print layout for the DoE comparison; keep parameters in the thesis caption."""
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator

    fig.set_size_inches(6.3, 4.8, forward=True)  # 16 cm thesis text width
    fig.set_dpi(150)
    fig.set_layout_engine(None)
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    handles = {}
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    for legend in list(fig.legends):
        legend.remove()
    legend = None
    if SHOW_LEGEND:
        legend = fig.legend(handles.values(), handles.keys(), loc="upper center",
                            bbox_to_anchor=(0.5, 1.0), ncol=1, frameon=False,
                            fontsize=8, handlelength=2.8, columnspacing=1.3)
        legend.set_in_layout(False)  # reserve its measured height only once
    ylabel = (r"$\max_t\|\mathbf{q}^{\mathrm{rel}}_{\mathrm{trans}}(t)\|_2$ [m]"
              if quantity == "gap" else r"$\max_t|q_{\mathrm{out}}(t)|$ [m]")
    if frf:
        ylabel = (r"$\|\mathbf{Y}^{\mathrm{rel}}_{\mathrm{trans},e}(\omega)\|_2$ [m/N]"
                  if quantity == "gap" else r"$|Y_{\mathrm{out},e}(\omega)|$ [m/N]")
    for i, ax in enumerate(fig.axes):
        if label_by_grid:   # any grid shape: left column and bottom row
            spec = ax.get_subplotspec()
            first_col, last_row = spec.is_first_col(), spec.is_last_row()
        else:               # the 2x2 DoE grid
            first_col, last_row = i % 2 == 0, i >= 2
        ax.set_ylabel(ylabel if first_col else "", fontsize=AXIS_LABEL_SIZE)
        ax.set_xlabel(r"$f=\omega/(2\pi)$ [Hz]" if last_row else "", fontsize=AXIS_LABEL_SIZE)
        ax.set_title(ax.get_title(), fontsize=PANEL_TITLE_SIZE, pad=6)
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE, length=3)
        ax.tick_params(axis="both", which="minor", length=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(False, which="both")
        ax.grid(True, which="major", color="0.88", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    for text in fig.findobj(Text):
        text.set_fontfamily(THESIS_FONT)
    def fit_layout(event=None):
        top = 1.0
        if legend is not None:
            bounds = legend.get_window_extent(fig.canvas.get_renderer())
            top = fig.transFigure.inverted().transform((bounds.x0, bounds.y0))[1]
            top -= 2.0 / (72.0 * fig.get_figheight())  # 2 pt below legend
        fig.tight_layout(rect=(0, 0, 1, top), pad=0.7, h_pad=1.0, w_pad=0.8)

    fig.canvas.draw()
    fit_layout()
    fig.canvas.mpl_connect("resize_event", fit_layout)
    # The toolbar calls Figure.savefig. Always export at thesis width, even
    # when the interactive window has been maximized, then restore that view.
    original_savefig = fig.savefig

    def save_at_thesis_size(*args, **kwargs):
        screen_size = fig.get_size_inches().copy()
        try:
            fig.set_size_inches(6.3, 4.8, forward=False)
            fit_layout()
            kwargs.setdefault("dpi", 300)
            return original_savefig(*args, **kwargs)
        finally:
            fig.set_size_inches(screen_size, forward=False)
            fit_layout()
            fig.canvas.draw_idle()

    fig.savefig = save_at_thesis_size
    fig.canvas.manager.set_window_title(
        "testbench_doe_linear_" + quantity + ("_frf" if frf else ""))
    return fig


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


def simple_frf(folder):
    """One figure, one axes: |a_1(u_out)| / F0 of every spec row, overlaid.

    For a nonlinear joint this is the first-harmonic receptance of the forced
    response, which reduces to the linear FRF at alpha = 0.
    """
    print(f"\n{folder.name}:")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for path, label, style, _, _ in plotting.read_style_spec(folder, STYLE_FILE):
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            cache[path] = plotting.read_result(path)
        curve = cache[path]
        ax.semilogy(curve["f"], curve["out_h1"] / curve["config"]["F0"],
                    label=label, **style)
    ax.set_xlim(*XLIM)
    ax.set_xlabel(r"$f$ [Hz]")
    ax.set_ylabel(r"$|Y_{\mathrm{out},e}(\omega)|$ [m/N]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small", frameon=False)


def stacked_frf(folder):
    """Thesis-style FRF from the excitation to the output channel: one panel per
    spec ``panel`` value, stacked top to bottom, without the gap figure."""
    print(f"\n{folder.name}:")
    rows = []
    for path, label, style, panel, scale in plotting.read_style_spec(folder, STYLE_FILE):
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            cache[path] = plotting.read_result(path)
        rows.append((cache[path], label, style, panel, scale))
    curves, labels, styles, extra = unpack(rows, folder / STYLE_FILE)
    fig = plotting.curve_grid(curves, labels, styles, key="out_h1", title="",
                              ylabel="Receptance [m/N]", xlim=XLIM, ncols=1, **extra)
    thesis_layout(fig, "output", frf=True, label_by_grid=True)


def single_output_study(folder):
    """Main and first-resonance zoom overlays of max_t |q_out(t)| / f_e,amp."""
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator

    print(f"\n{folder.name}:")
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    zoom_fig, zoom_ax = plt.subplots(figsize=(6.3, 4.8))
    plotted = []
    for path, label, style, panel, scale in plotting.read_style_spec(folder,
                                                                     STYLE_FILE):
        if panel:
            raise ValueError(f"{folder / STYLE_FILE}: single-output studies "
                             "must leave the panel column empty")
        if scale is None or scale <= 0:
            raise ValueError(f"{path.name}: normalize must contain a positive F0")
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            table = pd.read_csv(path, comment="#",
                                usecols=["freq_hz", "uout_time_max_m"])
            cache[path] = {
                "f": table["freq_hz"].to_numpy(),
                "out_max": table["uout_time_max_m"].to_numpy(),
            }
        curve = cache[path]
        print(f"  {len(curve['f'])} points, "
              f"{curve['f'].min():.3f}..{curve['f'].max():.1f} Hz  ->  {label}")
        ordinate = curve["out_max"] / scale
        plotted.append((curve["f"], ordinate))
        ax.semilogy(curve["f"], ordinate, label=label, **style)
        zoom_ax.semilogy(curve["f"], ordinate, label=label, **style)

    main_xlim = (1, 400)
    zoom_xlim = ((94, 110) if "cubic" in folder.name.lower()
                 else (90, 125))

    def limits_in_window(xlim):
        visible = [y[(x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y) & (y > 0)]
                   for x, y in plotted]
        values = np.concatenate([part for part in visible if part.size])
        low, high = np.log10(values.min()), np.log10(values.max())
        pad = max(0.08 * (high - low), 0.08)
        return 10 ** (low - pad), 10 ** (high + pad)

    main_ylim = limits_in_window(main_xlim)
    zoom_ylim = limits_in_window(zoom_xlim)

    def finish_figure(current_fig, current_ax, xlim, ylim, suffix):
        current_ax.set_xlim(*xlim)
        current_ax.set_ylim(*ylim)
        current_ax.set_xlabel(r"$f=\omega/(2\pi)$ [Hz]",
                              fontsize=AXIS_LABEL_SIZE)
        current_ax.set_ylabel(
            thesis_style.DISPLACEMENT_LABEL,
            fontsize=AXIS_LABEL_SIZE)
        current_ax.tick_params(axis="both", which="major",
                               labelsize=TICK_LABEL_SIZE, length=3)
        current_ax.tick_params(axis="both", which="minor", length=1.5)
        current_ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        current_ax.grid(False, which="both")
        current_ax.grid(True, which="major", color="0.88", linewidth=0.5)
        for spine in current_ax.spines.values():
            spine.set_linewidth(0.6)
        for item in current_fig.findobj(Text):
            item.set_fontfamily(THESIS_FONT)

        def fit_layout(event=None):
            current_fig.tight_layout(pad=0.7)

        current_fig.canvas.draw()
        fit_layout()
        current_fig.canvas.mpl_connect("resize_event", fit_layout)
        original_savefig = current_fig.savefig

        def save_at_thesis_size(*args, **kwargs):
            screen_size = current_fig.get_size_inches().copy()
            try:
                current_fig.set_size_inches(6.3, 4.8, forward=False)
                fit_layout()
                kwargs.setdefault("dpi", 300)
                return original_savefig(*args, **kwargs)
            finally:
                current_fig.set_size_inches(screen_size, forward=False)
                fit_layout()
                current_fig.canvas.draw_idle()

        current_fig.savefig = save_at_thesis_size
        current_fig.canvas.manager.set_window_title(folder.name + suffix)

    finish_figure(fig, ax, main_xlim, main_ylim, "_output")
    finish_figure(zoom_fig, zoom_ax, zoom_xlim, zoom_ylim, "_output_zoom")
    ax.add_patch(Rectangle((zoom_xlim[0], zoom_ylim[0]),
                           zoom_xlim[1] - zoom_xlim[0],
                           zoom_ylim[1] - zoom_ylim[0],
                           fill=False, edgecolor="0.25", linewidth=0.6,
                           zorder=100))
    fig.canvas.draw_idle()
    return fig, zoom_fig


def resolution_study(folder):
    """Two compact thesis plots, one per alpha, for the FRF-grid study."""
    import numpy as np
    import pandas as pd
    from matplotlib.text import Text
    from matplotlib.ticker import MaxNLocator

    print(f"\n{folder.name}:")
    grouped = {}
    panel_order = []
    for path, label, style, panel, scale in plotting.read_style_spec(folder,
                                                                     STYLE_FILE):
        if not panel:
            raise ValueError(f"{folder / STYLE_FILE}: every resolution-study "
                             "row must name its alpha panel")
        if scale is None or scale <= 0:
            raise ValueError(f"{path.name}: normalize must contain a positive F0")
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            table = pd.read_csv(path, comment="#",
                                usecols=["freq_hz", "uout_time_max_m"])
            cache[path] = {
                "f": table["freq_hz"].to_numpy(),
                "out_max": table["uout_time_max_m"].to_numpy(),
            }
        curve = cache[path]
        print(f"  {len(curve['f'])} points, "
              f"{curve['f'].min():.3f}..{curve['f'].max():.1f} Hz  ->  "
              f"{panel}: {label}")
        if panel not in grouped:
            grouped[panel] = []
            panel_order.append(panel)
        grouped[panel].append((curve, label, style, scale))

    if len(panel_order) != 2:
        raise ValueError(f"{folder / STYLE_FILE}: expected two alpha panels, "
                         f"found {panel_order}")

    xlim = (1, 400)
    visible = []
    for rows in grouped.values():
        for curve, _, _, scale in rows:
            x = curve["f"]
            y = curve["out_max"] / scale
            part = y[(x >= xlim[0]) & (x <= xlim[1])
                     & np.isfinite(y) & (y > 0)]
            if part.size:
                visible.append(part)
    values = np.concatenate(visible)
    low, high = np.log10(values.min()), np.log10(values.max())
    pad = max(0.12 * (high - low), 0.10)
    ylim = (10 ** (low - pad), 10 ** (high + pad))

    figures = []
    for panel in panel_order:
        fig, ax = plt.subplots(figsize=(6.3, 2.65))
        for curve, label, style, scale in grouped[panel]:
            ax.semilogy(curve["f"], curve["out_max"] / scale,
                        label=label, **style)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel(thesis_style.FREQUENCY_LABEL, fontsize=10)
        ax.set_ylabel(thesis_style.DISPLACEMENT_LABEL, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8, length=3)
        ax.tick_params(axis="both", which="minor", length=1.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(False, which="both")
        ax.grid(True, which="major", color="0.88", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        for item in fig.findobj(Text):
            item.set_fontfamily(THESIS_FONT)

        def fit_layout(event=None, current_fig=fig):
            current_fig.tight_layout(pad=0.55)

        fig.canvas.draw()
        fit_layout()
        fig.canvas.mpl_connect("resize_event", fit_layout)
        original_savefig = fig.savefig

        def save_at_thesis_size(*args, _fig=fig, _save=original_savefig,
                                _fit=fit_layout, **kwargs):
            screen_size = _fig.get_size_inches().copy()
            try:
                _fig.set_size_inches(6.3, 2.65, forward=False)
                _fit()
                kwargs.setdefault("dpi", 300)
                return _save(*args, **kwargs)
            finally:
                _fig.set_size_inches(screen_size, forward=False)
                _fit()
                _fig.canvas.draw_idle()

        fig.savefig = save_at_thesis_size
        fig.canvas.manager.set_window_title(folder.name + "_" + panel)
        figures.append(fig)
    return tuple(figures)


def draw(folder):
    """Every figure of one comparison folder: one pair per layout its spec uses."""
    if folder.name == RESOLUTION_STUDY_FOLDER:
        resolution_study(folder)
        return
    if folder.name in SINGLE_OUTPUT_STUDIES:
        single_output_study(folder)
        return
    if folder.name in SIMPLE_FRF_FOLDERS:
        simple_frf(folder)
        return
    if folder.name in STACKED_FRF_FOLDERS:
        stacked_frf(folder)
        return
    existing_figures = set(plt.get_fignums())
    print(f"\n{folder.name}:")
    overlaid, panelled = [], []
    spec = plotting.read_style_spec(folder, STYLE_FILE)
    for path, label, style, panel, scale in spec:
        if path not in cache:
            print(f"loading {path.name} ({path.stat().st_size / 1e6:.1f} MB) ...")
            cache[path] = plotting.read_result(path)
        curve = cache[path]
        if folder.name == FRF_FOLDER:
            config = curve["config"] or {}
            if 1 not in config.get("harmonics", []):
                raise ValueError(f"{path.name}: FRF requires the first harmonic")
            if scale is None or scale <= 0 or scale != config.get("F0"):
                raise ValueError(f"{path.name}: FRF normalization must be positive F0")
            if not config.get("joints") or any(
                joint.get("type") != "linear" and not
                (joint.get("type") == "cubic" and joint.get("alpha") == 0)
                for joint in config["joints"]
            ):
                raise ValueError(f"{path.name}: this FRF plot requires a linear joint")
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
        thesis = folder.name in THESIS_FOLDERS
        frf = folder.name == FRF_FOLDER
        extra["note"] = None if thesis else plotting.shared_note([c["config"] for c in curves])
        if frf:
            # CSV harmonics are physical complex amplitudes: a_1/F0 is
            # displacement/force receptance, with no factor-of-two conversion.
            output_fig = plotting.curve_grid(
                curves, labels, styles, key="out_h1", title="",
                ylabel="Receptance [m/N]", xlim=XLIM, **extra)
            gap_fig = plotting.curve_grid(
                curves, labels, styles, key="gap_h1", title="",
                ylabel="Relative receptance [m/N]", xlim=XLIM, **extra)
        else:
            output_fig = plotting.nfrc_grid(
                curves, labels, styles, xlim=XLIM, **extra,
                title=f"{folder.name} -- output channel, per panel")
            gap_fig = plotting.gap_grid(
                curves, labels, styles, xlim=XLIM, **extra,
                title=f"{folder.name} -- interface gap x_rel, per panel")
        if thesis:
            thesis_layout(output_fig, "output", frf=frf)
            thesis_layout(gap_fig, "gap", frf=frf)

    if not SHOW_LEGEND:
        for number in set(plt.get_fignums()) - existing_figures:
            fig = plt.figure(number)
            for legend in list(fig.legends):
                legend.remove()
            for ax in fig.axes:
                if ax.get_legend() is not None:
                    ax.get_legend().remove()


if __name__ == "__main__":
    for folder in folders:
        draw(folder)
    plt.show()
