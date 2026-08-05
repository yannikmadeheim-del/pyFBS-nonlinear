"""
Reading and plotting of physical-solution CSVs -- shared by main.py and
plot_comparison.py.

Three layers:

  * :func:`read_result` -- one CSV into the four plotted curves plus the CONFIG
    that produced it. Only the needed columns are read (these files get large),
    and the harmonic list comes from the file itself, never from a constant here.

  * :func:`auto_labels` -- legend labels derived from the config entries that
    actually DIFFER across the loaded files, e.g. ``linear+cubic | alpha=1e+08``.

  * :func:`read_style_spec` -- the comparison-folder infrastructure: a folder of
    CSVs plus a ``plot_styles.csv`` whose ROW ORDER is the plot and legend order
    and whose empty cells fall back to auto label / cycled style. The spec is
    written on first use, so a fresh folder works immediately and can be edited
    afterwards, and its ``normalize`` column divides each curve by its own
    excitation amplitude. Several such folders coexist side by side.

Two figure kinds sit on top, and the spec picks between them: :func:`overlay`
(with its ``nfrc_figure`` / ``gap_figure`` wrappers) stacks two quantities of one
curve set in one pair of axes, while :func:`curve_grid` (``nfrc_grid`` /
``gap_grid``) spreads the curves over one panel per value of the spec's ``panel``
column and shows one quantity per figure. A folder whose curves differ in TWO
parameters -- the resolution studies, where every ``alpha`` is swept over every
``f_resolution`` -- names one of them in the panel column and leaves the other to
the colours; rows that leave the column empty keep the single-axes overlay, and a
spec may use both by listing a file twice, once for each layout.
"""

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_T = 1024                          # common dense time grid for the period maxima
TRANS = ("ux", "uy", "uz")          # translational gap DoFs
STYLE_COLUMNS = ("file", "label", "color", "linestyle", "linewidth", "panel",
                 "normalize")
SKIP_CSV = ("manifest.csv",)        # study bookkeeping, not a result branch

DEFAULT_COLORS = ["#2ca02c", "black", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e",
                  "#17becf", "#8c564b", "#e377c2", "#7f7f7f"]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_LINEWIDTH = 1.4


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_header(path):
    """The leading ``#`` comment lines of a result CSV, without the marker."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line[1:].strip())
    return lines


def read_config(path):
    """The CONFIG dict recorded in the ``config_json`` header line, or None for
    files written without one (e.g. the older reference exports)."""
    for line in read_header(path):
        if line.startswith("config_json:"):
            return json.loads(line[len("config_json:"):].strip())
    return None


def _harmonics(config, columns):
    """Exported harmonics: from the file's own config, else from the column
    names. Hard-coding a list here would silently misread any run with a
    different harmonic set."""
    if config and "harmonics" in config:
        return [int(h) for h in config["harmonics"]]
    found = {int(m.group(1)) for m in
             (re.match(r"re_h(\d+)_", c) for c in columns) if m}
    if not found:
        raise KeyError(f"no re_h<h>_<dof> columns found -- not a "
                       f"physical-solution CSV?")
    return sorted(found)


def _output_channel(header, columns, harmonics):
    """Plotted output channel, taken from the header line the exporters write."""
    match = re.search(r"uout \(plotted output channel\) = (\S+)",
                      "\n".join(header))
    for name in ([match.group(1)] if match else []) + ["A_S1X"]:
        if f"re_h{harmonics[0]}_{name}" in columns:
            return name
    raise KeyError("output-channel columns not found; the header names "
                   f"{match.group(1) if match else 'nothing'}")


def _amplitudes(df, dof, harmonics):
    """Complex physical amplitudes a_h of ``dof``, shape (n_points, n_harmonics)."""
    re_ = df[[f"re_h{h}_{dof}" for h in harmonics]].to_numpy()
    im_ = df[[f"im_h{h}_{dof}" for h in harmonics]].to_numpy()
    return re_ + 1j * im_


def _period_signal(amps, harmonics):
    """u(theta) = Re( sum_h a_h e^{i h theta} ) on N_T samples of one period.
    Matmul contracts the trailing harmonic axis: (..., n_h) -> (..., N_T).
    Recomputing on a common grid makes files with different AFT sample counts
    comparable."""
    theta = np.linspace(0.0, 2.0 * np.pi, N_T, endpoint=False)
    return np.real(amps @ np.exp(1j * np.outer(harmonics, theta)))


def read_result(path):
    """
    One physical-solution CSV into the plotted curves.

    :returns: dict with ``f`` [Hz], ``out_max`` = max_t |u_out(t)|, ``out_h1`` =
        |a_1(u_out)|, ``gap_max`` = max_t ||x_rel(t)||_2 and ``gap_h1`` =
        ||a_1(x_rel)||_2 over the three translational gap DoFs, plus ``config``.
    """
    path = Path(path)
    config = read_config(path)
    header = read_header(path)
    columns = list(pd.read_csv(path, comment="#", nrows=0).columns)  # names only
    harmonics = _harmonics(config, columns)
    h0 = harmonics[0]

    out = _output_channel(header, columns, harmonics)
    if f"re_h{h0}_x_rel_{TRANS[0]}" in columns:
        gap_dofs = [f"x_rel_{d}" for d in TRANS]        # gap exported directly
    elif f"re_h{h0}_A_vp_{TRANS[0]}" in columns:
        gap_dofs = [f"{s}_vp_{d}" for s in ("A", "B") for d in TRANS]
    else:
        raise KeyError(f"{path.name}: neither x_rel_* nor A_vp_*/B_vp_* harmonic "
                       f"columns found -- not a physical-solution CSV?")

    wanted = {"freq_hz"} | {f"{p}_h{h}_{d}" for d in [out] + gap_dofs
                            for h in harmonics for p in ("re", "im")}
    df = pd.read_csv(path, comment="#", usecols=lambda c: c in wanted)
    missing = wanted - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} lacks expected columns, e.g. "
                       f"{sorted(missing)[:3]}")

    a_out = _amplitudes(df, out, harmonics)                          # (n, n_h)
    if gap_dofs[0].startswith("x_rel"):
        a_gap = np.stack([_amplitudes(df, f"x_rel_{d}", harmonics) for d in TRANS],
                         axis=1)
    else:                                   # pyhbm style: gap = VP_A - VP_B
        a_gap = np.stack([_amplitudes(df, f"A_vp_{d}", harmonics)
                          - _amplitudes(df, f"B_vp_{d}", harmonics) for d in TRANS],
                         axis=1)                                     # (n, 3, n_h)

    sig_out = _period_signal(a_out, harmonics)                       # (n, N_T)
    sig_gap = _period_signal(a_gap, harmonics)                       # (n, 3, N_T)
    i_h1 = harmonics.index(1) if 1 in harmonics else 0
    return dict(
        f       = df["freq_hz"].to_numpy(),
        out_max = np.abs(sig_out).max(axis=1),
        out_h1  = np.abs(a_out[:, i_h1]),
        gap_max = np.linalg.norm(sig_gap, axis=1).max(axis=1),
        gap_h1  = np.linalg.norm(a_gap[:, :, i_h1], axis=1),
        config  = config,
    )


# ---------------------------------------------------------------------------
# Automatic legend labels
# ---------------------------------------------------------------------------

def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _flatten(config):
    """Config as flat ``key -> string`` pairs. Joint fields become dotted keys
    (``linear.k``); a repeated joint type gets a running number (``linear2.k``)
    so two springs of the same type stay distinguishable."""
    flat, seen = {}, {}
    for spec in config.get("joints", []):
        kind = spec.get("type", "?")
        n = seen.get(kind, 0)
        seen[kind] = n + 1
        prefix = kind if n == 0 else f"{kind}{n + 1}"
        for key, value in spec.items():
            if key != "type":
                flat[f"{prefix}.{key}"] = _fmt(value)
    flat["joints"] = "+".join(dict.fromkeys(spec.get("type", "?")
                                            for spec in config.get("joints", [])))
    for key, value in config.items():
        if key != "joints":
            flat[key] = _fmt(value)
    return flat


def auto_labels(configs, fallback_names):
    """
    Legend labels built ONLY from the config entries that differ across the
    loaded files, e.g. ``linear+cubic | alpha=1e+08 | F0=200``. Falls back to
    ``fallback_names`` (normally the file stems) when a config is missing or all
    configs are identical.

    Top-level config keys are compared across ALL files, with ``"-"`` standing
    in where a file has no such key: solvers with different schemas (an FBS run
    has ``frf_source``, a Craig-Bampton run has ``condensation``) are then still
    told apart instead of falling back to the file names. Joint element keys --
    the dotted ones, ``linear.k``, ``cubic2.alpha`` -- keep the all-present rule:
    a missing joint field means the element itself is absent, which the leading
    joint-type slug already says.
    """
    if any(c is None for c in configs) or len(configs) < 2:
        return list(fallback_names)

    flats = [_flatten(c) for c in configs]
    keys = [k for k in dict.fromkeys(k for flat in flats for k in flat)
            if "." not in k or all(k in flat for flat in flats)]
    values = [{k: flat.get(k, "-") for k in keys} for flat in flats]
    differing = [k for k in keys if len({v[k] for v in values}) > 1]
    if not differing:
        return list(fallback_names)
    differing.sort(key=lambda k: (k != "joints", keys.index(k)))

    # show the short field name where it is unambiguous ("alpha", not "cubic.alpha")
    tails = [k.rsplit(".", 1)[-1] for k in differing]
    display = {k: (t if tails.count(t) == 1 else k)
               for k, t in zip(differing, tails)}

    labels = []
    for vals in values:
        parts = [vals[k] if k == "joints" else f"{display[k]}={vals[k]}"
                 for k in differing]
        labels.append(" | ".join(parts))
    return labels


# ---------------------------------------------------------------------------
# Shared-parameter note
# ---------------------------------------------------------------------------

# The counterpart of auto_labels: that one names what DIFFERS across the loaded
# files, this names what they all share. Only the physical entries are listed --
# tolerances, step-length settings and the predictor choice are solver
# bookkeeping and would bury the parameters a reader of the figure needs to
# interpret the curves.
NOTE_FIELDS = (
    ("joints",             "{} joint"),
    ("linear.k",           "k = {} N/m"),
    ("linear.c",           "c = {} Ns/m"),
    ("linear2.k",          "k_rot = {} Nm/rad"),
    ("linear2.c",          "c_rot = {} Nms/rad"),
    ("cubic.alpha",        "alpha = {} N/m^3"),
    ("friction.mu",        "mu = {}"),
    ("friction.N",         "N = {} N"),
    ("friction.alpha_reg", "alpha_reg = {} s/m"),
    ("F0",                 "F0 = {} N"),
    ("modal_damping",      "zeta = {}"),
    ("frf_source",         "{} FRF"),
    ("harmonics",          "harmonics {}"),
)


def _time_samples(config):
    """AFT time samples per period of one run. pyFBS stores the number itself;
    pyhbm stores the polynomial degree its sampling is built from, which its own
    exports expand as N_t = (degree + 1) * max(harmonics) + 1."""
    if "sample_number" in config:
        return int(config["sample_number"])
    if "polynomial_degree" in config:
        return ((int(config["polynomial_degree"]) + 1)
                * max(int(h) for h in config["harmonics"]) + 1)
    return None


def _time_sample_field(configs):
    """``N_t = 512 (pyfbs-nlfbs), 562 (pyhbm-cb)`` -- the one note entry that is
    NOT a shared value: the two pipelines parameterize the AFT grid differently,
    so a folder comparing them has no common sample count to state, and for a
    regularized friction law the count is exactly what a reader needs to judge
    whether the stick-slip transition is resolved. Empty when a file records
    neither field, or when one solver appears with two different counts -- the
    entry would then be a per-curve property and belongs in the legend.
    """
    per_solver = {}
    for config in configs:
        n_t = _time_samples(config)
        if n_t is None:
            return ""
        per_solver.setdefault(str(config.get("solver", "?")), set()).add(n_t)
    if any(len(counts) > 1 for counts in per_solver.values()):
        return ""
    distinct = {next(iter(counts)) for counts in per_solver.values()}
    if len(distinct) == 1:
        return f"N_t = {distinct.pop()}"
    return "N_t = " + ", ".join(f"{next(iter(counts))} ({solver})"
                                for solver, counts in per_solver.items())


def shared_note(configs):
    """
    One line naming the config entries every loaded file agrees on, e.g.
    ``shared: linear+cubic joint | k = 1e+06 N/m | ... | harmonics 1,3,5,7``.
    Fields absent from one of the files are left out, so a folder that mixes
    solvers with different schemas keeps only what is genuinely common.

    Returns "" when a file has no ``config_json`` header: the values the others
    share would then be asserted for a run whose settings are unknown.
    """
    if not configs or any(c is None for c in configs):
        return ""
    flats = [_flatten(c) for c in configs]
    shared = {k: v for k, v in flats[0].items()
              if all(f.get(k) == v for f in flats[1:])}
    parts = [text.format(shared[key]) for key, text in NOTE_FIELDS
             if key in shared]
    # only for a friction joint: there the time resolution of the regularized
    # stick-slip transition is a result-relevant setting, not bookkeeping
    if "friction" in shared.get("joints", ""):
        parts += [field for field in (_time_sample_field(configs),) if field]
    return "shared: " + " | ".join(parts) if parts else ""


def _draw_note(fig, note, width=180):
    """Write ``note`` small and grey along the bottom edge, wrapped between its
    ``|`` fields. Constrained layout does not budget for figure-level text, so
    the layout rectangle is shrunk by exactly the height the lines take --
    without that the note is drawn on top of the lower axes."""
    if not note:
        return
    lines, current = [], ""
    for part in note.split(" | "):
        candidate = f"{current} | {part}" if current else part
        if current and len(candidate) > width:
            lines.append(current)
            current = part
        else:
            current = candidate
    lines.append(current)
    fig.text(0.01, 0.008, "\n".join(lines), ha="left", va="bottom",
             fontsize=8, color="0.35")
    height = 0.018 * len(lines) + 0.012
    fig.get_layout_engine().set(rect=(0, height, 1, 1 - height))


# ---------------------------------------------------------------------------
# Comparison-folder style spec
# ---------------------------------------------------------------------------

def _linestyle(text):
    """
    One ``linestyle`` cell of a style spec.

    Either a matplotlib style string (``-``, ``--``, ``-.``, ``:``) or an on/off
    dash sequence in points, ``6,2`` or ``6,2,1.5,2``. The sequence form exists
    because matplotlib names only four styles: a folder that puts more than four
    curves into one panel has to give each of them a dash pattern of its own,
    which is the colour-independent second identity that keeps the curves apart
    for a colour-blind reader and in a black-and-white print.
    """
    if "," not in text:
        return text
    return (0, tuple(float(value) for value in text.split(",")))


def _scale(text, config, name):
    """
    One ``normalize`` cell: the excitation amplitude the curve is divided by.

    Either the name of a config key holding it (``F0``) -- so that every run
    brings its own amplitude and the spec does not repeat it -- or, for files
    exported without a ``config_json`` header, the force in newtons written out.
    """
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    if config is None or text not in config:
        raise KeyError(f"{name}: normalize='{text}' is neither a number nor a "
                       f"config key of that file")
    value = config[text]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name}: config key '{text}' is {value!r}, not a number")
    return float(value)


def _write_style_spec(spec_path, names, auto):
    with open(spec_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STYLE_COLUMNS)
        for i, name in enumerate(names):
            writer.writerow([name, auto[name],
                             DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                             DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                             DEFAULT_LINEWIDTH])


def read_style_spec(folder, style_file="plot_styles.csv"):
    """
    Resolve one comparison folder into the curves to plot, in plot order.

    ``<folder>/<style_file>`` has the columns
    file,label,color,linestyle,linewidth,panel,normalize and its row order is the
    plot and legend order; empty cells fall back to the auto label and a cycled
    default style. ``linestyle`` takes a matplotlib style string or a dash sequence, see
    :func:`_linestyle`. CSVs in the folder that the spec does not name are
    appended at the end (with a printed note), and a spec row naming a file that
    is not in the folder is an error. If the spec does not exist it is written
    from the folder contents first.

    ``panel`` names the subplot a curve belongs to and so decides its figure
    kind, row by row: rows that name one go into a :func:`curve_grid` with one
    panel per distinct value, rows that leave it empty share one pair of axes.
    A spec may use both, listing the same file twice -- once overlaid with its
    siblings, once beside its references in a panel of its own.

    ``normalize`` divides the curve by the excitation amplitude, which turns an
    amplitude sweep into the receptance-like |x| / F0 in m/N instead of curves
    that differ mostly by their force level. Left empty everywhere -- as in every
    spec written so far -- the curves are plotted raw.

    :returns: list of ``(path, label, style_kwargs, panel, scale)``, ``panel``
        empty and ``scale`` None where the row does not name one.
    """
    folder = Path(folder)
    spec_path = folder / style_file
    # Path.glob returns empty on a missing directory instead of raising, which
    # would surface below as a misleading "no result CSVs" message.
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{folder} is not a folder -- COMPARE_DIRS names comparison FOLDERS "
            f"inside results/, not CSV files")
    files = sorted(p for p in folder.glob("*.csv")
                   if p.name != style_file and p.name not in SKIP_CSV)
    if not files:
        raise FileNotFoundError(f"no result CSVs in {folder} (looked for *.csv "
                                f"besides {style_file})")

    configs = {p.name: read_config(p) for p in files}
    auto = dict(zip(configs.keys(),
                    auto_labels(list(configs.values()),
                                [p.stem for p in files])))

    if spec_path.exists():
        with open(spec_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        entries = []
        for line_no, row in enumerate(rows, start=2):      # 1 = header row
            name = (row.get("file") or "").strip()
            if not name:
                continue
            if not (folder / name).exists():
                raise FileNotFoundError(
                    f"{spec_path}, line {line_no}: '{name}' is not in {folder}")
            entries.append((name, row))
        listed = {name for name, _ in entries}
        extra = [p.name for p in files if p.name not in listed]
        if extra:
            print(f"note: not listed in {style_file}, appended at the end: "
                  f"{', '.join(extra)}")
        entries += [(name, {}) for name in extra]
    else:
        entries = [(p.name, {}) for p in files]
        _write_style_spec(spec_path, [name for name, _ in entries], auto)
        print(f"created {spec_path} from the folder contents -- edit it to "
              f"control order, labels and styles")

    out = []
    for i, (name, row) in enumerate(entries):
        width = (row.get("linewidth") or "").strip()
        out.append((
            folder / name,
            (row.get("label") or "").strip() or auto[name],
            dict(color=(row.get("color") or "").strip()
                       or DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                 linestyle=_linestyle((row.get("linestyle") or "").strip()
                                      or DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)]),
                 linewidth=float(width) if width else DEFAULT_LINEWIDTH),
            (row.get("panel") or "").strip(),
            _scale((row.get("normalize") or "").strip(), configs[name], name),
        ))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _scaled(curve, key, scale):
    """The plotted ordinate of one curve: raw, or divided by the excitation
    amplitude its spec row named."""
    return curve[key] if scale is None else curve[key] / scale


def _unit(scales):
    """Ordinate label pieces for a normalized and for a raw figure."""
    return (" / F0", "[m/N]") if scales is not None else ("", "[m]")


def curve_grid(curves, labels, styles, panels, key, title, ylabel, xlim=None,
               ncols=2, scales=None, note=None):
    """
    One semilogy panel per distinct entry of ``panels``, on a shared grid.

    Where :func:`overlay` stacks two quantities of ONE curve set in one pair of
    axes, this shows ONE quantity of several curve sets side by side: curve i
    goes into the panel named ``panels[i]``, panels appear in first-mention
    order, and they share both axes so the sets stay comparable across them.

    Splitting a folder up this way is what makes a two-parameter study readable:
    with the first parameter on the panels, only the second one is left for the
    colours, and a curve is identified by where it sits plus one colour instead
    of by a legend entry out of thirty. Labels therefore repeat from panel to
    panel, and one figure-level legend describes them all.

    :param curves: :func:`read_result` dicts, in plot order.
    :param labels: legend label per curve; equal labels share one legend entry.
    :param styles: matplotlib line kwargs per curve.
    :param panels: panel name per curve, used as the panel title.
    :param key: plotted curve key, e.g. ``"out_max"``.
    :param title: figure title above the panels.
    :param ylabel: y axis label, written on the left column.
    :param xlim: displayed frequency range, or None for the data range.
    :param ncols: panel columns; the number of rows follows from the panel count.
    :param scales: per-curve divisor, or None for raw curves.
    :param note: shared-parameter line along the bottom edge, or None.
    """
    if scales is None:
        scales = [None] * len(curves)
    names = list(dict.fromkeys(panels))
    nrows = -(-len(names) // ncols)                      # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    flat = axes.ravel()
    for curve, label, style, panel, scale in zip(curves, labels, styles, panels,
                                                 scales):
        flat[names.index(panel)].semilogy(curve["f"], _scaled(curve, key, scale),
                                          label=label, **style)
    for i, (ax, name) in enumerate(zip(flat, names)):
        ax.set_title(name)
        ax.grid(True, which="both", alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
        if i + ncols >= len(names):         # bottom-most USED panel of its column
            ax.set_xlabel("Frequency [Hz]")
            ax.tick_params(labelbottom=True)   # sharex hides these off the last row
    for ax in flat[len(names):]:            # a grid that does not divide evenly
        ax.set_axis_off()
    if xlim is not None:
        flat[0].set_xlim(*xlim)
    fig.suptitle(title)

    handles = {}
    for ax in flat[:len(names)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc="outside right upper",
               fontsize="small", frameon=False)
    _draw_note(fig, note)
    return fig


def overlay(curves, labels, styles, key_max, key_h1, title, ylabel_max,
            ylabel_h1, xlim=None, scales=None, note=None):
    """One window, 2x1 shared-x semilogy: period maximum on top, 1st-harmonic
    amplitude below; ``styles[i]`` are the matplotlib line kwargs of curve i and
    ``scales[i]``, where given, the amplitude it is divided by. ``note`` is the
    shared-parameter line written along the bottom edge."""
    if scales is None:
        scales = [None] * len(curves)
    # A legend with many entries is taller than the axes. Drawn inside them,
    # tight_layout honours it by shrinking the plots to slivers, so it goes
    # beside the axes instead and constrained_layout budgets the space. One
    # column keeps it a single top-to-bottom list in plot order, instead of
    # splitting the entries across columns.
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 8),
                                         sharex=True, layout="constrained")
    for curve, label, style, scale in zip(curves, labels, styles, scales):
        ax_top.semilogy(curve["f"], _scaled(curve, key_max, scale), label=label,
                        **style)
        ax_bot.semilogy(curve["f"], _scaled(curve, key_h1, scale), label=label,
                        **style)
    ax_top.set_title(title)
    ax_top.set_ylabel(ylabel_max)
    ax_top.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
                  fontsize="small", frameon=False)
    ax_bot.set_ylabel(ylabel_h1)
    ax_bot.set_xlabel("Frequency [Hz]")
    if xlim is not None:
        ax_bot.set_xlim(*xlim)
    for ax in (ax_top, ax_bot):
        ax.grid(True, which="both", alpha=0.3)
    _draw_note(fig, note)
    return fig


def nfrc_figure(curves, labels, styles, title="Output channel", xlim=None,
                scales=None, note=None):
    """Forced-response curves of the output channel."""
    per, unit = _unit(scales)
    return overlay(curves, labels, styles, "out_max", "out_h1", title,
                   f"max_t |u_out(t)|{per}  {unit}",
                   f"|a_1(u_out)|{per}  {unit}", xlim, scales, note)


def gap_figure(curves, labels, styles,
               title="Interface gap x_rel (translations)", xlim=None,
               scales=None, note=None):
    """Forced-response curves of the translational interface gap the joint acts on."""
    per, unit = _unit(scales)
    return overlay(curves, labels, styles, "gap_max", "gap_h1", title,
                   f"max_t ||x_rel(t)||_2{per}  {unit}",
                   f"||a_1(x_rel)||_2{per}  {unit}", xlim, scales, note)


def nfrc_grid(curves, labels, styles, panels, title="Output channel", xlim=None,
              ncols=2, scales=None, note=None):
    """Panel-per-``panels``-entry counterpart of :func:`nfrc_figure`. Only the
    period maximum is drawn: the panels already spend the second pair of axes."""
    per, unit = _unit(scales)
    return curve_grid(curves, labels, styles, panels, "out_max", title,
                      f"max_t |u_out(t)|{per}  {unit}", xlim, ncols, scales,
                      note)


def gap_grid(curves, labels, styles, panels,
             title="Interface gap x_rel (translations)", xlim=None, ncols=2,
             scales=None, note=None):
    """Panel-per-``panels``-entry counterpart of :func:`gap_figure`."""
    per, unit = _unit(scales)
    return curve_grid(curves, labels, styles, panels, "gap_max", title,
                      f"max_t ||x_rel(t)||_2{per}  {unit}", xlim, ncols, scales,
                      note)
