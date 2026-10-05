"""Saving, reading and plotting of two-rod results -- pyFBS and NLvib side by side.

Every result is one CSV inside a study folder ``Results/<study>/``. The pyFBS
runs are written by :func:`save_result` (called from ``main.py``), the NLvib
references by ``NLvib_verification/two_rod_unilateral_contact/
two_rod_nlvib_reference.m`` and copied into the study folder by hand. Running
this file overlays everything a study folder contains::

    python plotting.py dlft_verification
    python plotting.py dlft_verification --plots min_gap,frc

CSV contract (both writers follow it)
-------------------------------------
* Leading ``#`` lines describe the run; the LAST of them is
  ``# config_json: {...}`` with the full parameter set of the run.
* Columns ``omega_rad_s, freq_hz, iterations`` and then, harmonic-major, the
  complex amplitudes ``re_h<h>_<ch>, im_h<h>_<ch>`` of the channels

  - ``xr``    relative displacement of the rods, x_r = u_A + u_B,
  - ``fc``    contact force acting on x_r as the solver converged it, truncated
    to the retained harmonics, positive in compression,
  - ``tipA``, ``tipB``  the two tip displacements.

* Amplitudes are physical, ``u(t) = Re( sum_h A_h exp(1j h omega t) )``. pyFBS
  stores rFFT coefficients c_h, so A_0 = c_0/N_t and A_h = 2 c_h/N_t; NLvib's
  sine-cosine coefficients give A_0 = Q_0 and A_h = Qc_h - 1j Qs_h. Both are the
  same quantity.

The gap g(t) = g0 - x_r(t) follows from ``xr`` and the ``gap`` in the config:
g > 0 open, g = 0 closed, g < 0 interpenetration.

Plot style spec
---------------
Each study folder carries a ``plot_styles.csv``, written from the folder
contents on first use and edited by hand afterwards. Its row order is the plot
and legend order, and empty cells fall back to defaults. Columns::

    file, label, color, linestyle, linewidth, representation, marker,
    markersize, markevery

``representation`` is ``line``, ``markers`` or ``line+markers``; ``linestyle``
is a matplotlib style (``-``, ``--``, ``-.``, ``:``) or an on/off dash sequence
in points such as ``6,2``. A file may be listed twice with two styles. The
``#`` lines above the header hold the folder's own plot settings
(:data:`SETTINGS_KEYS`); a blank value keeps the default of this file::

    # plots: min_gap, frc, contact_law
    # xlim: 70000, 95000
    # omega_star: 80000
    # reference: pyfbs_dlft_aft.csv
    # harmonics_shown: 0, 1, 2, 3
    # gap_linthresh_um: 0.001

Plot methods are registered by name with :func:`plot_method`; a new one only
needs the decorator to become selectable.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================ selection =====================================
RESULT_DIR = "LB_REL=5"                 # study folder inside Results/ (or CLI argument)
PLOTS = ["min_gap", "frc", "time_signals", "contact_law", "deviation"]
SAVE_FORMAT = None                   # e.g. "pdf" or "png": also save every figure
                                     # into the study folder


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "Results"

CHANNELS = ("xr", "fc", "tipA", "tipB")
PERIOD_SAMPLES = 2048       # dense time grid every signal is reconstructed on, so
                            # runs with different N_t are compared on equal terms

# the parameters that define the physical problem and its discretisation; runs in
# one folder must agree on them, otherwise their curves describe different systems
PHYSICAL_KEYS = ("n_elem", "L_A", "LB_rel", "E", "rho", "A", "xi", "F0", "gap",
                 "harmonics", "sample_number")

STYLE_FILE = "plot_styles.csv"
STYLE_COLUMNS = ("file", "label", "color", "linestyle", "linewidth",
                 "representation", "marker", "markersize", "markevery")
REPRESENTATIONS = ("line", "markers", "line+markers")
SETTINGS_KEYS = ("plots", "xlim", "omega_star", "reference", "harmonics_shown",
                 "gap_linthresh_um")
DEFAULT_SETTINGS = dict(plots=PLOTS, xlim=None, omega_star=None, reference=None,
                        harmonics_shown=[0, 1, 2, 3], gap_linthresh_um=1e-3)

# Okabe-Ito, colour-blind safe; black first, for the DLFT run
DEFAULT_COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73", "#E69F00",
                  "#CC79A7", "#56B4E9", "#F0E442"]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_LINEWIDTH = 1.5
DEFAULT_MARKER = "o"
DEFAULT_MARKERSIZE = 3.0

SOLVER_TEXT = {"pyfbs-nlfbs": "pyFBS", "nlvib-hb": "NLvib"}
METHOD_TEXT = {"dlft_aft": "DLFT", "dlft": "DLFT (DLFTContact)", "aft": "AFT"}
LAW_TEXT = {"penalty": "penalty", "tanh": "tanh-regularized penalty",
            "softplus": "softplus-regularized penalty"}


# ============================ writing =======================================

def write_csv(path, description, config, omega, iterations, harmonics, amplitudes):
    """Write one result in the CSV contract of this module.

    :param path: target file; missing parent folders are created.
    :param description: human-readable header lines, without the leading ``#``.
    :param config: JSON-serializable parameter dict, written as ``config_json``.
    :param omega: (n,) angular frequencies [rad/s].
    :param iterations: (n,) corrector iterations per converged point.
    :param harmonics: retained harmonics, ascending.
    :param amplitudes: (n, n_harmonics, len(CHANNELS)) complex physical amplitudes.
    """
    omega = np.asarray(omega, dtype=float)
    n = len(omega)
    reim = np.stack([amplitudes.real, amplitudes.imag], axis=-1)   # re/im adjacent
    table = np.column_stack([omega, omega / (2.0 * np.pi),
                             np.asarray(iterations, dtype=float),
                             reim.reshape(n, -1)])    # harmonic-major, then channel
    columns = (["omega_rad_s", "freq_hz", "iterations"]
               + [f"{part}_h{h}_{ch}" for h in harmonics for ch in CHANNELS
                  for part in ("re", "im")])
    assert table.shape[1] == len(columns), (table.shape, len(columns))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for line in description:
            fh.write(f"# {line}\n")
        fh.write(f"# config_json: {json.dumps(config, sort_keys=True)}\n")
        fh.write(",".join(columns) + "\n")
        np.savetxt(fh, table, delimiter=",", fmt="%.10e")
    print(f"result written: {path}  ({n} points)")


def save_result(path, config, solution_set, problem, system, solve_time):
    """Write a converged pyFBS branch in the CSV contract of this module.

    ``xr`` is the Newton unknown itself, ``fc`` the interface force the solver's
    own nonlinear method returns for it (the truncated DLFT multiplier, or the
    AFT penalty force), and the tips come from ``FBSProblem.compute_full_response``.

    :param path: target CSV.
    :param config: parameter dict of the run (``main.run_config``).
    :param solution_set: the ``SolutionSet`` returned by ``solve_and_continue``.
    :param problem: the ``FBSProblem`` that produced it.
    :param system: the two-rod system, for the tip indices.
    :param solve_time: wall-clock time of the continuation [s].
    """
    # the solver is only needed for writing; reading and plotting work without it
    from pyfbs.nonlinearFBS import Fourier, FourierOmegaPoint

    harmonics = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples
    rfft_to_physical = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])

    n = len(solution_set.omega)
    amplitudes = np.empty((n, len(harmonics), len(CHANNELS)), dtype=complex)
    for i, (fourier, omega) in enumerate(zip(solution_set.fourier, solution_set.omega)):
        point = FourierOmegaPoint(fourier, omega)
        full = problem.compute_full_response(fourier, omega).coefficients[:, :, 0]
        amplitudes[i, :, CHANNELS.index("xr")] = fourier.coefficients[:, 0, 0]
        amplitudes[i, :, CHANNELS.index("fc")] = problem.method.compute_F_int(
            point, problem.ode).reshape(len(harmonics))
        amplitudes[i, :, CHANNELS.index("tipA")] = full[:, system.rod_tip_idx]
        amplitudes[i, :, CHANNELS.index("tipB")] = full[:, system.tipB_idx]
    amplitudes *= rfft_to_physical[None, :, None]

    config = dict(config, solve_time_s=float(solve_time), n_points=n)
    description = [
        f"two_rod_unilateral_contact -- pyFBS nonlinear FBS, "
        f"{describe(config, Path(path).stem)}",
        "structure: two clamped-free axial FE rods (A driven at its tip, B passive),"
        " coupled on x_r = u_A + u_B, contact when x_r > gap",
        "channels: xr = x_r, fc = contact force on x_r (truncated to the retained"
        " harmonics, positive in compression), tipA = u_A, tipB = u_B",
        "reconstruction: u(t) = Re( sum_h A_h exp(1j h omega t) ),"
        "  A_h = re_h<h>_<ch> + 1j im_h<h>_<ch>",
    ]
    write_csv(path, description, config, solution_set.omega,
              solution_set.iterations, harmonics, amplitudes)


# ============================ reading =======================================

def read_header(path):
    """The leading ``#`` lines of a result CSV, without the marker."""
    lines = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line[1:].strip())
    return lines


def read_config(path):
    """The parameter dict of the ``config_json`` header line, or None."""
    for line in read_header(path):
        if line.startswith("config_json:"):
            return json.loads(line[len("config_json:"):].strip())
    return None


def period_signal(amplitudes, harmonics, n_t=PERIOD_SAMPLES):
    """One period of u(t) = Re( sum_h A_h exp(1j h tau) ), tau = 2 pi k / n_t.

    :param amplitudes: (..., n_harmonics) complex physical amplitudes.
    :param harmonics: the harmonic of every amplitude column.
    :returns: (..., n_t) real signal.
    """
    tau = 2.0 * np.pi * np.arange(n_t) / n_t
    return np.real(np.asarray(amplitudes) @ np.exp(1j * np.outer(harmonics, tau)))


def read_result(path):
    """One result CSV into the quantities the figures need.

    :returns: dict with ``name``, ``config``, ``harmonics``, ``omega`` [rad/s],
        ``iterations``, ``amplitudes`` (channel -> (n, n_harmonics) complex), the
        period maxima ``xr_max``, ``tipA_max``, ``tipB_max``, the peak contact
        force ``fc_max``, the gap ``gap`` from the config and the minimum gap
        ``gap_min`` = min_t [g0 - x_r(t)] (None without a gap in the config).
    """
    path = Path(path)
    config = read_config(path)
    frame = pd.read_csv(path, comment="#")
    harmonics = sorted({int(m.group(1)) for m in
                        (re.match(r"re_h(\d+)_xr$", c) for c in frame.columns) if m})
    if not harmonics:
        raise KeyError(f"{path.name}: no re_h<h>_xr columns -- not a two-rod result CSV")
    amplitudes = {ch: np.column_stack([frame[f"re_h{h}_{ch}"].to_numpy()
                                       + 1j * frame[f"im_h{h}_{ch}"].to_numpy()
                                       for h in harmonics])
                  for ch in CHANNELS}

    xr = period_signal(amplitudes["xr"], harmonics)                 # (n, PERIOD_SAMPLES)
    gap = (config or {}).get("gap")
    return dict(
        name=path.stem,
        config=config,
        harmonics=harmonics,
        omega=frame["omega_rad_s"].to_numpy(),
        iterations=frame["iterations"].to_numpy(),
        amplitudes=amplitudes,
        xr_max=np.abs(xr).max(axis=1),
        tipA_max=np.abs(period_signal(amplitudes["tipA"], harmonics)).max(axis=1),
        tipB_max=np.abs(period_signal(amplitudes["tipB"], harmonics)).max(axis=1),
        fc_max=period_signal(amplitudes["fc"], harmonics).max(axis=1),
        gap=gap,
        gap_min=None if gap is None else gap - xr.max(axis=1),
    )


def _same_value(a, b):
    """Equality of two config entries, scalars or lists, up to float round-off
    (MATLAB and Python print the same number with different digits)."""
    if a is None or b is None:
        return a is None and b is None
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    return a.shape == b.shape and bool(np.allclose(a, b, rtol=1e-9, atol=0.0))


def check_consistency(curves):
    """Warn when the runs of one folder describe different physical systems.

    Every entry of :data:`PHYSICAL_KEYS` must agree across all files; each
    disagreement is printed with the value of every file, so a parameter changed
    on one side only (pyFBS or NLvib) is caught before the curves are compared.
    """
    unique = {curve["name"]: curve for curve in curves}.values()
    missing = [c["name"] for c in unique if c["config"] is None]
    if missing:
        print(f"WARNING: no config_json in {', '.join(missing)} -- their parameters"
              f" cannot be checked")
    known = [c for c in unique if c["config"] is not None]
    mismatches = 0
    for key in PHYSICAL_KEYS:
        values = {c["name"]: c["config"].get(key) for c in known}
        first = next(iter(values.values()), None)
        if not all(_same_value(value, first) for value in values.values()):
            mismatches += 1
            listing = ", ".join(f"{name}: {value}" for name, value in values.items())
            print(f"WARNING: parameter '{key}' differs within the folder -- {listing}")
    if known and not mismatches:
        print(f"  parameters consistent across {len(known)} files "
              f"({', '.join(PHYSICAL_KEYS)})")


# ============================ labels and style spec =========================

def _number(value):
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def describe(config, fallback):
    """Default legend label of one run: solver, method and contact law."""
    if not config:
        return fallback
    solver = SOLVER_TEXT.get(config.get("solver"), config.get("solver", "?"))
    method = METHOD_TEXT.get(config.get("method"), config.get("method", "?"))
    law = config.get("law")
    if law == "rigid":
        text = f"{solver} {method}, rigid contact"
        if config.get("epsilon_rel") is not None:
            text += f", eps = {_number(config['epsilon_rel'])} k_rod"
        return text
    text = (f"{solver} {method}, {LAW_TEXT.get(law, law)}, "
            f"k_c = {_number(config.get('k_rel'))} k_rod")
    if law in ("tanh", "softplus"):
        text += f", alpha = {_number(config.get('alpha'))} 1/m"
    return text


def auto_labels(curves):
    """One label per curve from :func:`describe`; runs that end up with the same
    label (e.g. two DLFT runs with different epsilon) get the config entries
    they differ in appended."""
    labels = [describe(c["config"], c["name"]) for c in curves]
    bookkeeping = {"solve_time_s", "n_points"}
    for label in set(labels):
        group = [i for i, other in enumerate(labels) if other == label]
        if len(group) < 2:
            continue
        configs = [curves[i]["config"] or {} for i in group]
        keys = sorted(k for k in set().union(*configs) if k not in bookkeeping
                      and len({json.dumps(c.get(k), sort_keys=True)
                               for c in configs}) > 1)
        for i, config in zip(group, configs):
            extra = ", ".join(f"{k}={json.dumps(config.get(k))}" for k in keys)
            labels[i] = f"{label} ({extra or curves[i]['name']})"
    return labels


def _style_kwargs(row, index):
    """One style-spec row into matplotlib line kwargs; empty cells take the
    cycled defaults of position ``index``."""
    def cell(key):
        return (row.get(key) or "").strip()

    representation = cell("representation") or "line"
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}, "
                         f"got {representation!r}")
    linestyle = cell("linestyle") or DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)]
    if "," in linestyle:                   # on/off dash sequence in points
        linestyle = (0, tuple(float(v) for v in linestyle.split(",")))

    style = dict(color=cell("color") or DEFAULT_COLORS[index % len(DEFAULT_COLORS)],
                 linewidth=float(cell("linewidth") or DEFAULT_LINEWIDTH),
                 linestyle="none" if representation == "markers" else linestyle)
    if representation != "line":
        style.update(marker=cell("marker") or DEFAULT_MARKER,
                     markersize=float(cell("markersize") or DEFAULT_MARKERSIZE),
                     markevery=int(cell("markevery") or 1))
    return style


def _parse_setting(key, value):
    """One ``# key: value`` line of the style spec into its Python value."""
    if key == "plots":
        return [s.strip() for s in value.split(",") if s.strip()]
    if key == "xlim":
        low, high = (float(v) for v in value.split(","))
        return (low, high)
    if key in ("omega_star", "gap_linthresh_um"):
        return float(value)
    if key == "harmonics_shown":
        return [int(v) for v in value.split(",")]
    return value                                             # reference


def _write_style_spec(spec_path, names, labels):
    """Seed a style spec from the folder contents, one row per CSV."""
    with open(spec_path, "w", newline="", encoding="utf-8") as fh:
        fh.write("# plot settings of this folder, blank -> the default in plotting.py\n")
        for key in SETTINGS_KEYS:
            fh.write(f"# {key}:\n")
        writer = csv.writer(fh)
        writer.writerow(STYLE_COLUMNS)
        for i, (name, label) in enumerate(zip(names, labels)):
            writer.writerow([name, label,
                             DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                             DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                             DEFAULT_LINEWIDTH, "line", DEFAULT_MARKER,
                             DEFAULT_MARKERSIZE, 1])


def load_folder(folder):
    """Read every result CSV of a study folder, styled by its ``plot_styles.csv``.

    The spec is written from the folder contents if it does not exist yet (pyFBS
    runs first, so the DLFT branch gets the first colour). Rows naming a missing
    file are skipped with a note, CSVs the spec does not list are appended with
    default styles.

    :returns: (entries, settings): entries is a list of (curve, label, style) in
        the spec's row order, settings the defaults updated by the folder's own.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a folder -- name a study folder"
                                 f" inside {RESULTS}")
    files = sorted((p for p in folder.glob("*.csv") if p.name != STYLE_FILE),
                   key=lambda p: (not p.name.startswith("pyfbs"), p.name))
    if not files:
        raise FileNotFoundError(f"no result CSVs in {folder}")
    curves = {p.name: read_result(p) for p in files}
    labels = dict(zip(curves, auto_labels(list(curves.values()))))

    spec_path = folder / STYLE_FILE
    if not spec_path.exists():
        _write_style_spec(spec_path, list(curves), list(labels.values()))
        print(f"created {spec_path} -- edit it to control labels, colours, line"
              f" styles, representation and the plot order")

    settings = dict(DEFAULT_SETTINGS)
    with open(spec_path, newline="", encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines:
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            key, value = key.strip(), value.strip()
            if key in SETTINGS_KEYS and value:
                settings[key] = _parse_setting(key, value)

    entries, listed = [], set()
    rows = csv.DictReader(line for line in lines if not line.startswith("#"))
    for row in rows:
        name = (row.get("file") or "").strip()
        if not name:
            continue
        if name not in curves:
            print(f"note: {STYLE_FILE} lists {name}, which is not in the folder -- skipped")
            continue
        listed.add(name)
        label = (row.get("label") or "").strip() or labels[name]
        entries.append((curves[name], label, _style_kwargs(row, len(entries))))
    extra = [name for name in curves if name not in listed]
    if extra:
        print(f"note: not listed in {STYLE_FILE}, appended with default styles:"
              f" {', '.join(extra)}")
        for name in extra:
            entries.append((curves[name], labels[name], _style_kwargs({}, len(entries))))
    return entries, settings


# ============================ figure helpers ================================

def _omega_axes(axes, settings):
    """Grid and frequency range of a column of axes sharing the omega axis."""
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        if settings["xlim"]:
            ax.set_xlim(*settings["xlim"])
    axes[-1].set_xlabel(r"$\omega$  [rad/s]")


def _time_style(style):
    """A curve style for one period on PERIOD_SAMPLES points: markers, if any,
    thinned to about 40 per period instead of one per sample."""
    style = dict(style)
    if "marker" in style:
        style["markevery"] = max(1, PERIOD_SAMPLES // 40)
    return style


def reference_entry(entries, settings):
    """The entry the others are measured against: the ``reference`` setting (a
    file name of the folder), else the first row of the style spec."""
    name = settings["reference"]
    if name:
        for entry in entries:
            if entry[0]["name"] == Path(name).stem:
                return entry
        raise KeyError(f"reference {name!r} is not among the plotted files")
    return entries[0]


def operating_point(entries, settings):
    """The frequency the time-domain figures are taken at, and each curve's point.

    The reference point is the reference curve's point nearest to the
    ``omega_star`` setting or, without it, the point where its contact force
    peaks, i.e. deep inside the contact band. Every curve then contributes its own
    converged point closest to that one in relative (omega, max_t|u_A|) -- the
    branches are sampled wherever each continuation happened to step, and on an
    overhang the amplitude term keeps all picks on the same branch.

    :returns: (omega_star, [index into each curve]).
    """
    reference = reference_entry(entries, settings)[0]
    if settings["omega_star"] is None:
        i_ref = int(np.argmax(reference["fc_max"]))
    else:
        i_ref = int(np.argmin(np.abs(reference["omega"] - settings["omega_star"])))
    omega_star = float(reference["omega"][i_ref])
    amplitude_star = float(reference["tipA_max"][i_ref])
    picks = []
    for curve, _, _ in entries:
        distance = (((curve["omega"] - omega_star) / omega_star) ** 2
                    + ((curve["tipA_max"] - amplitude_star) / amplitude_star) ** 2)
        picks.append(int(np.argmin(distance)))
    return omega_star, picks


def envelope_deviation(curve, reference, key, width):
    """Relative envelope deviation per frequency bin (thesis eq. envelope_metric).

    Within every bin of ``width`` rad/s covered by both branches, the largest
    value of ``key`` is compared: e = |env - env_ref| / env_ref. Taking the bin
    maximum keeps the metric well defined on overhangs, where a branch has
    several points per frequency.

    :returns: (bin centres, errors).
    """
    low = max(curve["omega"].min(), reference["omega"].min())
    high = min(curve["omega"].max(), reference["omega"].max())
    edges = np.arange(low, high + width, width)
    centres, errors = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        in_curve = (curve["omega"] >= a) & (curve["omega"] < b)
        in_reference = (reference["omega"] >= a) & (reference["omega"] < b)
        if in_curve.any() and in_reference.any():
            envelope = curve[key][in_curve].max()
            envelope_ref = reference[key][in_reference].max()
            centres.append(0.5 * (a + b))
            errors.append(abs(envelope - envelope_ref) / envelope_ref)
    return np.array(centres), np.array(errors)


def peak_deviation(curve, reference, key):
    """Relative deviation of the peak value and of its frequency (thesis eq.
    peak_metrics): (e_a, e_f)."""
    i, i_ref = int(np.argmax(curve[key])), int(np.argmax(reference[key]))
    e_a = abs(curve[key][i] - reference[key][i_ref]) / reference[key][i_ref]
    e_f = abs(curve["omega"][i] - reference["omega"][i_ref]) / reference["omega"][i_ref]
    return e_a, e_f


# ============================ plot methods ==================================
# Each method takes (entries, settings, title) and returns a Figure; entries are
# the (curve, label, style) triples of load_folder.

PLOT_METHODS = {}


def plot_method(name):
    """Register a plot method under ``name`` (see :data:`PLOT_METHODS`)."""
    def register(function):
        PLOT_METHODS[name] = function
        return function
    return register


@plot_method("min_gap")
def plot_min_gap(entries, settings, title):
    """Minimum time-domain gap over the frequency range, g_min = min_t [g0 - x_r(t)].

    g_min > 0: the rods never touch during the period; g_min = 0: they are in
    contact (the rigid DLFT branch sits here, up to the ripple of the truncated
    signal); g_min < 0: interpenetration, which a penalty law needs to produce a
    force, roughly -max f_c / k_c. The axis is symmetric-logarithmic in
    micrometres, linear within +-``gap_linthresh_um``, so that the open gap of
    g0 and sub-micrometre penetrations are readable in one figure.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    for curve, label, style in entries:
        if curve["gap_min"] is None:
            print(f"note: {curve['name']} has no gap in its config -- not in min_gap")
            continue
        ax.plot(curve["omega"], curve["gap_min"] * 1e6, label=label, **style)
    linthresh = settings["gap_linthresh_um"]
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axhline(0.0, color="0.3", linewidth=0.8, zorder=0)
    bottom, top = ax.get_ylim()
    bottom = min(bottom, -10.0 * linthresh)          # keep the penetration side visible
    ax.axhspan(bottom, 0.0, color="#D55E00", alpha=0.07, zorder=0,
               label="interpenetration")
    ax.set_ylim(bottom, top)
    ax.set_ylabel(r"$g_{\min} = \min_t\,[g_0 - x_r(t)]$  [µm]")
    ax.set_title(f"{title} -- minimum gap:  > 0 no contact,  = 0 contact,"
                 f"  < 0 interpenetration")
    ax.legend(fontsize=8, loc="best")
    _omega_axes([ax], settings)
    return fig


@plot_method("frc")
def plot_frc(entries, settings, title):
    """Forced response: period maxima of the driven tip u_A and of x_r, with the gap."""
    fig, (ax_tip, ax_rel) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                         layout="constrained")
    for curve, label, style in entries:
        ax_tip.plot(curve["omega"], curve["tipA_max"], label=label, **style)
        ax_rel.plot(curve["omega"], curve["xr_max"], label=label, **style)
    gaps = sorted({curve["gap"] for curve, _, _ in entries if curve["gap"] is not None})
    for i, gap in enumerate(gaps):
        ax_rel.axhline(gap, color="red", linestyle="--", linewidth=1.0,
                       label=r"gap $g_0$" if i == 0 else None)
    ax_tip.set_ylabel(r"$\max_t |u_A(t)|$  [m]")
    ax_rel.set_ylabel(r"$\max_t |x_r(t)|$  [m]")
    ax_tip.set_title(f"{title} -- forced response")
    ax_tip.legend(fontsize=8, loc="best")
    _omega_axes([ax_tip, ax_rel], settings)
    return fig


@plot_method("harmonics")
def plot_harmonics(entries, settings, title):
    """Amplitudes |A_h(x_r)| of the harmonics named in ``harmonics_shown``."""
    shown = settings["harmonics_shown"]
    fig, axes = plt.subplots(len(shown), 1, figsize=(9, 2.4 * len(shown)),
                             sharex=True, squeeze=False, layout="constrained")
    axes = axes[:, 0]
    for ax, h in zip(axes, shown):
        for curve, label, style in entries:
            if h in curve["harmonics"]:
                column = curve["harmonics"].index(h)
                ax.semilogy(curve["omega"], np.abs(curve["amplitudes"]["xr"][:, column]),
                            label=label, **style)
        ax.set_ylabel(rf"$|A_{{{h}}}(x_r)|$  [m]")
    axes[0].set_title(f"{title} -- harmonics of the relative displacement")
    axes[0].legend(fontsize=8, loc="best")
    _omega_axes(axes, settings)
    return fig


@plot_method("time_signals")
def plot_time_signals(entries, settings, title):
    """Gap g(t) and contact force f_c(t) over one period at one operating point
    (:func:`operating_point`), both reconstructed from the truncated harmonics."""
    omega_star, picks = operating_point(entries, settings)
    fig, (ax_gap, ax_force) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                           layout="constrained")
    t = np.arange(PERIOD_SAMPLES) / PERIOD_SAMPLES
    for (curve, label, style), i in zip(entries, picks):
        xr = period_signal(curve["amplitudes"]["xr"][i], curve["harmonics"])
        fc = period_signal(curve["amplitudes"]["fc"][i], curve["harmonics"])
        gap = curve["gap"] if curve["gap"] is not None else 0.0
        ax_gap.plot(t, (gap - xr) * 1e6, **_time_style(style),
                    label=rf"{label}  ($\omega$ = {curve['omega'][i]:.1f})")
        ax_force.plot(t, fc, **_time_style(style))
    for ax in (ax_gap, ax_force):
        ax.axhline(0.0, color="0.3", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.3)
    ax_gap.set_ylabel(r"$g(t) = g_0 - x_r(t)$  [µm]")
    ax_force.set_ylabel(r"$f_c(t)$  [N]")
    ax_force.set_xlabel(r"$t / T$")
    ax_force.set_xlim(0.0, 1.0)
    ax_gap.set_title(rf"{title} -- one period at $\omega$ = {omega_star:.1f} rad/s")
    ax_gap.legend(fontsize=8, loc="best")
    return fig


@plot_method("contact_law")
def plot_contact_law(entries, settings, title):
    """Contact force over gap during one period at the operating point.

    Left the whole period, right a zoom on the contact phase around g = 0. The
    thick black line is the ideal unilateral law (g >= 0, f_c >= 0, g f_c = 0):
    the rigid DLFT branch oscillates around it because g and f_c are both
    truncated to the retained harmonics, while the penalty laws follow their own
    force law into negative g.
    """
    omega_star, picks = operating_point(entries, settings)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    penetration, gap_max, force_max = 0.0, 0.0, 0.0
    for (curve, label, style), i in zip(entries, picks):
        xr = period_signal(curve["amplitudes"]["xr"][i], curve["harmonics"])
        fc = period_signal(curve["amplitudes"]["fc"][i], curve["harmonics"])
        g = (curve["gap"] if curve["gap"] is not None else 0.0) - xr
        for ax in (ax_full, ax_zoom):
            ax.plot(g * 1e6, fc, **_time_style(style),
                    label=label if ax is ax_full else None)
        penetration = max(penetration, -g.min())
        gap_max, force_max = max(gap_max, g.max()), max(force_max, fc.max())
    for ax in (ax_full, ax_zoom):
        ax.plot([gap_max * 1e6, 0.0, 0.0], [0.0, 0.0, 1.1 * force_max], color="k",
                linewidth=2.5, zorder=0,
                label="unilateral contact law" if ax is ax_full else None)
        ax.set_xlabel(r"$g = g_0 - x_r$  [µm]")
        ax.grid(True, alpha=0.3)
    half_width = max(5.0 * penetration, 1e-3 * gap_max) * 1e6
    ax_zoom.set_xlim(-half_width, half_width)
    ax_full.set_ylabel(r"$f_c$  [N]")
    ax_full.set_title(rf"{title} -- contact law at $\omega$ = {omega_star:.1f} rad/s")
    ax_zoom.set_title("contact phase, around g = 0")
    ax_full.legend(fontsize=8, loc="best")
    return fig


@plot_method("deviation")
def plot_deviation(entries, settings, title):
    """Deviation of every curve from the reference curve.

    Envelope deviation per frequency bin of 1 % of omega_1 for the driven tip
    and for x_r (:func:`envelope_deviation`), plus a printed table of median and
    maximum envelope deviation and of the peak deviations e_a, e_f of the driven
    tip (:func:`peak_deviation`).
    """
    reference, reference_label, _ = reference_entry(entries, settings)
    omega_1 = (reference["config"] or {}).get("omega_1")
    width = 0.01 * (omega_1 or np.ptp(reference["omega"]))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, layout="constrained")
    quantities = (("tipA_max", r"$e_{\mathrm{env}}(\max_t |u_A|)$"),
                  ("xr_max", r"$e_{\mathrm{env}}(\max_t |x_r|)$"))

    print(f"\n  deviation from the reference '{reference_label}' "
          f"(bins of {width:.1f} rad/s):")
    print(f"  {'curve':<60s} {'e_env median':>12s} {'e_env max':>10s}"
          f" {'e_a':>9s} {'e_f':>9s}")
    for curve, label, style in entries:
        if curve is reference:
            continue
        for ax, (key, _) in zip(axes, quantities):
            centres, errors = envelope_deviation(curve, reference, key, width)
            # an exact match would vanish from a log axis, so it is drawn at 1e-16
            ax.semilogy(centres, np.maximum(errors, 1e-16), label=label, **style)
        _, errors = envelope_deviation(curve, reference, "tipA_max", width)
        e_a, e_f = peak_deviation(curve, reference, "tipA_max")
        median, largest = ((np.median(errors), errors.max()) if len(errors)
                           else (np.nan, np.nan))
        print(f"  {label:<60s} {median:12.2e} {largest:10.2e} {e_a:9.2e} {e_f:9.2e}")
    for ax, (_, ylabel) in zip(axes, quantities):
        ax.set_ylabel(ylabel)
    axes[0].set_title(f"{title} -- envelope deviation from {reference_label}")
    axes[0].legend(fontsize=8, loc="best")
    _omega_axes(axes, settings)
    return fig


# ============================ driver ========================================

def draw(folder, plots=None):
    """Every selected figure of one study folder.

    Precedence of the figure selection: the command line, then the folder's own
    ``plot_styles.csv``, then :data:`PLOTS`.
    """
    folder = Path(folder)
    print(f"\n{folder.name}:")
    entries, settings = load_folder(folder)
    for curve, label, _ in entries:
        print(f"  {curve['name']}: {len(curve['omega'])} points, omega "
              f"{curve['omega'].min():.1f}..{curve['omega'].max():.1f} rad/s  ->  {label}")
    check_consistency([curve for curve, _, _ in entries])
    for name in plots or settings["plots"]:
        if name not in PLOT_METHODS:
            raise KeyError(f"unknown plot {name!r}; registered: "
                           f"{', '.join(sorted(PLOT_METHODS))}")
        fig = PLOT_METHODS[name](entries, settings, folder.name)
        if SAVE_FORMAT:
            fig.savefig(folder / f"{name}.{SAVE_FORMAT}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Overlay the result CSVs of study folders inside Results/.")
    parser.add_argument("folders", nargs="*",
                        help=f"study folder(s) inside Results/ (default: {RESULT_DIR})")
    parser.add_argument("--plots", default=None,
                        help="comma-separated plot names (default: the folder's "
                             "plot_styles.csv, else PLOTS)")
    args = parser.parse_args()
    selected = [s.strip() for s in args.plots.split(",")] if args.plots else None
    for folder_name in args.folders or [RESULT_DIR]:
        draw(RESULTS / folder_name, selected)
    plt.show()
