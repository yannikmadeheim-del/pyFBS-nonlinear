"""Saving and read-back helpers for the two bar+beam friction example.

The CSV contract is the one used by ``testbench_joint_comparison``: a block of
``#`` comment lines carrying the run metadata, the force laws and the
reconstruction formula, whose LAST line is always ``# config_json: {...}``, then
one row per converged continuation point.

Columns::

    freq_hz, omega_rad_s, iterations, step_length, uout_h1_abs, uout_time_max,
    re_h<h>_<dof>, im_h<h>_<dof>, ...        harmonic-major, then DOF

The harmonic block holds the FULL system solution -- the two interface relative
coordinates (the Newton unknowns), all six physical DOFs q1..q6 recovered with
``FBSProblem.compute_full_response``, and the converged interface force
[f_N, f_T] as the nonlinear method itself returned it -- as complex amplitudes
``a_h`` scaled so that

    u(t)    = Re( sum_h a_h exp(1j h omega t) )
    udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) )

i.e. ``|a_h|`` is the physical amplitude, not the raw rFFT solver coefficient
(``a_h = c_h/Nt`` for h = 0 and ``2 c_h/Nt`` otherwise).

Read back with ``pandas.read_csv(path, comment="#")`` or :func:`read_result`.

The plotting infrastructure lives here too. Figures are produced by named plot
methods in :data:`PLOT_METHODS`, registered with the :func:`plot_method`
decorator; ``plot_comparison.py`` only selects which of them to draw for which
folders. To add one, write a function taking
``(curves, labels, styles, title, xlim, force_omega)`` and decorate it -- it
becomes selectable by name immediately, with no change to any caller.
"""
import csv
import json
from pathlib import Path

import numpy as np
from numpy.fft import irfft

from pyfbs.nonlinearFBS import Fourier, FourierOmegaPoint

from .dynamical_system import DOF_LABELS, IF_LABELS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# every exported channel, in the order the harmonic block uses: the kinematics
# first, then the converged interface force in the interface ordering
LABELS = list(IF_LABELS) + list(DOF_LABELS)
FORCE_LABELS = ("f_N", "f_T")
ALL_LABELS = LABELS + list(FORCE_LABELS)
OUT_LABEL = "q1"          # the plotted output channel: axial tip of element 1

PERIOD_SAMPLES = 1024     # resampling of one period for the time-domain panels

STYLE_COLUMNS = ("file", "label", "color", "linestyle", "linewidth")
SKIP_CSV = ("manifest.csv",)
SETTINGS_KEYS = ("methods", "force_omega", "xlim")

# Okabe-Ito, colour-blind safe; the first two entries are the two branches that
# are always compared (DLFT black, AFT blue)
DEFAULT_COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73",
                  "#E69F00", "#CC79A7", "#56B4E9", "#F0E442"]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_LINEWIDTH = 1.6


# ============================ writing ======================================

def export_header(cfg, system, method, frf_source, solve_time, n_points, n_t):
    """Self-contained comment header: what was run, with which force laws.

    The machine-readable ``config_json`` line is appended by :func:`save_solution`.
    """
    p = system.p
    lines = [
        f"two_bar_beam_friction -- thesis case study 6.4, "
        f"{'AFT' if method == 'aft' else 'DLFT contact + AFT friction'}, "
        f"{frf_source} FRF provider",
        f"method: {method}",
        f"solve_time_s: {solve_time:.6f}",
        f"n_points: {n_points}",
        f"harmonics: {list(cfg['harmonics'])}  (harmonic 0 carries the mean the"
        f" rectifying contact leaves in the response)",
        f"sample_number: {n_t}",
        "structure: two identical cantilever bar+beam elements, ONE SUBSTRUCTURE EACH,",
        "  uncoupled in the linear system (block-diagonal M, K); DOFs per tip",
        "  [axial, transverse, slope] = q1,q2,q3 (element 1) and q4,q5,q6 (element 2)",
        f"  l={p.l}, EA={p.EA}, EI={p.EI}, lambda={p.lam}, C=beta*K with beta={p.beta}",
        "interface: x_rel = [x_N ; x_T] = [q5 - q2 ; q1 - q4]; contact when x_N > eps",
    ]
    if method == "aft":
        lines += [
            "normal (AFT, regularized unilateral spring, thesis eq. 6.15):",
            f"  N = ln(1 + exp(k*alpha2*(x_N - eps)))/alpha2, "
            f"k={p.k}, alpha2={p.alpha2}, eps={p.eps}",
        ]
    else:
        lines += [
            "normal (DLFT, RIGID unilateral contact -- no k, no alpha2):",
            f"  f^N = max(0, L_N[z_r - f^nl,A] + epsilon*(L_N x_rel - g0)), "
            f"epsilon={cfg['dlft_epsilon']}, g0=eps={p.eps}",
        ]
    lines += [
        "tangential (regularized dry friction, thesis eq. 6.11 -- IDENTICAL in both runs):",
        f"  f_T = mu * N * tanh(alpha1 * d/dt x_T), mu={p.mu}, alpha1={p.alpha1}",
        f"excitation: f_ext = [P1 cos(omega t), P5 cos(omega t), 0, 0, 0, 0]^T, "
        f"P1={p.P1} (axial -> tangential sliding), P5={p.P5} (transverse ->"
        f" drives the normal contact open and closed; NO static preload)",
        "content: complex harmonic amplitudes a_h = re_h<h>_<dof> + 1j im_h<h>_<dof>"
        " of the FULL system solution:",
        "  x_rel_N, x_rel_T: the 2 interface relative coordinates (the Newton unknowns)",
        "  q1..q6: the full physical response (compute_full_response) of both substructures",
        "  f_N, f_T: the CONVERGED interface force, as the solver's own nonlinear"
        " method returned it --",
        "    " + ("AFT: the two regularized laws above, [N, f_T]" if method == "aft"
                  else "DLFT: L_N^T f^N + L_F^T f^F, i.e. the rigid contact force"
                       " and the friction force it drives"),
        "reconstruction: u(t) = Re( sum_h a_h exp(1j h omega t) );  velocity:"
        " udot(t) = Re( sum_h 1j h omega a_h exp(1j h omega t) )",
        "units: the thesis model is non-dimensional; omega_rad_s is the primary"
        " frequency axis (first undamped resonance at omega = 1), freq_hz ="
        " omega/(2 pi) is written for CSV compatibility only",
        f"columns: freq_hz, omega_rad_s | iterations, step_length (corrector"
        f" diagnostics) | uout_h1_abs = |a_1({OUT_LABEL})|, uout_time_max ="
        f" max_t |{OUT_LABEL}(t)| | re/im of a_h, harmonic-major, then channel"
        f" order as listed",
        f"uout (plotted output channel) = {OUT_LABEL}",
        f"channel order: {', '.join(ALL_LABELS)}",
    ]
    return lines


def save_solution(path, cfg, ss, problem, system, method, frf_source, solve_time):
    """Write the converged branch, full system solution, one row per point.

    :returns: (omega, uout_h1_abs, uout_time_max) for an immediate plot.
    """
    hlist = [int(h) for h in Fourier.harmonics]
    n_t = Fourier.number_of_time_samples

    # raw rFFT-convention coefficients: interface unknowns, full physical
    # response, converged interface force
    x_rel = np.array([f.coefficients[:, :, 0] for f in ss.fourier])      # (n, Nh, 2)
    n = x_rel.shape[0]
    q_full = np.empty((n, len(hlist), problem.d_total), dtype=complex)
    f_int = np.empty((n, len(hlist), problem.d_int), dtype=complex)
    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        q_full[i] = problem.compute_full_response(four, w).coefficients[:, :, 0]
        # the force the solver actually converged on, taken from its own
        # nonlinear method rather than re-derived here: AFT returns
        # interface_force = [N, f_T], DLFT returns L_N^T f^N + L_F^T f^F + f^nl,A
        # with L_N = [1, 0], L_F = [0, 1] and f^nl,A = 0, so in both methods row 0
        # is the normal force and row 1 the friction force. For DLFT the normal
        # force is only obtainable this way -- it is a prediction/correction
        # quantity, not a function of the displacements.
        point = FourierOmegaPoint(four, w)
        f_int[i] = problem.method.compute_F_int(point, problem.ode).reshape(
            len(hlist), problem.d_int)

    raw = np.concatenate([x_rel, q_full, f_int], axis=2)                 # (n, Nh, 10)
    assert raw.shape[2] == len(ALL_LABELS), (raw.shape, len(ALL_LABELS))

    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in hlist])
    amp = raw * scale[None, :, None]                     # physical amplitudes a_h

    i_out, i_h1 = LABELS.index(OUT_LABEL), hlist.index(1)
    # period maximum: zero-pad the retained harmonics back onto a dense rFFT bin
    # vector, then one irfft per point gives the full time series at once
    padded = np.zeros((n, max(hlist) + 1), dtype=complex)
    padded[:, hlist] = raw[:, :, i_out]
    uout_time_max = np.abs(irfft(padded, n=n_t, axis=1)).max(axis=1)
    uout_h1_abs = np.abs(amp[:, i_h1, i_out])

    omega = np.asarray(ss.omega, dtype=float)
    freq = omega / (2.0 * np.pi)
    reim = np.stack([amp.real, amp.imag], axis=-1)       # re/im adjacent per DOF
    table = np.column_stack([
        freq, omega,
        np.asarray(ss.iterations, dtype=float),
        np.asarray(ss.step_length, dtype=float),
        uout_h1_abs, uout_time_max,
        reim.reshape(n, -1),                             # harmonic-major, then DOF
    ])
    cols = (["freq_hz", "omega_rad_s", "iterations", "step_length",
             "uout_h1_abs", "uout_time_max"]
            + [f"{part}_h{h}_{lab}" for h in hlist for lab in ALL_LABELS
               for part in ("re", "im")])
    assert table.shape[1] == len(cols), (table.shape[1], len(cols))

    header = export_header(cfg, system, method, frf_source, solve_time, n, n_t)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        for line in header:
            fh.write(f"# {line}\n")
        fh.write(f"# config_json: {json.dumps(cfg, sort_keys=True)}\n")
        fh.write(",".join(cols) + "\n")
        np.savetxt(fh, table, delimiter=",", fmt="%.10e")
    print(f"solution written: {path}  ({n} points, {len(ALL_LABELS)} channels x"
          f" {len(hlist)} harmonics, {solve_time:.1f} s solve)")
    return omega, uout_h1_abs, uout_time_max


def result_path(cfg, method, frf_source, name=None):
    """results/<name>.csv; the auto name is the method plus the FRF source."""
    if name is None:
        name = f"{method}_{frf_source}.csv"
    return RESULTS / name


# ============================ reading back =================================

def read_header(path):
    """The leading '#' lines, '# ' stripped."""
    lines = []
    with open(path, "r") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            lines.append(line[1:].strip())
    return lines


def read_config(path):
    """The dict written on the '# config_json:' line, or None."""
    for line in read_header(path):
        if line.startswith("config_json:"):
            return json.loads(line[len("config_json:"):].strip())
    return None


def read_result(path):
    """One result CSV into the arrays the figures need.

    :returns: dict with ``omega``, ``out_max``/``out_h1`` (the output channel),
        ``gap_max``/``gap_h1`` (the two interface relative coordinates, period
        maximum and 1st-harmonic amplitude), ``force_amp`` (the (n, Nh, 2)
        complex amplitudes of the interface force, or None for a CSV written
        before the force block existed), ``harmonics``, ``config``.
    """
    import pandas as pd

    path = Path(path)
    config = read_config(path)
    df = pd.read_csv(path, comment="#")
    harmonics = sorted({int(c.split("_")[1][1:]) for c in df.columns
                        if c.startswith("re_h")})
    n_t = int(config["sample_number"]) if config else 1024

    def amplitudes(label):
        """(n, Nh) complex a_h of one channel, in ascending harmonic order."""
        return np.column_stack([df[f"re_h{h}_{label}"].to_numpy()
                                + 1j * df[f"im_h{h}_{label}"].to_numpy()
                                for h in harmonics])

    def period_max(label):
        """max_t |u(t)| over one period, from the physical amplitudes a_h."""
        a = amplitudes(label)
        # undo the a_h scaling to get rFFT bins back, then one irfft per row
        scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])
        padded = np.zeros((len(df), max(harmonics) + 1), dtype=complex)
        padded[:, harmonics] = a / scale[None, :]
        return np.abs(irfft(padded, n=n_t, axis=1)).max(axis=1)

    i_h1 = harmonics.index(1)
    gap_labels = list(IF_LABELS)
    # results written before the force block exists (and the external pyhbm
    # references) simply have no force columns -- the force panels skip them
    has_forces = all(f"re_h{harmonics[0]}_{l}" in df.columns for l in FORCE_LABELS)
    return dict(
        omega=df["omega_rad_s"].to_numpy(),
        out_max=df["uout_time_max"].to_numpy(),
        out_h1=df["uout_h1_abs"].to_numpy(),
        q4_max=period_max("q4"),
        gap_max=np.column_stack([period_max(l) for l in gap_labels]),
        gap_h1=np.column_stack([np.abs(amplitudes(l)[:, i_h1]) for l in gap_labels]),
        gap_labels=gap_labels,
        force_amp=(np.stack([amplitudes(l) for l in FORCE_LABELS], axis=2)
                   if has_forces else None),
        force_labels=list(FORCE_LABELS),
        harmonics=harmonics,
        config=config,
        name=path.stem,
    )


def period_signal(amplitudes, harmonics, n_t=PERIOD_SAMPLES):
    """One period of u(t) = Re( sum_h a_h exp(1j h tau) ) from amplitudes a_h.

    The physical amplitudes are put back onto rFFT bins for the REQUESTED
    ``n_t``, not the run's sample_number, so every curve is drawn on the same
    dense time grid however coarsely it was solved.

    :returns: (tau/2pi in [0, 1), u) -- one period, phase-aligned with the
        excitation cos(omega t) through the stored amplitudes.
    """
    harmonics = list(harmonics)
    scale = np.array([(1.0 if h == 0 else 2.0) / n_t for h in harmonics])
    padded = np.zeros(max(harmonics) + 1, dtype=complex)
    padded[harmonics] = np.asarray(amplitudes) / scale
    return np.arange(n_t) / n_t, irfft(padded, n=n_t)


def read_style_spec(folder, style_file="plot_styles.csv"):
    """Read (and on first use write) the folder's plot style spec.

    :returns: [(path, label, style_kwargs)] in the spec's row order. CSVs the
        spec does not list are appended at the end with a printed note.
    """
    folder = Path(folder)
    found = sorted(p for p in folder.glob("*.csv")
                   if p.name != style_file and p.name not in SKIP_CSV)
    if not found:
        raise FileNotFoundError(f"{folder}: no result CSVs to plot")

    spec_path = folder / style_file
    if not spec_path.exists():
        # seed the labels from whatever actually differs between the runs, so a
        # study folder does not come out labelled 000_aft, 001_aft, ...
        configs = [read_config(p) for p in found]
        _write_style_spec(spec_path, [p.name for p in found],
                          auto_labels(configs, [p.stem for p in found]))
        print(f"wrote {spec_path.name} -- edit it to control labels, colours,"
              f" line styles and the plot order")

    rows, listed = [], set()
    with open(spec_path, newline="") as fh:
        # the leading '#' lines carry the folder's plot settings, not curves
        for row in csv.DictReader(l for l in fh if not l.startswith("#")):
            name = (row.get("file") or "").strip()
            if not name:
                continue
            path = folder / name
            if not path.exists():
                print(f"note: {style_file} lists {name}, which does not exist -- skipped")
                continue
            listed.add(name)
            rows.append((path, (row.get("label") or path.stem).strip(),
                         _style_kwargs(row)))

    extra = [p for p in found if p.name not in listed]
    if extra:
        print(f"note: {', '.join(p.name for p in extra)} not listed in"
              f" {style_file} -- appended with default styles")
        for i, path in enumerate(extra):
            rows.append((path, path.stem, _style_kwargs({}, len(rows) + i)))
    return rows


def _style_kwargs(row, index=0):
    """One spec row into matplotlib kwargs; a 'a,b' linestyle becomes a dash tuple."""
    color = (row.get("color") or "").strip() or DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
    ls = (row.get("linestyle") or "").strip() or DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)]
    lw = (row.get("linewidth") or "").strip()
    style = dict(color=color, linewidth=float(lw) if lw else DEFAULT_LINEWIDTH)
    if "," in ls:
        style["linestyle"] = (0, tuple(float(v) for v in ls.split(",")))
    else:
        style["linestyle"] = ls
    return style


def read_plot_settings(folder, style_file="plot_styles.csv"):
    """The folder's own plot settings, from the '#' lines above the CSV header.

    Recognised keys (see :data:`SETTINGS_KEYS`): ``methods`` (comma-separated
    plot method names), ``force_omega`` (the frequency the time-domain force
    panels are taken at) and ``xlim``. A key that is missing or left blank falls
    back to the defaults in plot_comparison.py, so a plain style spec with no
    comment lines -- the testbench_joint_comparison format -- keeps working.

    :returns: dict with only the keys the folder actually sets.
    """
    spec_path = Path(folder) / style_file
    settings = {}
    if not spec_path.exists():
        return settings
    with open(spec_path, newline="") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            key, value = key.strip(), value.strip()
            if key not in SETTINGS_KEYS or not value:
                continue
            if key == "methods":
                settings[key] = [s.strip() for s in value.split(",") if s.strip()]
            elif key == "force_omega":
                settings[key] = float(value)
            elif key == "xlim":
                lo, hi = (float(v) for v in value.split(","))
                settings[key] = (lo, hi)
    return settings


def _write_style_spec(spec_path, names, labels=None):
    """Seed a style spec from the folder contents, one row per CSV."""
    labels = labels or [Path(n).stem for n in names]
    with open(spec_path, "w", newline="") as fh:
        # a blank settings block, so the file documents what it can override
        fh.write("# per-folder settings, blank -> the default in plot_comparison.py\n")
        for key in SETTINGS_KEYS:
            fh.write(f"# {key}:\n")
        writer = csv.writer(fh)
        writer.writerow(STYLE_COLUMNS)
        for i, (name, label) in enumerate(zip(names, labels)):
            writer.writerow([name, label,
                             DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                             DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                             DEFAULT_LINEWIDTH])


def _flatten(config):
    """Config -> {key: scalar}, with the parameter block inlined as params.<name>.

    Only hashable leaves are kept: the label logic compares values for equality,
    and dict/list entries (solver_kwargs, harmonics) are never what distinguishes
    the runs of a study.

    Parameters the run's method ignores are dropped. study.py records them in
    ``inert_params``; without this a DLFT run would be labelled with an alpha2 it
    never used, implying the value meant something.
    """
    config = config or {}
    inert = set(config.get("inert_params", ()))
    flat = {}
    for key, value in config.items():
        if key == "params" and isinstance(value, dict):
            flat.update({f"params.{k}": v for k, v in value.items()
                         if k not in inert})
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    return flat


def auto_labels(configs, fallback_names):
    """One label per run, naming only the settings that actually DIFFER.

    A study folder is otherwise labelled ``000_aft, 001_aft, ...``, which says
    nothing. Runs whose configs are identical (or missing) keep their filename.

    Only runs that HAVE a config take part in the comparison: an externally
    produced CSV carries no config_json, and counting its absent keys as
    differences would mark every single setting as varying and put the whole
    config into every label.
    """
    flats = [_flatten(c) for c in configs]
    known = [f for f, c in zip(flats, configs) if c]
    keys = set().union(*known) if known else set()
    varying = sorted(k for k in keys
                     if len({f.get(k, None) for f in known}) > 1)
    if not varying:
        return list(fallback_names)
    # a key absent from a run is one that run does not use -- omit it rather than
    # printing "=None", which would read as a value
    return [", ".join(f"{k.split('.')[-1]}={f[k]}" for k in varying if k in f) or name
            for f, name in zip(flats, fallback_names)]


# ============================ plotting =====================================
# Plot methods are registered by name so they can be selected, swapped and
# extended from plot_comparison.py without touching this module's callers.
# A method takes (curves, labels, styles, title, xlim) and returns a Figure;
# every curve is one dict from read_result.

PLOT_METHODS = {}


def plot_method(name):
    """Register a plot method under ``name`` (see :data:`PLOT_METHODS`)."""
    def decorate(function):
        PLOT_METHODS[name] = function
        return function
    return decorate


def _series_rows(curves, series):
    """[(key, ylabel)] -> one (key, column, ylabel) per axis row.

    A 2-D field expands into ONE ROW PER COLUMN -- the normal and tangential
    interface coordinates differ by more than an order of magnitude here, so
    sharing an axis would flatten one of them onto the baseline. ``ylabel`` is
    then a format string taking the column label.
    """
    rows = []
    for key, ylabel in series:
        if curves[0][key].ndim == 1:
            rows.append((key, None, ylabel))
        else:
            rows += [(key, j, ylabel.format(col))
                     for j, col in enumerate(curves[0]["gap_labels"])]
    return rows


def _draw_series(axes, rows, curves, labels, styles):
    """Overlay every curve against omega, one axis per row from :func:`_series_rows`."""
    for ax, (key, column, ylabel) in zip(axes, rows):
        for curve, label, style in zip(curves, labels, styles):
            values = curve[key] if column is None else curve[key][:, column]
            ax.plot(curve["omega"], values, label=label, **style)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)


def overlay(curves, labels, styles, series, title, xlim=None):
    """Shared layout: one figure, one row per plotted quantity, curves overlaid."""
    import matplotlib.pyplot as plt

    rows = _series_rows(curves, series)
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 2.8 * len(rows)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    _draw_series(axes, rows, curves, labels, styles)
    axes[-1].set_xlabel(r"$\omega$  [rad/s]")
    if xlim:
        axes[-1].set_xlim(*xlim)
    axes[0].set_title(title)
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


@plot_method("rel_disp_max")
def plot_rel_disp_max(curves, labels, styles, title="", xlim=None, force_omega=None):
    """Maximum time-domain relative displacement, max_t |x_rel(t)|.

    One row per interface DOF: the normal gap coordinate x_rel_N and the
    tangential sliding coordinate x_rel_T. This is the period maximum of the
    reconstructed signal, NOT a harmonic amplitude, so it includes every
    retained harmonic and the static offset.
    """
    return overlay(curves, labels, styles,
                   [("gap_max", r"$\max_t |${}$(t)|$")],
                   title=f"{title} -- max time-domain relative displacement",
                   xlim=xlim)


@plot_method("rel_disp_h1")
def plot_rel_disp_h1(curves, labels, styles, title="", xlim=None, force_omega=None):
    """First-harmonic amplitude of the interface relative displacement."""
    return overlay(curves, labels, styles,
                   [("gap_h1", r"$|${}$^{{(1)}}|$")],
                   title=f"{title} -- 1st-harmonic relative displacement",
                   xlim=xlim)


@plot_method("tip_response")
def plot_tip_response(curves, labels, styles, title="", xlim=None, force_omega=None):
    """Axial tip response of both substructures -- the DOFs the thesis plots."""
    return overlay(curves, labels, styles,
                   [("out_max", r"$\max_t |q_1(t)|$"),
                    ("q4_max",  r"$\max_t |q_4(t)|$")],
                   title=f"{title} -- axial tip response", xlim=xlim)


def _force_omega(curves, force_omega=None):
    """The frequency the time-domain force panels are taken at.

    An explicit value wins. Otherwise it is the RESONANCE OF THE DLFT BRANCH,
    read off the peak of the output channel -- the same quantity the thesis
    plots as 2|Q_1|. NOT off the interface coordinates: neither of them peaks
    here. The normal one is pinned at the gap by the contact, and the tangential
    one DECREASES towards resonance, because the growing amplitude locks the
    joint and suppresses sliding, so its maximum would just be the edge of the
    swept window.

    The same frequency is then used for every curve, so the panels compare the
    methods at one operating point rather than each at its own peak.
    """
    if force_omega is not None:
        return float(force_omega)
    dlft = [c for c in curves if (c["config"] or {}).get("method") == "dlft_aft"]
    reference = dlft[0] if dlft else curves[0]
    if not dlft:
        print(f"note: no dlft_aft run in this folder -- the force frequency is"
              f" taken from the resonance of {reference['name']}")
    peak = int(np.argmax(reference["out_max"]))
    if peak in (0, len(reference["out_max"]) - 1):
        print(f"note: {reference['name']} peaks at the edge of its swept window"
              f" -- the window may not contain the resonance; set force_omega"
              f" explicitly to pick the operating point")
    return float(reference["omega"][peak])


@plot_method("rel_disp_forces")
def plot_rel_disp_forces(curves, labels, styles, title="", xlim=None,
                         force_omega=None):
    """Max time-domain relative displacement, plus the contact forces at one omega.

    Upper rows: max_t |x_rel(t)| over the swept branch, one row per interface
    DOF, with a dashed marker at the frequency the force panels are taken at and
    a dot on the point each curve contributes there.

    Lower rows: the normal force N(t) and the friction force f_T(t) over one
    period at that frequency, as the solver converged them -- so the rigid DLFT
    contact and the regularized AFT law are compared as they were actually
    evaluated, not as they would be re-derived from the displacements.

    Curves whose CSV predates the force block are drawn in the upper rows only.
    """
    import matplotlib.pyplot as plt

    omega_star = _force_omega(curves, force_omega)
    # each branch is sampled where the continuation happened to step, so every
    # curve contributes its OWN nearest point; the deviation goes into the legend
    picks = [int(np.argmin(np.abs(curve["omega"] - omega_star))) for curve in curves]

    rows = _series_rows(curves, [("gap_max", r"$\max_t |${}$(t)|$")])
    fig, axes = plt.subplots(len(rows) + 2, 1,
                             figsize=(9, 2.6 * (len(rows) + 2)))
    sweep_axes, force_axes = axes[:len(rows)], axes[len(rows):]

    _draw_series(sweep_axes, rows, curves, labels, styles)
    for ax, (key, column, _) in zip(sweep_axes, rows):
        for curve, style, i in zip(curves, styles, picks):
            values = curve[key] if column is None else curve[key][:, column]
            ax.plot(curve["omega"][i], values[i], marker="o", markersize=4,
                    linestyle="none", color=style["color"])
        ax.axvline(omega_star, color="0.45", linestyle="--", linewidth=1.0, zorder=0)
        if xlim:
            ax.set_xlim(*xlim)
    for ax in sweep_axes[:-1]:
        ax.tick_params(labelbottom=False)
    sweep_axes[-1].set_xlabel(r"$\omega$  [rad/s]")
    sweep_axes[0].set_title(f"{title} -- max time-domain relative displacement")
    sweep_axes[0].legend(fontsize=8, loc="best")

    missing = [c["name"] for c in curves if c["force_amp"] is None]
    if missing:
        print(f"note: {len(missing)} of {len(curves)} results carry no force block"
              f" -- shown in the sweep only, re-run them to export f_N/f_T"
              f" ({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})")

    drawn = 0
    for curve, label, style, i in zip(curves, labels, styles, picks):
        if curve["force_amp"] is None:
            continue
        tau, normal = period_signal(curve["force_amp"][i, :, 0], curve["harmonics"])
        _, friction = period_signal(curve["force_amp"][i, :, 1], curve["harmonics"])
        force_axes[0].plot(tau, normal, **style,
                           label=rf"{label}  ($\omega$ = {curve['omega'][i]:.4f})")
        force_axes[1].plot(tau, friction, **style)
        drawn += 1

    for ax, ylabel in zip(force_axes, [r"$N(t)$", r"$f_T(t)$"]):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        # the normal contact is quasi-static: N stays close to the clamping
        # preload, and an offset-notation axis would show only its ripple and
        # read as if the force itself were tiny
        ax.ticklabel_format(axis="y", useOffset=False)
    force_axes[0].tick_params(labelbottom=False)
    force_axes[1].set_xlabel(r"$t / T$")
    force_axes[0].set_title(rf"contact forces at $\omega$ = {omega_star:.4f}")
    if drawn:
        force_axes[0].legend(fontsize=8, loc="best")
    else:
        force_axes[0].text(0.5, 0.5, "no force data in this folder",
                           ha="center", va="center", transform=force_axes[0].transAxes)
    fig.tight_layout()
    return fig
