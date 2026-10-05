"""
Plot the results of one study.py run -- no solver, no ANSYS.

Choose a folder inside Results/ with STUDY below, or on the command line::

    python plot_study.py parameter_sweep

What is drawn is decided by ``plot_styles.csv`` inside that folder, one row per
curve; its row order is the plot and legend order. Columns:

    file       .npz result written by study.py; a leading '#' disables the row
    label      legend text; empty -> the stored parameters in which the files differ
    color      matplotlib colour; empty -> cycled default
    linestyle  matplotlib line style ('-', '--', ...) or a dash sequence '4,2'
    linewidth  line width in points; empty -> DEFAULT_LINEWIDTH
    quantity   key of QUANTITIES; empty -> 'accelerance'
    channel    output sensor channel x, y or z; empty -> 'y'
    harmonic   harmonic whose amplitude is plotted; empty -> 1
    figure     rows with the same name share one figure; empty -> grouped by
               quantity, channel and harmonic

If the spec does not exist yet it is written from the folder contents, one row
per .npz, so a fresh study plots right away and the spec can be edited
afterwards. Result files the spec does not list are not plotted; their names
are printed. A new plotted quantity is one function plus one QUANTITIES entry,
see :func:`accelerance`.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thesis_plot_style as thesis_style

thesis_style.configure()

# --- Settings --------------------------------------------------------------------
STUDY      = "higherValues_cubicDamping_study_18.09.2026"  # folder inside Results/ (= STUDY_NAME in study.py)
STYLE_FILE = "plot_styles.csv"     # plot spec inside that folder
XLIM       = None                  # displayed frequency range [Hz], None = data range
LOG_Y      = True
SAVE       = False                 # True -> one <figure>.png per figure in the folder
# ---------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "Results"

MM_TO_M = 1e-3                        # the FE model and study.py work in mm
CHANNELS = {"x": 12, "y": 13, "z": 14}   # sensor channel of each direction of the output sensor

STYLE_COLUMNS = ("file", "label", "color", "linestyle", "linewidth",
                 "quantity", "channel", "harmonic", "figure")
DEFAULT_QUANTITY, DEFAULT_CHANNEL, DEFAULT_HARMONIC = "accelerance", "y", 1
DEFAULT_COLORS = ["black", "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                  "#17becf", "#8c564b", "#e377c2", "#7f7f7f"]
DEFAULT_LINESTYLES = ["-", "--", "-.", ":"]
DEFAULT_LINEWIDTH = 1.4


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_result(path):
    """
    One study.py result file as a dict: arrays stay arrays, the stored scalars
    (F0, K_TRANS, sample_number, ...) become Python numbers.

    :param path: .npz written by study.py
    """
    with np.load(path) as data:
        return {key: data[key].item() if data[key].ndim == 0 else data[key]
                for key in data.files}


def harmonic_amplitude(result, coefficients, harmonic):
    """
    Physical complex amplitude of one harmonic of a stored Fourier series.

    study.py stores the raw rfft coefficients c_h. The physical amplitude U_h,
    with u(t) = Re(U_h exp(i h omega t)), is 2/Nt * c_h for h >= 1 and
    1/Nt * c_0 for the mean.

    :param result: dict from :func:`load_result`
    :param coefficients: (n, Nh, d) coefficient array of that result, e.g. ``result["u_out"]``
    :param harmonic: harmonic number h
    :returns: (n, d) complex amplitudes
    """
    harmonics = [int(h) for h in result["harmonics"]]
    if harmonic not in harmonics:
        raise ValueError(f"harmonic {harmonic} was not solved for, "
                         f"the result holds the harmonics {harmonics}")
    factor = (1.0 if harmonic == 0 else 2.0) / result["sample_number"]
    return factor * coefficients[:, harmonics.index(harmonic), :]


# ---------------------------------------------------------------------------
# Plotted quantities
# ---------------------------------------------------------------------------

def accelerance(result, channel, harmonic):
    """
    FRF from the excitation force to the acceleration at the output sensor [m/s^2/N].

    The excitation is F0*cos(omega t), so its complex amplitude is F0. The
    acceleration of harmonic h is -(h omega)^2 times its displacement amplitude.

    :param result: dict from :func:`load_result`
    :param channel: output sensor channel, 'x', 'y' or 'z'
    :param harmonic: harmonic number h
    :returns: complex FRF, one value per solution point
    """
    stored = [int(c) for c in result["out_channels"]]
    if CHANNELS[channel] not in stored:
        raise ValueError(f"channel {channel} (sensor channel {CHANNELS[channel]}) was not "
                         f"stored, the result holds the sensor channels {stored}")
    u = harmonic_amplitude(result, result["u_out"], harmonic)[:, stored.index(CHANNELS[channel])]
    a = -(harmonic * result["omega"]) ** 2 * u * MM_TO_M
    return a / result["F0"]


PEAK_VALIDATION = {}


def peak_acceleration(result, channel, harmonic=None):
    """max_t |q_ddot_out(t)| / F0, reconstructed using ALL saved harmonics.

    Input coefficients are unscaled rFFT displacement coefficients in mm.
    Differentiate spectrally, convert mm to m, and rescale the inverse FFT
    to a refined phase grid. Refine until every point's peak changes by less
    than 1e-6 relatively. The CSV harmonic field is ignored for this quantity.
    """
    stored = [int(c) for c in result['out_channels']]
    index = stored.index(CHANNELS[channel])
    harmonics = np.asarray(result['harmonics'], dtype=int)
    coefficients = np.asarray(result['u_out'])[:, :, index]
    acceleration = (-coefficients * (np.asarray(result['omega'])[:, None]
                                     * harmonics[None, :])**2 * MM_TO_M)
    if result['F0'] <= 0 or result['sample_number'] <= 0:
        raise ValueError('Positive excitation amplitude and time sample count required')

    def on_grid(samples):
        maxima = []
        for start in range(0, len(acceleration), 64):
            batch = acceleration[start:start+64]
            spectrum = np.zeros((len(batch), samples//2+1), dtype=complex)
            spectrum[:, harmonics] = batch * samples / result['sample_number']
            waveform = np.fft.irfft(spectrum, n=samples, axis=1)
            maxima.extend(np.abs(waveform).max(axis=1))
        return np.asarray(maxima) / result['F0']

    samples = max(2048, 2**int(np.ceil(np.log2(4 * max(harmonics) + 1))))
    previous = on_grid(samples)
    while samples < 65536:
        samples *= 2
        current = on_grid(samples)
        relative = np.max(np.abs(current-previous) / np.maximum(current, np.finfo(float).tiny))
        if relative < 1e-6:
            assert np.isfinite(current).all()
            PEAK_VALIDATION[f'{channel}_F0={result["F0"]:g}'] = dict(
                harmonics=harmonics.tolist(), time_samples=samples,
                max_relative_refinement_change=float(relative),
                peak_normalized_acceleration=float(current.max()))
            return current
        previous = current
    raise ValueError('Peak acceleration did not converge on the refined time grid')


# Spec name -> (function(result, channel, harmonic), y label of the magnitude plot)
QUANTITIES = {
    "peak_acceleration": (peak_acceleration, thesis_style.ACCELERATION_LABEL),
    "accelerance": (accelerance,
                    r"$\omega^2|Q_{\mathrm{r},z,1}/F^{\mathrm{ext}}_{\mathrm{e},1}|$"
                    r"  [$\mathrm{m\,s^{-2}\,N^{-1}}$]"),
}


# ---------------------------------------------------------------------------
# Plot spec
# ---------------------------------------------------------------------------

def auto_labels(results):
    """
    Legend labels from the stored numbers in which the results differ, e.g. ``F0=100``.

    :param results: dict file name -> result dict
    :returns: dict file name -> label; the file stem if nothing differs
    """
    first = next(iter(results.values()))
    numbers = [key for key, value in first.items() if isinstance(value, (int, float))]
    differing = [key for key in numbers
                 if len({result.get(key) for result in results.values()}) > 1]
    return {name: ", ".join(f"{key}={result[key]:g}" for key in differing)
                  or Path(name).stem
            for name, result in results.items()}


def write_style_spec(spec_path, labels):
    """
    Write a spec with one row per result file, every row plotting the default quantity.

    :param spec_path: path of the new plot_styles.csv
    :param labels: dict file name -> label, in plot order
    """
    with open(spec_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(STYLE_COLUMNS)
        for i, (name, label) in enumerate(labels.items()):
            writer.writerow([name, label,
                             DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                             DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)],
                             DEFAULT_LINEWIDTH, DEFAULT_QUANTITY, DEFAULT_CHANNEL,
                             DEFAULT_HARMONIC, ""])


def parse_linestyle(text):
    """A matplotlib line style, or a dash sequence written as '4,2' (on, off in points)."""
    if "," not in text:
        return text
    return (0, tuple(float(value) for value in text.split(",")))


def read_style_spec(spec_path, results, labels):
    """
    The curves one spec asks for, in plot order, with every empty cell filled in.

    :param spec_path: path of the plot_styles.csv
    :param results: dict file name -> result dict of every .npz in the folder
    :param labels: dict file name -> auto label, used where ``label`` is empty
    :returns: list of dicts with the keys file, label, style, quantity, channel,
        harmonic, figure
    """
    with open(spec_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    curves = []
    for line_no, row in enumerate(rows, start=2):           # line 1 is the header
        cell = {key: (row.get(key) or "").strip() for key in STYLE_COLUMNS}
        if not cell["file"] or cell["file"].startswith("#"):
            continue
        where = f"{spec_path}, line {line_no}"
        if cell["file"] not in results:
            raise FileNotFoundError(f"{where}: '{cell['file']}' is not in {spec_path.parent}")

        quantity = cell["quantity"] or DEFAULT_QUANTITY
        channel = cell["channel"] or DEFAULT_CHANNEL
        harmonic = int(cell["harmonic"] or DEFAULT_HARMONIC)
        if quantity not in QUANTITIES:
            raise KeyError(f"{where}: quantity '{quantity}' is none of {list(QUANTITIES)}")
        if channel not in CHANNELS:
            raise KeyError(f"{where}: channel '{channel}' is none of {list(CHANNELS)}")

        i = len(curves)
        curves.append(dict(
            file=cell["file"],
            label=cell["label"] or labels[cell["file"]],
            quantity=quantity, channel=channel, harmonic=harmonic,
            figure=cell["figure"] or f"{quantity}_{channel}_h{harmonic}",
            style=dict(
                color=cell["color"] or DEFAULT_COLORS[i % len(DEFAULT_COLORS)],
                linestyle=parse_linestyle(
                    cell["linestyle"] or DEFAULT_LINESTYLES[i % len(DEFAULT_LINESTYLES)]),
                linewidth=float(cell["linewidth"] or DEFAULT_LINEWIDTH))))

    unlisted = sorted(set(results) - {curve["file"] for curve in curves})
    if unlisted:
        print(f"note: not plotted, not listed in {spec_path.name}: {', '.join(unlisted)}")
    return curves


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def wrapped_phase_deg(values):
    """
    Phase in degrees in (-180, 180]. Where it wraps around, the point is set to
    NaN, so matplotlib breaks the line instead of drawing a vertical jump.
    """
    phase = np.degrees(np.angle(values))
    phase[1:][np.abs(np.diff(phase)) > 180.0] = np.nan
    return phase


def frf_figure(curves, results, title):
    """
    Accelerance magnitude of the curves over the excitation frequency.

    The points are drawn in the order of the continuation path, not sorted by
    frequency: a nonlinear response can fold over, and sorting would zigzag
    between its branches.

    :param curves: entries of :func:`read_style_spec` that share this figure
    :param results: dict file name -> result dict
    :param title: figure title
    :returns: the matplotlib figure
    """
    fig, ax_mag = plt.subplots(figsize=(9, 5.5))
    ylabels = []
    for curve in curves:
        result = results[curve["file"]]
        function, ylabel = QUANTITIES[curve["quantity"]]
        ylabel = ylabel.replace(
            r"Q_{\mathrm{r},z,1}",
            rf"Q_{{\mathrm{{r}},{curve['channel']},{curve['harmonic']}}}")
        if curve["harmonic"] != 1:
            ylabel = ylabel.replace(r"\omega^2", rf"({curve['harmonic']}\omega)^2")
        values = function(result, curve["channel"], curve["harmonic"])
        f = result["freq_hz"]
        ax_mag.plot(f, np.abs(values), label=curve["label"], **curve["style"])
        if ylabel not in ylabels:
            ylabels.append(ylabel)
        peak = np.argmax(np.abs(values))
        print(f"  {curve['label']}: {len(f)} points, {f.min():.1f}..{f.max():.1f} Hz, "
              f"max {np.abs(values)[peak]:.3e} at {f[peak]:.1f} Hz")

    is_peak = all(curve['quantity'] == 'peak_acceleration' for curve in curves)
    ax_mag.set_title("Force-normalized peak acceleration" if is_peak else "Accelerance")
    ax_mag.set_ylabel(" / ".join(ylabels))
    if LOG_Y:
        ax_mag.set_yscale("log")
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend()

    ax_mag.set_xlabel(r"$f=\omega/(2\pi)$ [Hz]")
    if XLIM:
        ax_mag.set_xlim(XLIM)
    thesis_style.finish(fig)
    return fig


def plot_study(folder):
    """
    Every figure of one study folder, as its plot_styles.csv asks for.

    :param folder: study folder inside Results/
    """
    if not folder.is_dir():
        available = sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
        raise NotADirectoryError(f"{folder} does not exist; Results/ holds {available}")
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no study.py results (*.npz) in {folder}")

    results = {path.name: load_result(path) for path in files}
    labels = auto_labels(results)
    spec_path = folder / STYLE_FILE
    if not spec_path.exists():
        write_style_spec(spec_path, labels)
        print(f"created {spec_path} -- edit it to choose curves, labels and styles")

    figures = {}
    for curve in read_style_spec(spec_path, results, labels):
        figures.setdefault(curve["figure"], []).append(curve)
    for name, curves in figures.items():
        print(f"\n{folder.name} / {name}:")
        fig = frf_figure(curves, results, title=f"{folder.name} -- {name}")
        if SAVE:
            fig.savefig(folder / f"{name}.png", dpi=150)
            print(f"  saved {name}.png")


if __name__ == "__main__":
    for study in sys.argv[1:] or [STUDY]:
        plot_study(RESULTS / study)
    plt.show()
