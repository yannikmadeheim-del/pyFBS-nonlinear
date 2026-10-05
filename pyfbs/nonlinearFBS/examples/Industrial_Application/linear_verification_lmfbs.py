"""
Linear verification of the Subframe + ControlArm coupling against LM-FBS.

With alpha = beta = 0 the bushing law is linear, so the coupled admittance
follows in closed form from LM-FBS with relaxed compatibility (thesis
eq. 2.11). Channels and impacts are not collocated, so compatibility acts on
the channels (Bc) and equilibrium on the impacts (Bf):

    Y_tilde = Y - Y Bf^T (Bc Y Bf^T + Gamma)^-1 Bc Y,   Gamma = diag(1 / (k + i omega c))

The nonlinear FBS runs use the settings of study.py on the same admittance Y,
the same Bc and Bf and the same bushing parameters. The comparison therefore
checks the coupling formulation and its implementation, not the FE model or
the VPT.

The reference is evaluated twice:

    grid    every line of the FRF grid in F_LO..F_HI, from ds.Y directly;
            this is the plotted curve
    points  every solution frequency of the nonlinear run, from the spline
            interpolation of ds.Y that the solver uses; the error metrics use
            this one, so the interpolation does not enter the comparison

Run from this folder, like study.py::

    python linear_verification_lmfbs.py

Everything is written to Results/<STUDY_NAME>/: the nonlinear runs and their
manifest.csv in the format of study.py, LM-FBS_reference.npz,
verification_summary.json and the figure as .pdf and .png. If the
verification passes, the .pdf is also copied into the thesis.
"""

import csv
import json
import shutil
import sys
import time
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dynamical_system as ds
import study
from plot_study import CHANNELS, MM_TO_M, harmonic_amplitude, load_result
from pyfbs.nonlinearFBS import ExperimentalFRF, HarmonicBalanceMethod

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import thesis_plot_style as style

# --- Settings --------------------------------------------------------------------
STUDY_NAME = "linear_verification_LMFBS_23.09.2026"

# Bushing parameters of the parameter studies in SI, with the cubic terms switched
# off. The key order must be the one of study.SWEEP (file names, manifest columns).
BUSHING = dict(
    K_TRANS     = 2e5,      # N/m
    K_ROT       = 1e2,      # N*m/rad
    Alpha_TRANS = 0.0,      # N/m^3
    Alpha_ROT   = 0.0,      # N*m/rad^3
    C_TRANS     = 4e2,      # N*s/m
    C_ROT       = 2e-1,     # N*m*s/rad
    Beta_TRANS  = 0.0,      # N*s^3/m^3
    Beta_ROT    = 0.0,      # N*m*s^3/rad^3
)
F0 = 1000                   # N, the verified and plotted run
F0_CHECK = 10               # N, second run for the linearity check

CHANNEL = "z"               # output: Sensor 7z on the subframe (group 10)
OUT = CHANNELS[CHANNEL]     # 14 in the channel numbering
INP = ds.inp_full           # 24: Impact 19 on the control arm (group 20)

# Acceptance: largest relative deviation from the point reference, and largest
# ratio |U_h| / |U_1| of the higher harmonics at the output channel
MAX_REL_DEVIATION = 1e-5
MAX_HARMONIC_RATIO = 1e-8
# ---------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "Results" / STUDY_NAME
THESIS_FIGURE = (HERE.parents[5] / "thesis" / "figures" / "Industrial_solverPlots"
                 / "linear_LMFBS_verification.pdf")
EQUATION = ("Y_tilde = Y - Y Bf^T (Bc Y Bf^T + Gamma)^-1 Bc Y, "
            "Gamma = diag(1 / (k + i omega c))")


# ---------------------------------------------------------------------------
# Nonlinear FBS runs
# ---------------------------------------------------------------------------

def run(params, provider):
    """
    Solve and save one parameter combination exactly as study.py does. An
    existing result is kept, as in study.py.

    :param params: one value per study.SWEEP key, in the same order
    :param provider: FRF provider shared by both runs
    :returns: path of the .npz
    """
    name = study.run_name(params)
    path = OUT_DIR / f"{name}.npz"
    if path.exists():
        print(f"{name}: skipped (result exists)")
        return path

    print(f"\n{name}")
    manifest = OUT_DIR / "manifest.csv"
    t0 = time.perf_counter()
    try:
        problem, ss = study.solve(params, provider)
        omega = study.save(path, params, problem, ss)
    except Exception as exc:
        study.append_manifest(manifest, [name, *params.values(), "failed", 0, "",
                                         f"{time.perf_counter() - t0:.1f}",
                                         f"{type(exc).__name__}: {exc}"])
        raise
    runtime = time.perf_counter() - t0
    f_min = omega.min() / (2 * np.pi)
    study.append_manifest(manifest, [name, *params.values(), "ok", len(omega),
                                     f"{f_min:.2f}", f"{runtime:.1f}", ""])
    print(f"[{name}] ok: {len(omega)} points down to {f_min:.1f} Hz in {runtime:.0f} s")
    return path


def manifest_runtimes():
    """Solver runtime [s] of every successful run, from manifest.csv."""
    with open(OUT_DIR / "manifest.csv", newline="", encoding="utf-8") as fh:
        return {row["name"]: float(row["runtime_s"])
                for row in csv.DictReader(fh) if row["status"] == "ok"}


# ---------------------------------------------------------------------------
# LM-FBS reference
# ---------------------------------------------------------------------------

def lm_fbs_column(Y, omega, k, c):
    """
    Excited column of the coupled admittance with relaxed compatibility,

        Y_tilde = Y - Y Bf^T (Bc Y Bf^T + Gamma)^-1 Bc Y,   Gamma = diag(1 / (k + i omega c))

    evaluated for every frequency line at once. Only the column INP is
    needed, so Bc Y[:, INP] is the right-hand side instead of Bc Y.

    :param Y: uncoupled admittance [mm/N], shape (n_freq, N_out, N_in)
    :param omega: angular frequencies [rad/s], shape (n_freq,)
    :param k: bushing stiffness per interface DoF in model units, shape (12,)
    :param c: bushing damping per interface DoF in model units, shape (12,)
    :returns: Y_tilde[:, :, INP] [mm/N], shape (n_freq, N_out)
    """
    Gamma = np.zeros((len(omega), ds.N_IF, ds.N_IF), dtype=complex)
    Gamma[:, np.arange(ds.N_IF), np.arange(ds.N_IF)] = 1.0 / (k + 1j * omega[:, None] * c)
    Y_int = ds.Bc @ Y @ ds.Bf.T                                        # (n_freq, 12, 12)
    # interface forces per unit excitation force
    lam = np.linalg.solve(Y_int + Gamma, ds.Bc @ Y[:, :, INP, None])   # (n_freq, 12, 1)
    return Y[:, :, INP] - (Y @ ds.Bf.T @ lam)[:, :, 0]


def admittance_at(provider, omega):
    """
    Uncoupled admittance at the given frequencies, interpolated by the
    provider exactly as the solver interpolates it.

    :param provider: ExperimentalFRF of the runs
    :param omega: angular frequencies [rad/s], shape (n,)
    :returns: admittance [mm/N], shape (n, N_out, N_in)
    """
    all_out, all_in = np.arange(ds.N_out), np.arange(ds.N_in)
    return np.array([provider.compute_FRF(w, [1], all_out, all_in)[0] for w in omega])


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def output_column(result):
    """Position of the output channel OUT among the stored sensor channels."""
    return [int(ch) for ch in result["out_channels"]].index(OUT)


def receptance(result):
    """
    Harmonic-1 receptance from the excitation to the output channel [m/N],
    one value per solution point, in continuation order.

    :param result: dict from plot_study.load_result
    """
    U1 = harmonic_amplitude(result, result["u_out"], 1)[:, output_column(result)]
    return U1 / result["F0"] * MM_TO_M


def harmonic_ratio(result):
    """Largest |U_h| / |U_1| at the output channel over all points and all h >= 3."""
    U = {h: np.abs(harmonic_amplitude(result, result["u_out"], h)[:, output_column(result)])
         for h in result["harmonics"]}
    return float(max((U[h] / U[1]).max() for h in U if h >= 3))


def deviation(f, H, H_ref):
    """
    Deviation of a nonlinear FBS receptance from the LM-FBS point reference.

    :param f: solution frequencies [Hz]
    :param H: nonlinear FBS receptance at f
    :param H_ref: LM-FBS receptance at f
    :returns: (relative deviation per point, dict of summary numbers)
    """
    eps_rel = np.abs(H - H_ref) / np.abs(H_ref)
    # Normalized by the largest reference value instead of the local one, so a
    # small |H_ref| near an antiresonance does not inflate it
    eps_norm = np.abs(H - H_ref) / np.abs(H_ref).max()
    dphi = np.angle(H / H_ref)                         # wrapped to (-pi, pi]
    i = int(np.argmax(eps_rel))
    return eps_rel, dict(
        eps_rel_max=float(eps_rel[i]),
        eps_rel_max_at_hz=float(f[i]),
        eps_rel_median=float(np.median(eps_rel)),
        eps_rel_points_below_machine_eps=int((eps_rel < np.finfo(float).eps).sum()),
        eps_norm_max=float(eps_norm.max()),
        phase_diff_max_abs_rad=float(np.abs(dphi).max()),
        phase_diff_max_abs_deg=float(np.degrees(np.abs(dphi).max())))


def iteration_stats(result):
    """Newton iterations per solution point: range, mean and how often each count occurs."""
    it = np.asarray(result["iterations"], dtype=int)
    return dict(min=int(it.min()), max=int(it.max()), mean=float(it.mean()),
                points_per_count={str(n): int(m) for n, m in enumerate(np.bincount(it)) if m})


def linearity(f, H, H_ref, f_check, H_check, H_ref_check):
    """
    Largest relative deviation between the receptances of the two force
    levels, at the frequencies f of the main run inside the frequency range
    of both runs.

    The runs have different solution frequencies, so H_check is interpolated
    onto f. Its ratio to the LM-FBS reference is interpolated instead of
    H_check itself: the ratio is close to one everywhere, whereas the resonant
    receptance varies too fast between continuation points for a linear
    interpolation. The direct interpolation is reported for comparison.

    :returns: dict of summary numbers
    """
    order = np.argsort(f_check)                    # np.interp needs ascending f
    lo, hi = max(f.min(), f_check.min()), min(f.max(), f_check.max())
    inside = (f >= lo) & (f <= hi)

    def interpolate(values):
        return (np.interp(f[inside], f_check[order], values[order].real)
                + 1j * np.interp(f[inside], f_check[order], values[order].imag))

    H_via_ratio = interpolate(H_check / H_ref_check) * H_ref[inside]
    lin = np.abs(H_via_ratio - H[inside]) / np.abs(H[inside])
    direct = np.abs(interpolate(H_check) - H[inside]) / np.abs(H[inside])
    i = int(np.argmax(lin))
    return dict(max_rel_deviation=float(lin[i]),
                max_at_hz=float(f[inside][i]),
                median_rel_deviation=float(np.median(lin)),
                points_compared=int(inside.sum()),
                overlap_hz=[float(lo), float(hi)],
                max_rel_deviation_direct_interpolation=float(direct.max()))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def log_limits(freq, values, pad=0.08):
    """Limits of a log axis that fit every value inside F_LO..F_HI."""
    inside = (freq >= study.F_LO) & (freq <= study.F_HI)
    lo, hi = np.log10(values[inside].min()), np.log10(values[inside].max())
    margin = max(pad * (hi - lo), 0.08)
    return 10 ** (lo - margin), 10 ** (hi + margin)


def figure(f, H, freq_grid, H_grid):
    """
    Accelerance of the nonlinear FBS run and of the LM-FBS grid reference,
    omega^2 |Y| from the receptances H and H_grid [m/N].

    The legend is typeset in LaTeX, so the figure has none.
    """
    A = (2 * np.pi * f) ** 2 * np.abs(H)
    A_grid = (2 * np.pi * freq_grid) ** 2 * np.abs(H_grid)
    for values in (A, A_grid):
        assert np.isfinite(values).all() and (values > 0).all(), \
            "every plotted value must be finite and positive"

    style.configure()
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    # pyFBS first and LM-FBS on top, so both stay visible where they coincide;
    # the pyFBS points keep their continuation order
    ax.semilogy(f, A, color="black", linestyle="-", linewidth=2)
    ax.semilogy(freq_grid, A_grid, color="#2ca02c", linestyle="-.", linewidth=1.2)
    ax.set(xlim=(study.F_LO, study.F_HI),
           ylim=log_limits(np.r_[f, freq_grid], np.r_[A, A_grid]),
           xlabel=style.FREQUENCY_LABEL,
           ylabel=r"$\omega^2|Y_{\mathrm{out,e}}(\omega)|$ [$\mathrm{m\,s^{-2}\,N^{-1}}$]")
    style.finish(fig)
    return fig


# ---------------------------------------------------------------------------

def main():
    if any(BUSHING[key] != 0.0 for key in ("Alpha_TRANS", "Alpha_ROT", "Beta_TRANS", "Beta_ROT")):
        raise ValueError("the LM-FBS reference exists only for a linear bushing law")
    # Beyond the FRF grid ExperimentalFRF only warns and extrapolates the spline
    if max(study.HARMONICS) * study.F_HI >= ds.freq[-1]:
        raise ValueError(f"max(HARMONICS) * F_HI = {max(study.HARMONICS) * study.F_HI:g} Hz "
                         f"lies beyond the FRF grid ({ds.freq[-1]:g} Hz)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- nonlinear FBS, as study.py ---
    provider = ExperimentalFRF(ds.omega, ds.Y)
    HarmonicBalanceMethod.update_dependencies(study.HARMONICS, study.SAMPLE_NUMBER)   # before FBSProblem

    results, names = {}, {}
    for f0 in (F0, F0_CHECK):
        params = dict(BUSHING, F0=f0)
        assert list(params) == list(study.SWEEP), "BUSHING must follow the key order of study.SWEEP"
        names[f0] = study.run_name(params)
        results[f0] = load_result(run(params, provider))

    # --- LM-FBS, grid and points ---
    bushing = ds.bushing_parameters_from_si(BUSHING["K_TRANS"], BUSHING["K_ROT"],
                                            BUSHING["C_TRANS"], BUSHING["C_ROT"])
    k, c = bushing["k"], bushing["c"]

    # the grid is an arange; the tolerance keeps lines that sit on F_LO or F_HI
    tol = 1e-6 * ds.f_resolution
    keep = (ds.freq >= study.F_LO - tol) & (ds.freq <= study.F_HI + tol)
    freq_grid = ds.freq[keep]
    H_grid = lm_fbs_column(ds.Y[keep], ds.omega[keep], k, c)[:, OUT]              # mm/N

    H_pts = {f0: lm_fbs_column(admittance_at(provider, result["omega"]),
                               result["omega"], k, c)[:, OUT]                     # mm/N
             for f0, result in results.items()}

    np.savez(OUT_DIR / "LM-FBS_reference.npz",
             freq_hz=freq_grid, H_grid=H_grid,
             freq_pts=results[F0]["freq_hz"], H_pts=H_pts[F0],
             freq_pts_check=results[F0_CHECK]["freq_hz"], H_pts_check=H_pts[F0_CHECK],
             units="mm/N", out_channel=OUT, inp=INP, k=k, c=c,
             equation=EQUATION, **BUSHING)

    # --- comparison in m/N ---
    f = {f0: result["freq_hz"] for f0, result in results.items()}
    H = {f0: receptance(result) for f0, result in results.items()}
    H_ref = {f0: H_pts[f0] * MM_TO_M for f0 in results}
    runtimes = manifest_runtimes()

    runs = {}
    for f0, result in results.items():
        _, numbers = deviation(f[f0], H[f0], H_ref[f0])
        runs[f"F0={f0:g}"] = dict(
            file=f"{names[f0]}.npz", n_points=len(f[f0]),
            f_min_hz=float(f[f0].min()), f_max_hz=float(f[f0].max()),
            runtime_s=runtimes.get(names[f0]),
            newton_iterations=iteration_stats(result),
            harmonic_ratio_max=harmonic_ratio(result), **numbers)

    main_run = runs[f"F0={F0:g}"]
    passed = (main_run["eps_rel_max"] <= MAX_REL_DEVIATION
              and main_run["harmonic_ratio_max"] <= MAX_HARMONIC_RATIO)

    summary = dict(
        study=STUDY_NAME, equation=EQUATION,
        output=dict(channel=CHANNEL, index=OUT), excitation=dict(index=INP),
        bushing_si=BUSHING,
        bushing_model_units=dict(k=k.tolist(), c=c.tolist(),
                                 units="k: N/mm, N*mm/rad; c: N*s/mm, N*mm*s/rad"),
        admittance=dict(provider="ExperimentalFRF", no_modes=ds.no_modes,
                        modal_damping=ds.modal_damping,
                        static_correction=ds.static_correction,
                        f_resolution_hz=ds.f_resolution),
        solver=dict(harmonics=study.HARMONICS, sample_number=study.SAMPLE_NUMBER,
                    f_lo_hz=study.F_LO, f_hi_hz=study.F_HI, **study.SOLVER_KWARGS),
        reference=dict(grid_lines=len(freq_grid),
                       grid_hz=[float(freq_grid[0]), float(freq_grid[-1])]),
        runs=runs,
        linearity=linearity(f[F0], H[F0], H_ref[F0],
                            f[F0_CHECK], H[F0_CHECK], H_ref[F0_CHECK]),
        acceptance=dict(max_rel_deviation=MAX_REL_DEVIATION,
                        max_harmonic_ratio=MAX_HARMONIC_RATIO,
                        applies_to=f"F0={F0:g}", passed=bool(passed)))

    (OUT_DIR / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit("verification FAILED -- see verification_summary.json; "
                         "no figure written, nothing copied into the thesis")

    # --- figure ---
    fig = figure(f[F0], H[F0], freq_grid, H_grid * MM_TO_M)
    pdf = OUT_DIR / "linear_LMFBS_verification.pdf"
    fig.savefig(pdf)
    fig.savefig(pdf.with_suffix(".png"), dpi=160)
    plt.close(fig)
    print(f"figure: {pdf}")

    if THESIS_FIGURE.parent.is_dir():
        shutil.copy2(pdf, THESIS_FIGURE)
        print(f"copied to {THESIS_FIGURE}")
    else:
        print(f"note: {THESIS_FIGURE.parent} does not exist, figure not copied")


if __name__ == "__main__":
    main()
