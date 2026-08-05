"""
Overlay the joint diagnostics of one or more physical-solution CSVs at ONE
excitation frequency -- no solver, no ANSYS: everything is reconstructed from
the exported harmonic amplitudes (pyFBS nonlinearFBS exports and any pyhbm
reference following the same convention, e.g. for solver validation).

Usage: type the CSVs into CSV_FILES below and set TARGET_FREQ_HZ, or drag &
drop the files onto this script / pass them as command-line arguments (a bare
number among the arguments is taken as the target frequency). Bare names are
resolved relative to this folder. Per file the converged point NEAREST the
target frequency is used; the actual frequencies are printed and shown in the
legend. The harmonic set is detected per file from its columns, so files with
different HARMONICS settings can be compared directly.

Figure (one color per CSV):
  top          NFRC context: ||a_1|| of the tangential VP gap [ux, uy] over
               each file's whole branch + a marker at the compared point,
  middle left  hysteresis loop, friction force vs gap at the compared point,
  middle right force-velocity law, friction force vs gap velocity,
  bottom       one period of gap (left axis) and friction force (right axis),
               plotted over the period fraction t/T.
All tangential quantities are projected on the slip direction e of the FIRST
file (major axis of its 1st-harmonic gap ellipse), so every file shares the
same projection. The friction force is re-evaluated from the law

    f_T = 2*mu_trans*N*tanh(alpha*||v_T||) * v_T/||v_T||

with the parameters below (NOT from the file headers): identical law for all
files, only the kinematics differ -- that is the quantity a solver comparison
should agree on.

A second window shows the output-channel forced response (uout = A_S1X), like
main.py's figure 1: top max_t |u_out(t)| over one period, bottom |a_1|, overlaid
per file with a marker at the compared point.

3D view (SHOW_3D): the undeformed A/B meshes (STL, context only, mm -> m) with
one animated sphere per exported sensor/VP position and its closed orbit over
one period, magnified by one common scale; one color per CSV. Positions and
directions are parsed from the CSV comment header; files without that metadata
reuse the positions of the first file that has it (matched by label). VP
rotations are not drawn.

The gap is taken from x_rel_* where present (pyFBS export); otherwise it is
built as A_vp_* - B_vp_* (pyhbm convention), exactly like plot_comparison.py.
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Inputs: CSVs to overlay and the frequency to compare at. Command-line
# arguments override both (file paths and/or one bare number = frequency).
# ---------------------------------------------------------------------------
CSV_FILES      = ["reference_modal_fbs_hbm.csv",
                  "reference_rbe2_cb_dryfriction_hbm.csv",
                  "reference_rbe3_cb_dryfriction_hbm.csv"]
TARGET_FREQ_HZ = 580

# joint law used to re-evaluate the friction force from each file's kinematics
mu_trans = 0.3        # friction coefficient [-]
N        = 200.0      # clamping force [N] -> slip force 2*mu_trans*N
alpha    = 1.0e3      # tanh regularization sharpness [s/m]

SHOW_3D = True        # pyvista window with animated sensor/VP orbits
R_SCALE = 0.08        # 3D: max animated displacement as fraction of the model diagonal
N_T     = 720         # time samples of one period for the reconstructions

HERE   = os.path.dirname(os.path.abspath(__file__))
TRANS  = ("ux", "uy")
OUT    = "A_S1X"      # output channel for the forced-response window (main.py fig. 1)
COLORS = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]

_num = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
args = sys.argv[1:]
if args:
    files = [a for a in args if not _num.match(a)]
    nums  = [float(a) for a in args if _num.match(a)]
    if files:
        CSV_FILES = files
    if nums:
        TARGET_FREQ_HZ = nums[-1]

# ---------------------------------------------------------------------------
# CSV loading: header metadata, per-file harmonic set, gap + channel amplitudes.
# ---------------------------------------------------------------------------
_META = re.compile(r"^#\s+(\S+): '.*' grouping .* at \(([^)]+)\) m, "
                   r"direction \[([^\]]+)\]")


def load_csv(path):
    """One file -> dict with the branch NFRC, the point nearest the target
    frequency (complex amplitudes per DoF) and the channel position metadata."""
    meta = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _META.match(line)
            if m:
                meta[m.group(1)] = tuple(
                    np.array([float(x) for x in grp.split(",")])
                    for grp in (m.group(2), m.group(3)))
    df = pd.read_csv(path, comment="#")

    hset = sorted({int(m.group(1)) for c in df.columns
                   for m in [re.match(r"re_h(\d+)_", c)] if m})
    dofs = [c[len(f"re_h{hset[0]}_"):] for c in df.columns
            if c.startswith(f"re_h{hset[0]}_")]

    def amp(dof, idx=None):
        """Complex amplitudes a_h of one DoF: (n_points, Nh) or (Nh,) at idx."""
        a = (df[[f"re_h{h}_{dof}" for h in hset]].to_numpy()
             + 1j * df[[f"im_h{h}_{dof}" for h in hset]].to_numpy())
        return a if idx is None else a[idx]

    def gap(dof, idx=None):
        if f"x_rel_{dof}" in dofs:                       # pyFBS export
            return amp(f"x_rel_{dof}", idx)
        return amp(f"A_vp_{dof}", idx) - amp(f"B_vp_{dof}", idx)   # pyhbm

    freq = df["freq_hz"].to_numpy()
    i = int(np.argmin(np.abs(freq - TARGET_FREQ_HZ)))
    a_T = np.stack([gap(d, i) for d in TRANS])           # (2, Nh) at the point
    nfrc = np.sqrt(sum(np.abs(gap(d)[:, hset.index(1)]) ** 2 for d in TRANS))

    # channel DoFs for the 3D view (skip the gap columns and VP rotations)
    chn = [d for d in dofs if not d.startswith("x_rel_") and "_vp_r" not in d]
    a_chn = {d: amp(d, i) for d in chn}

    # output-channel forced response over the WHOLE branch (uout = OUT):
    # max_t |u_out(t)| and |a_1|, mirroring main.py's forced-response figure.
    if OUT in dofs:
        a_out = amp(OUT)                                          # (n, Nh)
        th = np.linspace(0.0, 2.0 * np.pi, N_T, endpoint=False)
        sig = np.real(a_out @ np.exp(1j * np.outer(hset, th)))    # (n, N_T)
        out_max, out_h1 = np.abs(sig).max(axis=1), np.abs(a_out[:, hset.index(1)])
    else:
        out_max = out_h1 = None

    return dict(name=os.path.splitext(os.path.basename(path))[0],
                freq=freq, nfrc=nfrc, i=i, f=freq[i],
                omega=2.0 * np.pi * freq[i], hset=hset,
                a_T=a_T, a_chn=a_chn, meta=meta,
                out_max=out_max, out_h1=out_h1)


def period_signal(a, hset, theta, dfactor=None):
    """u(theta) = Re( sum_h a_h e^{1j h theta} ) on the given theta grid;
    dfactor (e.g. 1j*h*omega) turns it into the corresponding velocity."""
    h = np.asarray(hset, dtype=float)
    E = np.exp(1j * np.outer(h, theta))                  # (Nh, N_T)
    fac = np.ones_like(h) if dfactor is None else dfactor
    return np.real((np.atleast_2d(a) * fac[None, :]) @ E)


def friction_force(vT):
    """f_T = 2*mu*N*tanh(alpha*||v||)*v/||v|| per time sample, (2, N_T)."""
    g = np.linalg.norm(vT, axis=0)
    s = np.where(g > 1e-12, np.tanh(alpha * g) / np.maximum(g, 1e-30), alpha)
    return (2.0 * mu_trans * N) * s[None, :] * vT


entries, theta = [], np.linspace(0.0, 2.0 * np.pi, N_T, endpoint=False)
for name in CSV_FILES:
    path = name if os.path.isabs(name) else os.path.join(HERE, name)
    e = load_csv(path)
    h = np.asarray(e["hset"], dtype=float)
    e["x_T"] = period_signal(e["a_T"], e["hset"], theta)                   # (2, N_T) [m]
    e["v_T"] = period_signal(e["a_T"], e["hset"], theta, 1j * h * e["omega"])
    e["f_T"] = friction_force(e["v_T"])
    entries.append(e)
    print(f"{e['name']}: {len(e['freq'])} points, compared at {e['f']:.3f} Hz "
          f"(target {TARGET_FREQ_HZ:g}), harmonics {e['hset']}")

# common slip direction e: major axis of the FIRST file's h1 gap ellipse
a1 = entries[0]["a_T"][:, entries[0]["hset"].index(1)]
e_dir = np.linalg.svd(np.stack([a1.real, a1.imag], axis=1))[0][:, 0]
Fs = 2.0 * mu_trans * N
print(f"slip direction e = [{e_dir[0]:+.3f}, {e_dir[1]:+.3f}] (from "
      f"{entries[0]['name']}), slip force 2*mu*N = {Fs:.1f} N")

# ---------------------------------------------------------------------------
# Figure: NFRC context + point diagnostics, one color per file.
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 11), layout="constrained")
gs  = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.0, 0.9])
ax0 = fig.add_subplot(gs[0, :])
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1])
ax3 = fig.add_subplot(gs[2, :])
ax3r = ax3.twinx()

for e, c in zip(entries, COLORS):
    lab = f"{e['name']} ({e['f']:.1f} Hz)"
    xs, vs, fs = e_dir @ e["x_T"], e_dir @ e["v_T"], e_dir @ e["f_T"]
    ax0.semilogy(e["freq"], 1e6 * e["nfrc"], color=c, lw=1.1, label=lab)
    ax0.plot(e["f"], 1e6 * e["nfrc"][e["i"]], "v", color=c, ms=7)
    ax1.plot(1e6 * xs, fs, color=c, lw=1.0)
    ax2.plot(1e3 * vs, fs, color=c, lw=1.0)
    ax3.plot(theta / (2 * np.pi), 1e6 * xs, color=c, lw=1.2, label=f"gap, {lab}")
    ax3r.plot(theta / (2 * np.pi), fs, color=c, lw=0.9, ls="--")

ax0.axvline(TARGET_FREQ_HZ, color="gray", lw=0.8, ls=":")
ax0.set_xlabel("frequency [Hz]")
ax0.set_ylabel("||a_1|| tangential gap  [um]")
ax0.set_title(f"Tangential VP gap NFRC -- compared at {TARGET_FREQ_HZ:g} Hz "
              f"(mu = {mu_trans:g}, N = {N:g} N, alpha = {alpha:g} s/m)")
ax0.legend(fontsize=8)

for ax in (ax1, ax2):
    ax.axhline(+Fs, color="gray", ls="--", lw=0.8)
    ax.axhline(-Fs, color="gray", ls="--", lw=0.8)
ax1.set_xlabel("gap along e  [um]")
ax1.set_ylabel("friction force along e  [N]")
ax1.set_title("hysteresis loop")
ax2.set_xlabel("gap velocity along e  [mm/s]")
ax2.set_ylabel("friction force along e  [N]")
ax2.set_title("force-velocity law  2 mu N tanh(alpha ||v_T||)")

ax3.set_xlabel("fraction of one period  t/T")
ax3.set_ylabel("gap along e  [um]")
ax3r.set_ylabel("friction force along e  [N]  (dashed)")
ax3.legend(fontsize=8, loc="upper right")
for ax in (ax0, ax1, ax2, ax3):
    ax.grid(True, which="both", alpha=0.3)


# ---------------------------------------------------------------------------
# Separate window: output-channel forced response (uout = OUT), like main.py's
# figure 1 -- top max_t|u_out(t)|, bottom |a_1|, overlaying every file.
# ---------------------------------------------------------------------------
def output_channel_figure(entries):
    have = [e for e in entries if e.get("out_max") is not None]
    if not have:
        print(f"output-channel window skipped: '{OUT}' not in any CSV")
        return
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                         layout="constrained")
    for e, c in zip(have, COLORS):
        lab = f"{e['name']} ({e['f']:.1f} Hz)"
        ax_top.semilogy(e["freq"], e["out_max"], color=c, lw=1.1, label=lab)
        ax_top.plot(e["f"], e["out_max"][e["i"]], "v", color=c, ms=7)
        ax_bot.semilogy(e["freq"], e["out_h1"], color=c, lw=1.1, label=lab)
        ax_bot.plot(e["f"], e["out_h1"][e["i"]], "v", color=c, ms=7)
    for ax in (ax_top, ax_bot):
        ax.axvline(TARGET_FREQ_HZ, color="gray", lw=0.8, ls=":")
        ax.grid(True, which="both", alpha=0.3)
    ax_top.set_ylabel("max_t |u_out(t)|  [m]")
    ax_top.set_title(f"Output channel {OUT} -- forced response "
                     f"(marker = compared point, {TARGET_FREQ_HZ:g} Hz)")
    ax_top.legend(fontsize=8)
    ax_bot.set_ylabel(f"|a_1({OUT})|  [m]")
    ax_bot.set_xlabel("frequency [Hz]")


output_channel_figure(entries)


# ---------------------------------------------------------------------------
# 3D view: undeformed meshes + animated sensor/VP spheres with their orbits.
# ---------------------------------------------------------------------------
def orbits_3d(entry, fallback_meta):
    """Per physical point: position and its 3D displacement orbit (3, N_T),
    summed over the point's directional channels. Labelled channels without
    position metadata fall back to `fallback_meta` (matched by label)."""
    meta = {**fallback_meta, **entry["meta"]}
    groups = {}
    for dof, a in entry["a_chn"].items():
        if dof not in meta:
            continue
        pos, dvec = meta[dof]
        key = tuple(np.round(pos, 6))
        u = period_signal(a, entry["hset"], theta)[0]              # (N_T,)
        base, orb = groups.setdefault(key, (pos, np.zeros((3, N_T))))
        orb += dvec[:, None] * u[None, :]
    return list(groups.values())


def show_3d(entries, off_screen=False, screenshot=None):
    import pyvista as pv

    fallback = next((e["meta"] for e in entries if e["meta"]), {})
    sets = [orbits_3d(e, fallback) for e in entries]
    umax = max((np.linalg.norm(orb, axis=0).max() for pts in sets
                for _, orb in pts), default=0.0)
    if umax == 0.0:
        print("3D view skipped: no channel positions found in any CSV header")
        return
    scale = R_SCALE * 0.62 / umax          # 0.62 m = A+B bounding-box diagonal

    pl = pv.Plotter(off_screen=off_screen, window_size=(1400, 950))
    pl.set_background("white")
    for stl, colr in (("A", "#B4B2A9"), ("B", "#5DCAA5")):
        p = os.path.join(HERE, "lab_testbench", "STL", stl + ".stl")
        if os.path.exists(p):
            pl.add_mesh(pv.read(p).scale(1e-3), color=colr, opacity=0.22)

    movers = []
    for (e, c), pts in zip(zip(entries, COLORS), sets):
        base = np.array([b for b, _ in pts])
        orbs = np.stack([o for _, o in pts])                      # (n_pts, 3, N_T)
        for b, o in pts:                                          # closed orbit curves
            ring = (b[None, :] + scale * o.T)
            pl.add_mesh(pv.lines_from_points(np.vstack([ring, ring[:1]]), close=False),
                        color=c, line_width=2)
        cloud = pv.PolyData(base + scale * orbs[:, :, 0])
        pl.add_points(cloud, color=c, point_size=14, render_points_as_spheres=True,
                      label=f"{e['name']} ({e['f']:.1f} Hz)")
        movers.append((cloud, base, orbs))

    pl.add_legend(bcolor="white", loc="upper left",
                  size=(0.42, 0.05 * len(entries)))
    pl.add_text(f"displacement scale x {scale:.3g}", position="lower_left",
                font_size=9, color="black")

    frame = {"i": 0}
    def tick(step):
        frame["i"] = (frame["i"] + 4) % N_T
        for cloud, base, orbs in movers:
            cloud.points = base + scale * orbs[:, :, frame["i"]]
        pl.render()

    if off_screen:
        tick(0)
        pl.screenshot(screenshot)
        pl.close()
        return
    pl.add_timer_event(max_steps=10 ** 8, duration=30, callback=tick)
    pl.show()


if SHOW_3D:
    plt.show(block=False)      # figure and 3D window up together
    plt.pause(0.2)
    show_3d(entries)
plt.show()
