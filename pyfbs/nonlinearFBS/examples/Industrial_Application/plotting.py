"""
FRF-Ergebnisse aus main.py plotten.

    python plotting.py                      -> frf_uy.csv
    python plotting.py frf_uy.csv a.csv     -> mehrere Kurven uebereinander
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

# --- Parameter -------------------------------------------------------------------
FILES = sys.argv[1:] or ["frf_uy.csv"]
LOG_Y = True
SAVE  = None          # z.B. "frf_uy.png", None = nur anzeigen

# --- Plot ------------------------------------------------------------------------
fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

for name in FILES:
    d = np.genfromtxt(HERE / name, delimiter=",", names=True)
    f, mag, phase = d["freq_hz"], d["frf_abs_mm_per_N"], d["frf_phase_rad"]
    ax_mag.plot(f, mag, lw=1.2, label=Path(name).stem)
    ax_ph.plot(f, np.degrees(phase), lw=1.2)
    print(f"{name}: max |H| = {mag.max():.3e} mm/N bei {f[np.argmax(mag)]:.1f} Hz")

ax_mag.set_ylabel("|H|  [mm/N]")
ax_mag.set_title("Anregung ControlArm  ->  Antwort Subframe uy")
if LOG_Y:
    ax_mag.set_yscale("log")
ax_mag.grid(True, which="both", alpha=0.3)
ax_mag.legend()

ax_ph.set_xlabel("Frequenz [Hz]")
ax_ph.set_ylabel("Phase [Grad]")
ax_ph.set_yticks([-180, -90, 0, 90, 180])
ax_ph.grid(True, alpha=0.3)

fig.tight_layout()
if SAVE:
    fig.savefig(HERE / SAVE, dpi=150)
    print("gespeichert:", SAVE)
plt.show()
