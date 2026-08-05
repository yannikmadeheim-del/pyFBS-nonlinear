"""
Calibrate the cubic damping coefficients of runs D1-D3 from the B0 baseline.

Fixed beta guesses are not defensible: the damping force beta*xdot^3 scales
with the CUBE of the interface gap velocity, and the translational (m/s) and
rotational (rad/s) gaps live on entirely different scales. Instead, beta is
anchored to what B0 actually does:

    beta_mod   = 0.10 * f_star / v_star^3      (peak damping force ~10% of the
    beta_stark = 0.50 * f_star / v_star^3       linear interface force scale)

with, per DoF group (trans/rot), v_star = max gap velocity and f_star = max
linear interface force |k * x_gap| along the B0 branch -- both taken from the
<gapstats> CSV the study runners export. The pyhbm (reference) run is the
primary source; the pyFBS gapstats are printed alongside for comparison.

Patches rows D1-D3 of params_study.csv in place and prints the values.
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PARAMS_CSV = HERE / "params_study.csv"

FRACTION_MOD = 0.10
FRACTION_STARK = 0.50


def read_gapstats(path):
    with open(path, newline="", encoding="utf-8") as fh:
        row = next(csv.DictReader(fh))
    return {k: float(v) for k, v in row.items()}


def main():
    primary = RESULTS / "B0_gapstats_pyhbm.csv"
    g = read_gapstats(primary)
    print(f"B0 gap statistics ({primary.name}):")
    for k, v in g.items():
        print(f"  {k:20s} = {v:.4e}")
    secondary = RESULTS / "B0_gapstats_pyfbs.csv"
    if secondary.exists():
        print(f"for comparison ({secondary.name}):")
        for k, v in read_gapstats(secondary).items():
            print(f"  {k:20s} = {v:.4e}")

    beta = {}
    for grp, unit in (("trans", "N s^3/m^3"), ("rot", "Nm s^3/rad^3")):
        v_star = g[f"v_{grp}_max_m_s" if grp == "trans" else f"v_{grp}_max_rad_s"]
        f_star = g[f"f_lin_{grp}_max_N" if grp == "trans" else f"f_lin_{grp}_max_Nm"]
        beta[grp, "mod"] = FRACTION_MOD * f_star / v_star**3
        beta[grp, "stark"] = FRACTION_STARK * f_star / v_star**3
        print(f"beta_{grp}: v* = {v_star:.4e}, f* = {f_star:.4e}  ->  "
              f"mod = {beta[grp, 'mod']:.3e}, stark = {beta[grp, 'stark']:.3e} [{unit}]")

    with open(PARAMS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    patch = {
        "D1": {"beta_trans": beta["trans", "mod"]},
        "D2": {"beta_trans": beta["trans", "stark"], "beta_rot": beta["rot", "stark"]},
        "D3": {"beta_rot": beta["rot", "mod"]},
    }
    for row in rows:
        for key, value in patch.get(row["run_id"], {}).items():
            row[key] = f"{value:.3e}"
    with open(PARAMS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"patched D1-D3 in {PARAMS_CSV.name}")


if __name__ == "__main__":
    main()
