"""
Parameter study, pyFBS side: run the cubic-spring FBS coupling (ModalVPFRF
provider) for every row of params_study.csv and export the forced-response
branch of each run as CSV.

The expensive testbench build (ANSYS modes -> VPT -> admittance) is
spring-independent, so it happens ONCE; per run only the k/alpha/beta
diagonals are swapped and the AFT + arc-length HBM continuation is repeated
with the exact solver settings of ../main.py (run_nfrc).

Outputs per run (results/):
  <ID>_pyfbs.csv          freq_hz,omega_rad_s,amp_m,amp_h1_m  (branch order)
  <ID>_linear_pyfbs.csv   freq_hz,receptance_m_per_N          (LM-FBS backbone)
  <ID>_gapstats_pyfbs.csv interface-gap amplitude/velocity/force maxima
plus one status line per run appended to results/manifest_pyfbs.csv.

Runs are skipped when their result CSV already exists (resume) or when a
parameter is NaN (beta placeholders before the B0 calibration).
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent           # .../testbench_cubicSpring/parameter_study
EXAMPLE_DIR = HERE.parent
RESULTS = HERE / "results"
PARAMS_CSV = HERE / "params_study.csv"

sys.path.insert(0, str(EXAMPLE_DIR))
os.chdir(EXAMPLE_DIR)                            # build_testbench_data uses ./lab_testbench/...

try:                                             # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

import numpy as np

import pyfbs
from pyfbs.nonlinearFBS import (
    Fourier_Real, FourierOmegaPoint, FBSProblem, ModalVPFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)
from dynamical_system import build_testbench_data, TestbenchCubicSpring, N_IF

# fixed study settings -- mirror ../main.py; only the six spring parameters vary
F0 = 50.0
F_RESOLUTION = 1.0        # coarse grid suffices: only the linear backbone reads it,
                          # ModalVPFRF synthesizes at the exact n*omega anyway
HARMONICS = [1, 3, 5, 7]
SAMPLE_NUMBER = 256       # TestbenchCubicSpring default AFT sampling
F_LO, F_HI = 20.0, 1000.0
W_LO, W_HI = 2.0 * np.pi * F_LO, 2.0 * np.pi * F_HI


def read_params(only=None):
    """Rows of params_study.csv as dicts with float parameters, file order."""
    with open(PARAMS_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("k_trans", "k_rot", "alpha_trans", "alpha_rot",
                    "beta_trans", "beta_rot"):
            r[key] = float(r[key])
    if only:
        keep = {s.strip() for s in only.split(",")}
        rows = [r for r in rows if r["run_id"] in keep]
    return rows


def manifest_line(run_id, params, status, n_points, runtime_s, error=""):
    path = RESULTS / "manifest_pyfbs.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run_id", "solver", "k_trans", "k_rot", "alpha_trans",
                        "alpha_rot", "beta_trans", "beta_rot", "status",
                        "n_points", "runtime_s", "error"])
        w.writerow([run_id, "pyfbs",
                    params["k_trans"], params["k_rot"], params["alpha_trans"],
                    params["alpha_rot"], params["beta_trans"], params["beta_rot"],
                    status, n_points, f"{runtime_s:.1f}", error])


def linear_backbone(data, k_diag):
    """
    Linear (alpha=0) LM-FBS spring receptance |Y_out,inp| on the synthesis grid,
    restricted to 20..1000 Hz -- the formula of ../main.py step 2, but coupling
    only the interface DoFs with k > 0 (a zero-stiffness DoF is simply
    uncoupled; Gamma = 1/k would blow up on it).
    """
    freq, Y, Bc = data["freq"], data["Y"], data["Bc"]
    act = np.nonzero(k_diag > 0)[0]
    Ba = Bc[act]
    Gamma = np.zeros((len(freq), len(act), len(act)))
    d = np.arange(len(act))
    Gamma[:, d, d] = 1.0 / k_diag[act]

    BY = Ba @ Y
    YBt = Y @ Ba.T
    Yint = Ba @ YBt
    Y_spring = Y - YBt @ pyfbs.tpinv(Yint + Gamma, trunc=0) @ BY

    sel = (freq >= F_LO) & (freq <= F_HI)
    mag = np.abs(Y_spring[:, data["out_full"], data["inp_full"]])
    return freq[sel], mag[sel]


def solve_run(row, data, provider):
    """One AFT + arc-length HBM continuation; identical settings to run_nfrc."""
    data["k_diag"] = np.array([row["k_trans"]] * 3 + [row["k_rot"]] * 3)
    data["alpha_diag"] = np.array([row["alpha_trans"]] * 3 + [row["alpha_rot"]] * 3)
    data["beta_diag"] = np.array([row["beta_trans"]] * 3 + [row["beta_rot"]] * 3)

    system = TestbenchCubicSpring(data, F0=F0)
    problem = FBSProblem(system, provider, AFT())
    solver = HarmonicBalanceMethod(
        harmonics=HARMONICS, freq_domain_ode=problem,
        corrector_parameterization=ArcLengthParameterization,
        predictor=TangentPredictorBordered)

    ig = FourierOmegaPoint.zero_amplitude(dimension=N_IF, omega=W_HI)
    rd = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((N_IF, 1), complex), omega=-1.0)

    ss = solver.solve_and_continue(
        initial_guess                 = ig,
        initial_reference_direction   = rd,
        maximum_number_of_solutions   = 10000,
        angular_frequency_range       = [W_LO, W_HI],
        solver_kwargs                 = {"maximum_iterations": 300,
                                         "absolute_tolerance": 1e-6},
        step_length_adaptation_kwargs = {"base": 4.0,
                                         "initial_step_length": 0.1 * 2 * np.pi,
                                         "maximum_step_length": 5 * 2 * np.pi,
                                         "minimum_step_length": 1e-7,
                                         "goal_number_of_iterations": 3},
        jacobian_update_frequency     = 1,
    )
    return system, problem, ss


def export_run(run_id, row, data, system, problem, ss):
    """
    Branch CSV in the reference format plus interface-gap statistics.

    amp_m    = peak |u_out(t)| over one period (the quantity ../main.py plots)
    amp_h1_m = first-harmonic amplitude of u_out, via rfft over the uniform
               one-period time grid (2|U_1|/Nt -- same convention as the pyhbm
               reference export)
    Gap statistics feed the beta calibration: the time series of the 6
    relative interface DoFs x_r = Bc u and its physical velocity (spectral
    differentiation: harmonic j of the tau grid oscillates at j*omega).
    """
    Bc = data["Bc"]
    n = len(ss.omega)
    freq = np.asarray(ss.omega) / (2.0 * np.pi)
    amp = np.empty(n)
    amp_h1 = np.empty(n)
    x_t = x_r = v_t = v_r = 0.0                          # branch maxima

    for i, (four, w) in enumerate(zip(ss.fourier, ss.omega)):
        full = problem.compute_full_response(four, w)
        Fourier_Real.compute_time_series(full)
        ts = full.time_series[:, :, 0]                   # (Nt, N) one period
        nt = ts.shape[0]

        u_out = ts[:, system.out_full]
        amp[i] = float(np.max(np.abs(u_out)))
        amp_h1[i] = 2.0 * abs(np.fft.rfft(u_out)[1]) / nt

        x = ts @ Bc.T                                    # (Nt, 6) interface gap
        X = np.fft.rfft(x, axis=0)
        V = X * (1j * np.arange(X.shape[0]) * w)[:, None]
        xdot = np.fft.irfft(V, n=nt, axis=0)             # physical gap velocity
        x_t = max(x_t, float(np.max(np.abs(x[:, :3]))))
        x_r = max(x_r, float(np.max(np.abs(x[:, 3:]))))
        v_t = max(v_t, float(np.max(np.abs(xdot[:, :3]))))
        v_r = max(v_r, float(np.max(np.abs(xdot[:, 3:]))))

    np.savetxt(RESULTS / f"{run_id}_pyfbs.csv",
               np.c_[freq, np.asarray(ss.omega), amp, amp_h1],
               delimiter=",", comments="",
               header="freq_hz,omega_rad_s,amp_m,amp_h1_m")

    with open(RESULTS / f"{run_id}_gapstats_pyfbs.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_trans_max_m", "x_rot_max_rad", "v_trans_max_m_s",
                    "v_rot_max_rad_s", "f_lin_trans_max_N", "f_lin_rot_max_Nm"])
        w.writerow([x_t, x_r, v_t, v_r,
                    row["k_trans"] * x_t, row["k_rot"] * x_r])
    return n


def main():
    global RESULTS, PARAMS_CSV, F_RESOLUTION, F_LO, F_HI, W_LO, W_HI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="comma-separated run ids (default: all rows)")
    ap.add_argument("--params", help="params CSV filename in this folder "
                    "(default: params_study.csv)")
    ap.add_argument("--results", help="results subfolder name (default: results)")
    ap.add_argument("--resolution", type=float,
                    help="linear FRF grid resolution Delta f [Hz] (default: 1.0)")
    ap.add_argument("--flo", type=float, help="window lower bound [Hz] (default: 20)")
    ap.add_argument("--fhi", type=float, help="window upper bound [Hz] (default: 1000)")
    args = ap.parse_args()

    if args.params:
        PARAMS_CSV = HERE / args.params
    if args.results:
        RESULTS = HERE / args.results
    if args.resolution:
        F_RESOLUTION = args.resolution
    if args.flo is not None:
        F_LO = args.flo
    if args.fhi is not None:
        F_HI = args.fhi
    W_LO, W_HI = 2.0 * np.pi * F_LO, 2.0 * np.pi * F_HI

    RESULTS.mkdir(exist_ok=True)
    print(f"window {F_LO:g}..{F_HI:g} Hz | Delta f {F_RESOLUTION:g} Hz | "
          f"params {PARAMS_CSV.name} | results {RESULTS.name}")
    rows = read_params(args.only)
    print(f"pyFBS study: {len(rows)} run(s): {', '.join(r['run_id'] for r in rows)}")

    print("building testbench data once (spring-independent) ...")
    t0 = time.perf_counter()
    data = build_testbench_data(1.0e3, 1.0e3, 1.0e6, 1.0e6, 0.0, 0.0,
                                f_resolution=F_RESOLUTION)
    provider = ModalVPFRF.from_substructures(
        [(data["MK_A"], data["vpt_A"], data["df_chn_A"], data["df_imp_A"]),
         (data["MK_B"], data["vpt_B"], data["df_chn_B"], data["df_imp_B"])],
        modal_damping=data["modal_damping"])
    HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)
    print(f"build done in {time.perf_counter() - t0:.0f} s")

    for row in rows:
        run_id = row["run_id"]
        out_csv = RESULTS / f"{run_id}_pyfbs.csv"
        if out_csv.exists():
            print(f"[{run_id}] skipped (result exists)")
            continue
        if any(math.isnan(row[k]) for k in ("k_trans", "k_rot", "alpha_trans",
                                            "alpha_rot", "beta_trans", "beta_rot")):
            print(f"[{run_id}] skipped (NaN parameter -- beta not calibrated yet)")
            manifest_line(run_id, row, "skipped_nan", 0, 0.0)
            continue

        print(f"[{run_id}] k=({row['k_trans']:g},{row['k_rot']:g}) "
              f"alpha=({row['alpha_trans']:g},{row['alpha_rot']:g}) "
              f"beta=({row['beta_trans']:g},{row['beta_rot']:g})")
        t0 = time.perf_counter()
        try:
            k_diag = np.array([row["k_trans"]] * 3 + [row["k_rot"]] * 3)
            f_lin, mag_lin = linear_backbone(data, k_diag)
            np.savetxt(RESULTS / f"{run_id}_linear_pyfbs.csv",
                       np.c_[f_lin, mag_lin], delimiter=",", comments="",
                       header="freq_hz,receptance_m_per_N")

            system, problem, ss = solve_run(row, data, provider)
            n = export_run(run_id, row, data, system, problem, ss)
            dt = time.perf_counter() - t0
            manifest_line(run_id, row, "ok", n, dt)
            print(f"[{run_id}] ok: {n} branch points in {dt:.0f} s")
        except Exception as exc:
            dt = time.perf_counter() - t0
            manifest_line(run_id, row, "failed", 0, dt,
                          f"{type(exc).__name__}: {exc}")
            print(f"[{run_id}] FAILED after {dt:.0f} s: {exc}")
            traceback.print_exc()

    print("pyFBS study finished.")


if __name__ == "__main__":
    main()
