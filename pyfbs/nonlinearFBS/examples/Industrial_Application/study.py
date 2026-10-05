"""
Parameter study for the Subframe + ControlArm bushing coupling.

Every combination of the value lists in SWEEP is solved with the continuation
settings of main.py. Results go to Results/<STUDY_NAME>/: one .npz per combination,
named after its parameters (tags in TAGS), plus a manifest.csv with the status of
every run. Combinations whose .npz already exists are skipped, so an interrupted
study resumes when it is started again with the same STUDY_NAME.

Each .npz holds the raw Fourier coefficients produced by the solver, unscaled:

    q_rel  (n, Nh, 12)  relative displacement [VP1 ux uy uz rx ry rz | VP2 ...]
    u_out  (n, Nh, len(OUT_CH))   response at the output sensor channels OUT_CH (stored as out_channels)

n is the number of solution points, Nh the number of harmonics (see ``harmonics``).
The FRF from the excitation point to the output sensor is u_out / F0.
Read a result back with ``np.load(path)``.
"""

import csv
import itertools
import time
import traceback
from pathlib import Path

import numpy as np

import dynamical_system as ds
from pyfbs.nonlinearFBS import (
    Fourier, FourierOmegaPoint, FBSProblem, ExperimentalFRF, AFT,
    HarmonicBalanceMethod, ArcLengthParameterization, TangentPredictorBordered,
)

# --- Study -----------------------------------------------------------------------
STUDY_NAME = "higherValues_cubicDamping_study_18.09.2026"

# Every parameter takes a list of values; the study solves all combinations.
# Bushing parameters are given in SI; ds.bushing_parameters_from_si converts them
# to the mm units of the FE model.
SWEEP = dict(
    K_TRANS     = [2e5],           # N/m
    K_ROT       = [1e2],           # N*m/rad
    Alpha_TRANS = [-5e9],          # N/m^3
    Alpha_ROT   = [-5e2],            # N*m/rad^3
    C_TRANS     = [4e2],           # N*s/m
    C_ROT       = [2e-1],          # N*m*s/rad
    Beta_TRANS  = [1e3, 1e5, 1e7],          # N*s^3/m^3
    Beta_ROT    = [1e-3, 1e-1, 1e1],         # N*m*s^3/rad^3
    F0          = [1000],   # N at the excitation point
)

# Short tags for the result file names, e.g. KT100000_KR2.5e+06_..._F0100.npz
TAGS = dict(K_TRANS="KT", K_ROT="KR", Alpha_TRANS="AT", Alpha_ROT="AR",
            C_TRANS="CT", C_ROT="CR", Beta_TRANS="BT", Beta_ROT="BR", F0="F0")

# --- Solver settings (as in main.py) ---------------------------------------------
F_LO, F_HI = 1.0, 1000.0          # continuation window [Hz]
HARMONICS = [1, 3, 5, 7, 9]
SAMPLE_NUMBER = 37
OUT_CH = [12, 13, 14]              # output sensor, group 10: channels 12, 13, 14 = x, y, z

SOLVER_KWARGS = {"maximum_iterations": 200, "absolute_tolerance": 1e-8}
STEP_KWARGS = {"base": 2.0, "initial_step_length": 0.01, "maximum_step_length": 5,
               "minimum_step_length": 1e-6, "goal_number_of_iterations": 3}
MAXIMUM_NUMBER_OF_SOLUTIONS = 10000
# ---------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
MANIFEST_COLUMNS = ["name", *SWEEP, "status", "n_points", "f_min_hz", "runtime_s", "error"]


def run_name(params):
    """File name of one combination, e.g. ``KT100000_KR2.5e+06_..._F0100``."""
    return "_".join(f"{TAGS[key]}{value:g}" for key, value in params.items())


def solve(params, provider):
    """Solve one combination with a downward sweep from F_HI to F_LO.

    :param params: one value per SWEEP key
    :param provider: FRF provider shared by all runs
    :returns: (problem, solution set)
    """
    w_lo, w_hi = 2 * np.pi * F_LO, 2 * np.pi * F_HI
    bushing = ds.bushing_parameters_from_si(
        params["K_TRANS"], params["K_ROT"], params["C_TRANS"], params["C_ROT"],
        params["Alpha_TRANS"], params["Alpha_ROT"], params["Beta_TRANS"], params["Beta_ROT"])
    system = ds.Subframe_ControlArm(**bushing, F0=params["F0"], sample_number=SAMPLE_NUMBER)

    problem = FBSProblem(system, provider, AFT())
    solver = HarmonicBalanceMethod(harmonics=HARMONICS, freq_domain_ode=problem,
                                   corrector_parameterization=ArcLengthParameterization,
                                   predictor=TangentPredictorBordered)

    # Start at w_hi; the reference direction omega = -1 makes the sweep go downwards
    ig = FourierOmegaPoint.zero_amplitude(dimension=ds.N_IF, omega=w_hi)
    rd = FourierOmegaPoint.new_from_first_harmonic(np.zeros((ds.N_IF, 1), complex), omega=-1.0)

    # Copies of the kwargs, so the solver cannot carry changes over into the next run
    ss = solver.solve_and_continue(
        initial_guess=ig, initial_reference_direction=rd,
        maximum_number_of_solutions=MAXIMUM_NUMBER_OF_SOLUTIONS,
        angular_frequency_range=[w_lo, w_hi],
        solver_kwargs=dict(SOLVER_KWARGS),
        step_length_adaptation_kwargs=dict(STEP_KWARGS),
        jacobian_update_frequency=1, verbose=True)
    return problem, ss


def save(path, params, problem, ss):
    """Write the raw solver Fourier coefficients and the settings of one run to ``path``."""
    q_rel = np.array([four.coefficients[:, :, 0] for four in ss.fourier])      # (n, Nh, 12)

    # The Newton unknown is only the interface gap; compute_full_response recovers
    # the response of all channels from it, of which the output sensor is kept.
    u_out = np.array([problem.compute_full_response(four, w).coefficients[:, OUT_CH, 0]
                      for four, w in zip(ss.fourier, ss.omega)])               # (n, Nh, 3)

    omega = np.asarray(ss.omega, dtype=float)
    np.savez(path,
             freq_hz=omega / (2 * np.pi), omega=omega,
             harmonics=np.array([int(h) for h in Fourier.harmonics]),
             q_rel=q_rel, u_out=u_out, out_channels=np.array(OUT_CH),
             iterations=np.asarray(ss.iterations),
             step_length=np.asarray(ss.step_length),
             sample_number=Fourier.number_of_time_samples,
             f_lo=F_LO, f_hi=F_HI,
             maximum_number_of_solutions=MAXIMUM_NUMBER_OF_SOLUTIONS,
             **{f"solver_{key}": value for key, value in SOLVER_KWARGS.items()},
             **{f"step_{key}": value for key, value in STEP_KWARGS.items()},
             **params)
    return omega


def append_manifest(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(row)


def main():
    # Beyond the FRF grid ExperimentalFRF only warns and extrapolates the spline,
    # which gives meaningless admittances for the higher harmonics.
    if max(HARMONICS) * F_HI >= ds.freq[-1]:
        raise ValueError(f"max(HARMONICS) * F_HI = {max(HARMONICS) * F_HI:g} Hz lies beyond "
                         f"the FRF grid ({ds.freq[-1]:g} Hz, f_end in dynamical_system.py)")

    out_dir = HERE / "Results" / STUDY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    runs = [dict(zip(SWEEP, values)) for values in itertools.product(*SWEEP.values())]
    print(f"study '{STUDY_NAME}': {len(runs)} combination(s) -> {out_dir}")

    # Y does not depend on the swept parameters, so one provider serves every run
    provider = ExperimentalFRF(ds.omega, ds.Y)
    HarmonicBalanceMethod.update_dependencies(HARMONICS, SAMPLE_NUMBER)   # before FBSProblem

    for i, params in enumerate(runs, start=1):
        name = run_name(params)
        path = out_dir / f"{name}.npz"
        if path.exists():
            print(f"[{i}/{len(runs)}] {name}: skipped (result exists)")
            continue

        print(f"\n[{i}/{len(runs)}] {name}")
        t0 = time.perf_counter()
        try:
            problem, ss = solve(params, provider)
            omega = save(path, params, problem, ss)
            runtime = time.perf_counter() - t0
            f_min = omega.min() / (2 * np.pi)
            append_manifest(manifest, [name, *params.values(), "ok", len(omega),
                                       f"{f_min:.2f}", f"{runtime:.1f}", ""])
            print(f"[{name}] ok: {len(omega)} points down to {f_min:.1f} Hz in {runtime:.0f} s")
        except Exception as exc:
            runtime = time.perf_counter() - t0
            append_manifest(manifest, [name, *params.values(), "failed", 0, "",
                                       f"{runtime:.1f}", f"{type(exc).__name__}: {exc}"])
            print(f"[{name}] FAILED after {runtime:.0f} s: {exc}")
            traceback.print_exc()

    print(f"\nstudy '{STUDY_NAME}' finished -- manifest: {manifest}")


if __name__ == "__main__":
    main()
