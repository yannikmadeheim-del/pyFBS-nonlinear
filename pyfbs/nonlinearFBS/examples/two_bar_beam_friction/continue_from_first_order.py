"""DLFT contact + AFT friction, seeded from a converged first-order solution.

The cold start in :func:`main.initial_guess` does not reach the ``dlft_aft``
branch at the circular-movement parameters (P1 = P2 = 0.1, P5 = 0.02): the
transverse drive is five times the static clamp, so the joint separates for
about 40% of the period and the semismooth Newton chatters instead of
converging.

A converged branch for the SAME parameters already exists, traced with the
first-order formulation. This script takes ONE point from that result, solves
the rigid DLFT problem at that frequency, and hands the converged point to the
ordinary arc-length continuation, which traces the branch on its own in both
directions. Nothing but the first point comes from the other solver.

Results are written exactly like ``study.py`` writes them: one CSV per run in
``results/<STUDY_NAME>/`` plus a ``manifest.csv`` row.

Run with::

    python continue_from_first_order.py
    python continue_from_first_order.py --seed-omega 1.2 --results results
"""
import argparse
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from pyfbs.nonlinearFBS import (
    DLFTContactAFT, FBSProblem, Fourier, Fourier_Real, FourierOmegaPoint,
    HarmonicBalanceMethod, SolutionSet,
)

from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import main as run
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import plotting_saving as io
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import study
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction.dynamical_system import (
    BarBeamParams, IF_LABELS, L_F, L_N, N_IF, SYSTEM_TYPES,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
STUDY_NAME = "DLFT_from_first_order"
RUN_ID     = "000_dlft_aft"

# The branch to seed from. Its parameters must be the ones in CONFIG["params"];
# only the harmonic list is machine-checked, the rest is on you.
SEED_CSV   = io.RESULTS / "thesis_DLFT_circular_movement" / "first_order.csv"

# Frequency whose first-order point seeds the continuation. Not every point of
# the reference branch is a usable seed: the regularized branch is only close to
# the rigid one where the contact is actually closed. omega = 0.5 is NOT usable
# -- there the reference solution never touches (x_N max 0.00805 < eps 0.01)
# while its softplus still carries the full contact force, and the rigid solve
# fails for every dlft_epsilon between 0.01 and 100. omega = 0.9 converges in 19
# iterations at dlft_epsilon = 10.
SEED_OMEGA = 0.9
SEED_SCAN  = True       # on failure, look for a usable seed near SEED_OMEGA

CONFIG = dict(
    # BarBeamParams defaults ARE the first_order.csv parameters: l=1, EA=EI=1/3,
    # lambda=1, k=500, alpha2=2, eps=0.01, alpha1=150, mu=0.1, P1=P2=0.1,
    # P5=0.02, beta=0.05. k and alpha2 are inert for the rigid DLFT contact.
    params = BarBeamParams().to_dict(),
    solver = "pyfbs-nlfbs",
    method = "dlft_aft",
    dlft_epsilon = 5.0,

    harmonics = list(range(0, 26)),  # harmonic 0 carries the static clamp P5
    sample_number = 1000,
    omega_lo = 0.5, omega_hi = 2.0, sweep = "up",
    omega_resolution = 0.002,
    frf_source = "modal",
    parameterization = "ArcLengthParameterization",
    predictor = "TangentPredictorBordered",
    step_adaptation = "ExponentialAdaptation",
    solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-5},
    step_kwargs = {"base": 2.0, "initial_step_length": 0.01,
                   "maximum_step_length": 0.05, "minimum_step_length": 1e-9,
                   "goal_number_of_iterations": 4},
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
    inert_params = ["alpha2", "k"],
)

# The continuation runs in chunks so a branch that stops advancing in omega can
# be abandoned instead of running to maximum_number_of_solutions. A stalled
# predictor still "converges" every point, so point count alone never ends it.
CHUNK_POINTS = 500
MAX_CHUNKS   = CONFIG["maximum_number_of_solutions"] // CHUNK_POINTS
STALL_DOMEGA = 1e-3     # per chunk; below this the direction is abandoned
# ---------------------------------------------------------------------------


def read_seed_branch(path, harmonics):
    """Interface harmonics of the reference branch, as physical amplitudes.

    :returns: (omega, a) with omega (n,) ascending and a (n, Nh, N_IF) complex,
        where u(t) = Re(sum_h a[:, h] exp(1j h omega t)).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"seed branch not found: {path}")

    # The coefficient vector length is fixed by the harmonic list, so a mismatch
    # would silently seed the solver with a differently-shaped state.
    header = io.read_header(path)
    stated = next((line.split(":", 1)[1] for line in header
                   if line.strip().startswith("harmonics:")), None)
    if stated is not None:
        stated = [int(h) for h in stated.split("]")[0].strip(" [").split(",")]
        if stated != list(harmonics):
            raise ValueError(
                f"{path.name} was solved with harmonics {stated[0]}..{stated[-1]} "
                f"but this run uses {harmonics[0]}..{harmonics[-1]}; reconcile "
                f"CONFIG['harmonics'] with the seed branch before continuing.")

    columns = ["omega_rad_s"] + [f"{part}_h{h}_{label}" for h in harmonics
                                 for label in IF_LABELS for part in ("re", "im")]
    table = (pd.read_csv(path, comment="#", usecols=columns)
               .sort_values("omega_rad_s").reset_index(drop=True))

    a = np.empty((len(table), len(harmonics), N_IF), dtype=complex)
    for i, h in enumerate(harmonics):
        for j, label in enumerate(IF_LABELS):
            a[:, i, j] = (table[f"re_h{h}_{label}"].to_numpy()
                          + 1j * table[f"im_h{h}_{label}"].to_numpy())
    return table["omega_rad_s"].to_numpy(), a


def seed_point(a_row, omega, harmonics):
    """A reference row as a FourierOmegaPoint for the current Fourier setup.

    The CSV stores physical amplitudes a_h; the solver state is the raw rFFT
    coefficient c_h = a_h * N_t / (1 for h = 0, else 2). N_t is THIS run's
    sample count, not the one the reference branch was solved with: a_h does not
    depend on it, c_h scales linearly with it.
    """
    n_t = Fourier.number_of_time_samples
    scale = np.array([n_t / (1.0 if h == 0 else 2.0) for h in harmonics])
    return FourierOmegaPoint(Fourier((a_row * scale[:, None])[:, :, None]), float(omega))


def build(cfg, provider):
    """The problem and continuation driver for one configuration."""
    HarmonicBalanceMethod.update_dependencies(list(cfg["harmonics"]),
                                              cfg["sample_number"])
    p = run.params_of(cfg)
    system = SYSTEM_TYPES[cfg["method"]](p, sample_number=cfg["sample_number"])
    problem = FBSProblem(system, provider,
                         DLFTContactAFT(L_N, L_F, epsilon=cfg["dlft_epsilon"],
                                        g_zero=p.eps))
    solver = HarmonicBalanceMethod(
        harmonics=list(cfg["harmonics"]), freq_domain_ode=problem,
        corrector_parameterization=run.SOLVER_PARTS[cfg["parameterization"]],
        predictor=run.SOLVER_PARTS[cfg["predictor"]],
        step_length_adaptation=run.SOLVER_PARTS[cfg["step_adaptation"]])
    return system, problem, solver


def solve_seed(solver, cfg, omega_grid, amplitudes, omega_target, scan):
    """Rigid-contact solution at the seed frequency, or at the nearest usable one.

    ``solve_and_continue`` would start with this solve anyway; doing it here
    separates "the seed was unusable" from "the continuation stopped early", and
    lets a failed seed be retried at another frequency without tracing anything.

    :returns: (converged point, seed omega, iterations) or (None, None, None).
    """
    harmonics = list(cfg["harmonics"])
    solver_kwargs = dict(cfg["solver_kwargs"])
    # solve_and_continue applies this rescaling internally; solve_fixed_frequency
    # does not, so the seed must be held to the same tolerance by hand
    solver_kwargs["absolute_tolerance"] *= np.sqrt(2) / Fourier.number_of_time_samples

    targets = [omega_target]
    if scan:
        coarse = np.linspace(omega_grid[0], omega_grid[-1], 31)
        targets += sorted(coarse, key=lambda w: abs(w - omega_target))

    tried = set()
    for target in targets:
        index = int(np.abs(omega_grid - target).argmin())
        omega = float(omega_grid[index])
        if omega in tried:
            continue
        tried.add(omega)

        guess = seed_point(amplitudes[index], omega, harmonics)
        solution, iterations, success, _ = solver.solve_fixed_frequency(
            guess, **solver_kwargs)
        if success:
            return solution, omega, iterations
        print(f"   seed at omega = {omega:.4f}: no convergence in {iterations} iterations")
        if not scan:
            break
    return None, None, None


def trace(solver, cfg, seed, direction):
    """One sweep direction from ``seed``, in chunks, abandoned when omega stalls.

    :returns: (points, reason) with points a list of
        (Fourier, omega, iterations, step_length) in continuation order,
        starting at the seed.
    """
    reference = FourierOmegaPoint.new_from_first_harmonic(
        np.zeros((N_IF, 1), complex), omega=direction)
    guess, points, reason = seed, [], "solution cap"

    for chunk in range(MAX_CHUNKS):
        omega_before = guess.omega
        solutions = solver.solve_and_continue(
            initial_guess                 = guess,
            initial_reference_direction   = reference,
            maximum_number_of_solutions   = CHUNK_POINTS,
            # sorted in place by the solver, so it gets a fresh list every chunk
            angular_frequency_range       = [cfg["omega_lo"], cfg["omega_hi"]],
            solver_kwargs                 = dict(cfg["solver_kwargs"]),
            step_length_adaptation_kwargs = dict(cfg["step_kwargs"]),
            jacobian_update_frequency     = cfg["jacobian_update_frequency"],
            verbose                       = False,
        )
        # every chunk after the first restarts on its predecessor's last point
        first = 0 if chunk == 0 else 1
        points += list(zip(solutions.fourier[first:], solutions.omega[first:],
                           solutions.iterations[first:],
                           solutions.step_length[first:]))
        advance = abs(solutions.omega[-1] - omega_before)
        print(f"   chunk {chunk + 1}: {len(solutions.omega)} points, "
              f"omega {omega_before:.4f} -> {solutions.omega[-1]:.4f}")

        if len(solutions.omega) < CHUNK_POINTS:
            reason = "branch ended (window edge, predictor or corrector stop)"
            break
        if advance < STALL_DOMEGA:
            reason = (f"stalled: omega advanced {advance:.2e} over {CHUNK_POINTS} "
                      f"points, below STALL_DOMEGA = {STALL_DOMEGA:g}")
            break
        guess = FourierOmegaPoint(
            Fourier(solutions.fourier[-1].coefficients.copy()),
            float(solutions.omega[-1]))

    return points, reason


def merge(downward, upward):
    """The two sweeps as one SolutionSet, in continuation order through the seed.

    Both sweeps start at the shared seed, so the downward one is reversed and
    contributes it; the upward one drops it. Points are NOT sorted by omega --
    a fold would interleave the two sides of the branch and scramble it.
    """
    ordered = list(reversed(downward)) + upward[1:]
    fourier, omega, iterations, step_length = zip(*ordered)

    merged = SolutionSet(FourierOmegaPoint(fourier[0], omega[0]),
                         iterations[0], step_length[0])
    for f, w, its, step in zip(fourier[1:], omega[1:], iterations[1:], step_length[1:]):
        merged.append(FourierOmegaPoint(f, w), its, step)
    return merged


def report_contact(problem, solutions, eps):
    """Contact fraction and residual penetration along the branch.

    A rigid contact should sit exactly on the gap. What the branch actually
    carries is the truncation error of a discontinuous contact force over a
    finite harmonic set, and it belongs in the run log rather than in the CSV
    alone.
    """
    fractions, penetrations = [], []
    for fourier, omega in zip(solutions.fourier, solutions.omega):
        point = FourierOmegaPoint(fourier, omega)
        problem.method.compute_F_int(point, problem.ode)
        fractions.append(float(point.contact_mask[:, 0, 0].mean()))
        Fourier_Real.compute_time_series(fourier)
        penetrations.append(float(fourier.time_series[:, 0, 0].max() - eps))
    worst = max(penetrations)
    print(f"   contact {100 * min(fractions):.0f}%..{100 * max(fractions):.0f}%"
          f" of the period; max penetration {worst:+.2e}"
          f" = {100 * worst / eps:.1f}% of the gap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="results folder name in this example folder")
    parser.add_argument("--seed-omega", type=float, default=SEED_OMEGA,
                        help=f"frequency of the seeding point (default {SEED_OMEGA})")
    parser.add_argument("--seed-csv", default=str(SEED_CSV),
                        help="reference branch to seed from")
    args = parser.parse_args()

    out_dir = HERE / args.results / STUDY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"
    csv_path = out_dir / f"{RUN_ID}.csv"

    cfg = deepcopy(CONFIG)
    p = run.params_of(cfg)
    collapsed = ", ".join(f"{key}={cfg['params'][key]}" for key in cfg["inert_params"])
    varied = (f"method={cfg['method']}, dlft_epsilon={cfg['dlft_epsilon']}, "
              f"seed_omega={args.seed_omega}")

    print(f"study '{STUDY_NAME}' -> {out_dir}")
    if csv_path.exists():
        print(f"{RUN_ID}: skipped (result exists)")
        return

    print(f"\nseed branch: {Path(args.seed_csv).name}")
    omega_grid, amplitudes = read_seed_branch(args.seed_csv, cfg["harmonics"])
    print(f"   {len(omega_grid)} points, omega {omega_grid[0]:.4f} .. {omega_grid[-1]:.4f}")

    print(f"\nbuilding {cfg['frf_source']} FRF provider "
          f"(omega grid to {run.synthesis_omega_end(cfg):g} rad/s) ...")
    t0 = time.perf_counter()
    provider = run.build_provider(cfg, cfg["frf_source"])
    system, problem, solver = build(cfg, provider)
    print(f"build done in {time.perf_counter() - t0:.1f} s")

    print(f"\n[{RUN_ID}] {varied}   [ignored by method: {collapsed}]")
    t0 = time.perf_counter()
    try:
        seed, omega_seed, iterations = solve_seed(
            solver, cfg, omega_grid, amplitudes, args.seed_omega, SEED_SCAN)
        if seed is None:
            raise RuntimeError(
                f"no usable seed found near omega = {args.seed_omega} at "
                f"dlft_epsilon = {cfg['dlft_epsilon']}; try another "
                f"dlft_epsilon or another seed frequency")
        print(f"   seed: omega = {omega_seed:.4f}, converged in {iterations} iterations")

        print("   downward:")
        downward, why_down = trace(solver, cfg, seed, -1.0)
        print("   upward:")
        upward, why_up = trace(solver, cfg, seed, +1.0)
        solutions = merge(downward, upward)
        solve_time = time.perf_counter() - t0

        omega = np.asarray(solutions.omega)
        print(f"   branch: {len(omega)} points, omega {omega.min():.4f} .. {omega.max():.4f}")
        print(f"   downward stop: {why_down}")
        print(f"   upward stop:   {why_up}")
        report_contact(problem, solutions, p.eps)

        io.save_solution(csv_path, cfg, solutions, problem, system,
                         cfg["method"], cfg["frf_source"], solve_time)
        runtime = time.perf_counter() - t0
        status = "stalled" if "stalled" in (why_down + why_up) else "ok"
        study.append_manifest(manifest, [RUN_ID, cfg["method"], varied, collapsed,
                                         status, len(omega), f"{runtime:.1f}",
                                         csv_path.name, ""])
        print(f"[{RUN_ID}] {status}: {len(omega)} points in {runtime:.1f} s")
        print(f"solution written: {csv_path}")
    except Exception as exc:
        runtime = time.perf_counter() - t0
        study.append_manifest(manifest, [RUN_ID, cfg["method"], varied, collapsed,
                                         "failed", 0, f"{runtime:.1f}", "",
                                         f"{type(exc).__name__}: {exc}"])
        print(f"[{RUN_ID}] FAILED after {runtime:.1f} s: {exc}")
        traceback.print_exc()
        return

    print(f"\nstudy '{STUDY_NAME}' finished -- manifest: {manifest}")
    print(f"plot with:  python -m pyfbs.nonlinearFBS.examples"
          f".two_bar_beam_friction.plot_comparison {STUDY_NAME}")


if __name__ == "__main__":
    main()
