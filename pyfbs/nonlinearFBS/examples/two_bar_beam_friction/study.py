"""
Parameter permutations of the bar+beam contact joint, one result CSV per run.

The joint is FIXED here -- normal contact plus friction, always both -- so there
is nothing to compose: what varies is the parameter block. Edit the top block:

  * BASE     -- the defaults, a CONFIG exactly like main.py's.
  * PARAMS   -- the physical parameter block. ANY entry may be written as a LIST
                of values; the study then runs every combination of those lists.
  * GLOBAL_SWEEP -- same, for non-parameter config keys (method, dlft_epsilon,
                sample_number, frf_source, ...).

The two are crossed, so PARAMS x GLOBAL_SWEEP is the full run list.

Parameters a method does not use are pinned to their dataclass default and not
swept, per METHOD_IGNORES: the DLFT normal contact is rigid, so k and alpha2
mean nothing to it and sweeping them there would only produce identical
duplicate runs. The manifest's ``collapsed`` column records which parameters
were pinned this way, i.e. which are inert for that run.

Runs are grouped by the settings that determine the FRF provider, so it is built
once per group and only the solve is repeated. Results go to
results/<STUDY_NAME>/ together with a manifest.csv; existing CSVs are skipped so
an interrupted study resumes, and a failing run is logged and does not abort the
rest.

Run it with no arguments, or::

    python study.py --only 000,003 --results results
"""

import argparse
import csv
import itertools
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import main as run
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction import plotting_saving as io
from pyfbs.nonlinearFBS.examples.two_bar_beam_friction.dynamical_system import (
    BarBeamParams)

HERE = Path(__file__).resolve().parent
DEFAULTS = BarBeamParams().to_dict()    # placeholder for method-ignored parameters

# ---------------------------------------------------------------------------
STUDY_NAME = "DLFT_circular_movement"

BASE = dict(
    params = {},                    # filled per run from PARAMS
    solver = "pyfbs-nlfbs",
    method = "aft",                 # swept below
    dlft_epsilon = 10.0,

    harmonics = list(range(0, 26)), # harmonic 0 is mandatory (static preload P5)
    sample_number = 1000,            # >= 4H+1 = 61, else the analytic Jacobian is inexact
    omega_lo = 0.5, omega_hi = 2.0, sweep = "up",
    omega_resolution = 0.002,       # only consumed by the "experimental" provider
    frf_source = "modal",
    parameterization = "ArcLengthParameterization",
    predictor = "TangentPredictorBordered",
    step_adaptation = "ExponentialAdaptation",
    solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-7},
    step_kwargs = {"base":  2.0, "initial_step_length": 0.01,
                   "maximum_step_length": 1.0, "minimum_step_length": 1e-9,
                   "goal_number_of_iterations": 4},
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
)

# The joint: regularized unilateral normal contact + regularized dry friction.
# Any scalar below may become a list to sweep it.
PARAMS = dict(
    # structure
    l = 1.0, EA = 1.0 / 3, EI = 1.0 / 3, lam = 1.0, beta = 0.05,
    # normal contact (AFT only -- DLFT is rigid)
    k = [500], alpha2 = [2],
    eps = [0.01],
    # friction (both methods)
    alpha1 = 150.0, mu = 0.1,
    # forcing
    P1 = 1.0 , P2 = 0.3, P5 = 1.0
)

GLOBAL_SWEEP = dict(method=["dlft_aft"], dlft_epsilon=[5])
# ---------------------------------------------------------------------------

# parameters each method ignores -> never swept for it, collapsed to BASE's value
METHOD_IGNORES = {"aft": (), "dlft_aft": ("k", "alpha2")}

MANIFEST_COLUMNS = ("run_id", "method", "varied", "collapsed", "status",
                    "n_points", "runtime_s", "csv", "error")


def build_runs(params, global_sweep):
    """Expand the study into ``(run_id, cfg, varied, collapsed)``, in run order.

    Duplicates are dropped: once a method's ignored parameters are collapsed,
    two runs differing only in those would be the same run.
    """
    param_axes = [(key, list(value)) for key, value in params.items()
                  if isinstance(value, list)]
    global_keys = list(global_sweep)

    runs, seen = [], set()
    for global_combo in itertools.product(*global_sweep.values()):
        overrides = dict(zip(global_keys, global_combo))
        ignored = METHOD_IGNORES.get(overrides.get("method", BASE["method"]), ())

        for param_combo in itertools.product(*[v for _, v in param_axes]):
            resolved = {k: v for k, v in params.items() if not isinstance(v, list)}
            varied, collapsed = dict(overrides), {}
            for (key, _), value in zip(param_axes, param_combo):
                if key in ignored:
                    # pinned to the dataclass default, not to an arbitrary point
                    # of the sweep: the value is unused, and picking one from the
                    # sweep would make the run look like part of it
                    resolved[key] = DEFAULTS[key]
                    collapsed[key] = resolved[key]
                else:
                    resolved[key] = value
                    varied[f"params.{key}"] = value

            cfg = deepcopy(BASE)
            cfg["params"] = resolved
            cfg.update(overrides)
            # travels into the CSV header, so the plot labels can skip them
            cfg["inert_params"] = sorted(collapsed)

            fingerprint = (tuple(sorted(resolved.items())), tuple(sorted(overrides.items())))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            runs.append((f"{len(runs):03d}_{cfg['method']}", cfg, varied, collapsed))
    return runs


def group_key(cfg):
    """Everything the FRF provider depends on -- runs sharing it reuse one build.

    The provider is synthesized from the LINEAR structure only, so the contact
    and friction parameters are absent by design.
    """
    p = cfg["params"]
    return (cfg["frf_source"], p["l"], p["EA"], p["EI"], p["lam"], p["beta"],
            cfg["omega_resolution"], run.synthesis_omega_end(cfg))


def append_manifest(path, row):
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(MANIFEST_COLUMNS)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated run ids (default: all)")
    parser.add_argument("--results", default="results",
                        help="results folder name in this example folder")
    args = parser.parse_args()

    out_dir = HERE / args.results / STUDY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    runs = build_runs(PARAMS, GLOBAL_SWEEP)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs
                if r[0] in keep or r[0].split("_", 1)[0] in keep]
    print(f"study '{STUDY_NAME}': {len(runs)} run(s) -> {out_dir}")

    groups = {}
    for entry in runs:
        groups.setdefault(group_key(entry[1]), []).append(entry)

    done = 0
    for key, group in groups.items():
        pending = [e for e in group if not (out_dir / f"{e[0]}.csv").exists()]
        for run_id, _, _, _ in group:
            if (out_dir / f"{run_id}.csv").exists():
                done += 1
                print(f"[{done}/{len(runs)}] {run_id}: skipped (result exists)")
        if not pending:
            continue

        print(f"\nbuilding {key[0]} FRF provider "
              f"(omega grid to {key[-1]:g} rad/s, {len(pending)} run(s)) ...")
        t0 = time.perf_counter()
        provider = run.build_provider(pending[0][1], pending[0][1]["frf_source"])
        print(f"build done in {time.perf_counter() - t0:.1f} s")

        for run_id, cfg, varied, collapsed in pending:
            done += 1
            shown = ", ".join(f"{k}={v}" for k, v in varied.items())
            note = ", ".join(f"{k}={v}" for k, v in collapsed.items())
            print(f"\n[{done}/{len(runs)}] {run_id}: {shown}"
                  + (f"   [ignored by method: {note}]" if note else ""))
            csv_path = out_dir / f"{run_id}.csv"
            t0 = time.perf_counter()
            try:
                system, problem, ss, solve_time = run.solve_config(cfg, provider)
                io.save_solution(csv_path, cfg, ss, problem, system,
                                 cfg["method"], cfg["frf_source"], solve_time)
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, cfg["method"], shown, note, "ok",
                                           len(ss.omega), f"{runtime:.1f}",
                                           csv_path.name, ""])
                print(f"[{run_id}] ok: {len(ss.omega)} points in {runtime:.1f} s")
            except Exception as exc:
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, cfg["method"], shown, note,
                                           "failed", 0, f"{runtime:.1f}", "",
                                           f"{type(exc).__name__}: {exc}"])
                print(f"[{run_id}] FAILED after {runtime:.1f} s: {exc}")
                traceback.print_exc()

    print(f"\nstudy '{STUDY_NAME}' finished -- manifest: {manifest}")
    print(f"plot with:  python -m pyfbs.nonlinearFBS.examples"
          f".two_bar_beam_friction.plot_comparison {STUDY_NAME}")


if __name__ == "__main__":
    main()
