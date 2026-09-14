"""
Parameter permutations of the composable joint, one result CSV per run.

Edit the top block: BASE holds the defaults (a CONFIG exactly like main.py's),
JOINT_SETS lists the joint compositions to compare, and GLOBAL_SWEEP varies
non-joint config keys. Inside a joint set ANY scalar field may be written as a
LIST of values -- the study then runs every combination of those lists, crossed
with every combination of GLOBAL_SWEEP.

Values are varied INSIDE the joint spec rather than in one global sweep table
because a parameter then only multiplies the runs whose joint actually has it: a
sweep over ``alpha`` in the cubic set produces no duplicate runs in the linear or
friction sets. ``dofs`` and ``spin_dof`` are excluded from sweeping by name --
they are legitimately tuples/strings; to compare different DoF assignments,
write a second joint set.

Runs are grouped by the settings that determine the (expensive) testbench build,
so the ANSYS -> VPT -> admittance pipeline runs once per group and only the
solve is repeated. Results go to results/<STUDY_NAME>/, together with a
manifest.csv; existing CSVs are skipped so an interrupted study resumes, and a
failing run is logged and does not abort the rest.

Run it with no arguments, or::

    python study.py --only 000,003 --results results
"""

import argparse
import csv
import itertools
import sys
import time
import traceback
from pathlib import Path

try:                                    # live, UTF-8 progress prints on Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, ValueError):
    pass

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import main as run
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
STUDY_NAME = "collocated_impacts_static_correction_linear"

BASE = dict(
    joints = [],                   # filled per run from JOINT_SETS
    F0 = 80.0, modal_damping = 0.005,        # F0 swept below
    solver = "pyfbs-nlfbs",
    workbook = "collocated_impact_locations",
    # limit_modes / no_modes deliberately absent -> None / 100 in
    # build_data_and_provider (the run's config_json has no such keys)
    frf_source = "experimental",
    limit_modes = 100,
    f_resolution = 0.1,            # only the VPT synthesis grid, modal ignores it
    harmonics = [1, 3, 5, 7], sample_number = 256,
    f_lo = 1.0, f_hi = 1000.0, sweep = "down",
    parameterization = "ArcLengthParameterization",
    predictor = "TangentPredictorBordered",
    step_adaptation = "ExponentialAdaptation",
    solver_kwargs = {"maximum_iterations": 300, "absolute_tolerance": 1e-6},
    step_kwargs = {"base": 2.0, "initial_step_length": 0.01,
                   "maximum_step_length": 0.5, "minimum_step_length": 1e-6,
                   "goal_number_of_iterations": 3},
    maximum_number_of_solutions = 50000, jacobian_update_frequency = 1,
)

ALL6 = ("ux", "uy", "uz", "rx", "ry", "rz")
JOINT_SETS = [
    [dict(type="linear", k=1.0e6, c=0.5, dofs=ALL6),
     dict(type="cubic",  alpha=[0],
          dofs=ALL6)],
]

GLOBAL_SWEEP = dict(F0=[80])
# ---------------------------------------------------------------------------

SWEEP_EXEMPT = ("dofs", "spin_dof")   # tuples/strings, not sweepable value lists
MANIFEST_COLUMNS = ("run_id", "joint_types", "varied", "status", "n_points",
                    "runtime_s", "csv", "error")


def joint_prefixes(specs):
    """``linear``, ``linear2``, ``cubic``, ... -- one name per spec, so a repeated
    joint type stays distinguishable in the manifest (same scheme as the
    automatic plot labels)."""
    seen, out = {}, []
    for spec in specs:
        kind = spec.get("type", "?")
        n = seen.get(kind, 0)
        seen[kind] = n + 1
        out.append(kind if n == 0 else f"{kind}{n + 1}")
    return out


def joint_slug(specs):
    return "+".join(dict.fromkeys(spec["type"] for spec in specs))


def build_runs(base, joint_sets, global_sweep):
    """Expand the study into ``(run_id, cfg, varied)`` triples, in run order."""
    global_keys = list(global_sweep)
    runs = []
    for specs in joint_sets:
        prefixes = joint_prefixes(specs)
        axes = [(i, key, list(value)) for i, spec in enumerate(specs)
                for key, value in spec.items()
                if key not in SWEEP_EXEMPT and isinstance(value, list)]
        for joint_combo in itertools.product(*[values for _, _, values in axes]):
            resolved = [dict(spec) for spec in specs]
            varied_joint = {}
            for (i, key, _), value in zip(axes, joint_combo):
                resolved[i][key] = value
                varied_joint[f"{prefixes[i]}.{key}"] = value
            for global_combo in itertools.product(*global_sweep.values()):
                cfg = dict(base)
                cfg["joints"] = [dict(spec) for spec in resolved]
                cfg.update(zip(global_keys, global_combo))
                varied = dict(varied_joint, **dict(zip(global_keys, global_combo)))
                runs.append((f"{len(runs):03d}_{joint_slug(resolved)}", cfg, varied))
    return runs


def group_key(cfg):
    """Everything the testbench build depends on -- runs sharing it reuse one
    data set and one FRF provider."""
    return (cfg["f_resolution"], cfg["modal_damping"],
            run.synthesis_f_end(cfg), cfg["frf_source"],
            cfg.get("limit_modes"), cfg.get("no_modes", 100),
            cfg.get("workbook", measurements.DEFAULT),
            cfg.get("static_correction", True))


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

    runs = build_runs(BASE, JOINT_SETS, GLOBAL_SWEEP)
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
        for run_id, _, _ in group:
            if (out_dir / f"{run_id}.csv").exists():
                done += 1
                print(f"[{done}/{len(runs)}] {run_id}: skipped (result exists)")
        if not pending:
            continue

        print(f"\nbuilding testbench data for f_resolution={key[0]:g}, "
              f"modal_damping={key[1]:g}, f_end={key[2]:g} Hz, "
              f"frf_source={key[3]}, limit_modes={key[4]} "
              f"({len(pending)} run(s)) ...")
        t0 = time.perf_counter()
        data, provider = run.build_data_and_provider(pending[0][1])
        print(f"build done in {time.perf_counter() - t0:.0f} s")

        for run_id, cfg, varied in pending:
            done += 1
            shown = ", ".join(f"{k}={v}" for k, v in varied.items())
            print(f"\n[{done}/{len(runs)}] {run_id}: {shown}")
            csv_path = out_dir / f"{run_id}.csv"
            t0 = time.perf_counter()
            try:
                system, problem, ss, solve_time = run.solve_config(cfg, data, provider)
                run.save_solution(csv_path, cfg, ss, problem, system, data,
                                  solve_time, full=run.SAVE_FULL_RESPONSE)
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, joint_slug(cfg["joints"]),
                                           shown, "ok", len(ss.omega),
                                           f"{runtime:.1f}", csv_path.name, ""])
                print(f"[{run_id}] ok: {len(ss.omega)} points in {runtime:.0f} s")
            except Exception as exc:
                runtime = time.perf_counter() - t0
                append_manifest(manifest, [run_id, joint_slug(cfg["joints"]),
                                           shown, "failed", 0, f"{runtime:.1f}",
                                           "", f"{type(exc).__name__}: {exc}"])
                print(f"[{run_id}] FAILED after {runtime:.0f} s: {exc}")
                traceback.print_exc()

    print(f"\nstudy '{STUDY_NAME}' finished -- manifest: {manifest}")


if __name__ == "__main__":
    main()
