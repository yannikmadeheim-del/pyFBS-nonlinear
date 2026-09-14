"""
Free- and fixed-interface modes of both substructures, for one measurement table.

Free-interface modes belong to the substructure alone: they are what pyFBS's
modal FRF synthesis superposes, and they do not depend on the table at all.
Fixed-interface modes are what Craig-Bampton adds on top of the static
constraint modes in pyhbm, and they DO depend on the table -- the clamped set is
exactly the retained boundary, so a table with fewer interface rows clamps fewer
DoFs, leaves a larger interior, and lands its fixed-interface modes lower.

That contrast is the point of this script: it says how many CB modes each table
needs before the pyhbm side can be compared against the pyFBS one at all.

    python -m pyfbs.nonlinearFBS.examples.testbench_joint_comparison.measurements.interface_modes
    ... --workbook collocated_sensor_coupling_example --n-modes 60 --csv modes.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import pyfbs
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements

HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[6]                    # Fast_numerical_solution_...
PYHBM_EXAMPLE = (THESIS_ROOT / "code" / "pyhbm" / "examples"
                 / "testbench_joint_comparison_CB")
# importing the pyhbm example also resolves pyhbm's src/ onto sys.path, which is
# what lets the reduction helpers below import from an uninstalled checkout
sys.path.insert(0, str(PYHBM_EXAMPLE))
import main as pyhbm_main                                       # noqa: E402
# the example's dynamical_system is a thin shim that loads the shared cubic-spring
# infrastructure as its `cb` attribute and re-exports only what main.py needs; the
# reduction helpers below are not among them, so reach through the shim
from dynamical_system import cb as pyhbm_cb                     # noqa: E402

build_directional_boundary = pyhbm_cb.build_directional_boundary
craig_bampton = pyhbm_cb.craig_bampton
natural_frequencies = pyhbm_cb.natural_frequencies
rotate_and_partition = pyhbm_cb.rotate_and_partition

BAND_HZ = 1000.0            # the analysis band the joint studies continue over


def substructure_modes(data, vpt_rows, io_rows, n_modes, n_free):
    """(free-interface Hz, fixed-interface Hz) of one substructure."""
    free = natural_frequencies(data["K"], data["M"], n_free)
    # the boundary is the UNION of channel and impact rows, exactly as
    # ReducedSubstructure._build_directional assembles it, so the clamped set
    # here is the one the RBE_average run actually uses
    union = vpt_rows["channels"] + vpt_rows["impacts"]
    boundary = build_directional_boundary(union, io_rows, len(data["nodes"]))
    blocks = rotate_and_partition(data["K"], data["M"], boundary)
    *_, fixed = craig_bampton(blocks, n_modes)
    return free, fixed, boundary.n_boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=measurements.DEFAULT,
                        help=f"one of: {', '.join(measurements.available())}")
    parser.add_argument("--n-modes", type=int, default=32,
                        help="fixed-interface modes to solve (default: %(default)s)")
    parser.add_argument("--n-free", type=int, default=32,
                        help="free-interface modes to solve (default: %(default)s)")
    parser.add_argument("--csv", help="also write the table here")
    args = parser.parse_args()

    ctx = pyhbm_main.load_substructures(args.workbook)
    rows, summary = [], {}
    for name in ("A", "B"):
        free, fixed, n_boundary = substructure_modes(
            ctx["substructures"][name], ctx["vpt_rows"][name],
            ctx["io_rows"][name], args.n_modes, args.n_free)
        summary[name] = (free, fixed, n_boundary)
        # the two sets are independent and generally of different length, so
        # they go out as separate rows: pairing them would silently truncate
        # both to min(--n-free, --n-modes)
        for kind, values in (("free", free), ("fixed", fixed)):
            for i, hz in enumerate(values, start=1):
                rows.append(dict(substructure=name, kind=kind, mode=i, hz=hz))

    for name, (free, fixed, n_boundary) in summary.items():
        n_in_band = int(np.count_nonzero(fixed < BAND_HZ))
        print(f"\n[{name}] boundary {n_boundary} DoFs | "
              f"free-interface: {free[0]:.2f} .. {free[-1]:.1f} Hz | "
              f"fixed-interface: {fixed[0]:.1f} .. {fixed[-1]:.1f} Hz")
        print(f"      {n_in_band} of {len(fixed)} fixed-interface modes below "
              f"{BAND_HZ:g} Hz")
        if fixed[-1] < BAND_HZ:
            print(f"      WARNING: the whole fixed-interface set is inside the "
                  f"band -- raise --n-modes")
        print(f"      {'mode':>4}  {'free [Hz]':>12}  {'fixed [Hz]':>12}")
        for i in range(max(len(free), len(fixed))):
            a = f"{free[i]:12.3f}" if i < len(free) else " " * 12
            b = f"{fixed[i]:12.3f}" if i < len(fixed) else " " * 12
            print(f"      {i + 1:>4}  {a}  {b}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
