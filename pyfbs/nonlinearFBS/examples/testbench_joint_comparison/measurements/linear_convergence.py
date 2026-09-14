"""
Linear coupled response of the testbench, swept over both reduction bases.

With the cubic term at alpha = 0 the joint is linear, so neither side needs
continuation: each is one complex solve per frequency line. That makes it cheap
enough to sweep the truncations that actually separate the two pipelines --
pyFBS superposes free-interface modes of the uncoupled substructures, pyhbm
superposes fixed-interface Craig-Bampton modes on top of static constraint
modes. Until BOTH families have flattened, a pyFBS-vs-pyhbm gap says nothing
about the interface treatment.

The transformation matrices are NOT a variable here: Tu and Tf agree between the
two pipelines to ~1e-13 relative (tests/test_vpt_rbe_average_equivalence.py).

    python -m pyfbs.nonlinearFBS.examples.testbench_joint_comparison.measurements.linear_convergence
    ... --workbook collocated_sensor_coupling_example --pyfbs-modes 20,60,100,200
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import main as pyfbs_main
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison.dynamical_system import N_IF

HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[6]                    # Fast_numerical_solution_...
PYHBM_EXAMPLE = (THESIS_ROOT / "code" / "pyhbm" / "examples"
                 / "testbench_joint_comparison_CB")
# importing the pyhbm example also resolves pyhbm's src/ onto sys.path
sys.path.insert(0, str(PYHBM_EXAMPLE))
import main as pyhbm_main                                       # noqa: E402


def joint_impedance(omega, k, c):
    """Z_j(omega) on the 6 VP gap DoFs, as an (n_freq, 6) diagonal."""
    return np.repeat((k + 1j * omega * c)[:, None], N_IF, axis=1)


def pyfbs_response(workbook, limit_modes, no_modes, freq, k, c,
                   modal_damping, F0, f_hi):
    """|u_out|(omega) of the linearly-coupled testbench, pyFBS admittance route.

    FBS_System's residual is R = q_rel - B Y F_ext + B Y B^T F_nl; substituting
    the linear joint F_nl = Z_j q_rel and solving for q_rel gives

        q_rel = (I + B Y B^T Z_j)^-1 B Y F_ext
        u     = Y F_ext - Y B^T Z_j q_rel

    The joint load returns through B^T because Y is already a VIRTUAL-POINT
    admittance, whose force and displacement DoFs are dual by construction --
    the pyhbm route below needs a separate Bc_load for exactly that reason.

    The admittance comes from the ModalVPFRF provider rather than the gridded
    data["Y"]: it synthesizes at the exact omega asked for, which is both what
    the continuation runs use and the only way to hit a lightly damped crest
    without materialising a very fine (n_freq, N, N) array.
    """
    cfg = dict(pyfbs_main.CONFIG, workbook=workbook, limit_modes=limit_modes,
               no_modes=no_modes, modal_damping=modal_damping,
               frf_source="modal", f_hi=f_hi)
    data, provider = pyfbs_main.build_data_and_provider(cfg)
    Bc = data["Bc"]
    dofs = np.arange(data["N"])
    Z = joint_impedance(2 * np.pi * freq, k, c)

    f_ext = np.zeros(data["N"])
    f_ext[data["inp_full"]] = F0
    eye = np.eye(N_IF)
    out = np.empty(len(freq), dtype=complex)
    for i, w in enumerate(2 * np.pi * freq):
        Y = provider.compute_FRF(w, [1], dofs, dofs)[0]
        BY = Bc @ Y
        q_rel = np.linalg.solve(eye + (BY @ Bc.T) * Z[i], BY @ f_ext)
        out[i] = (Y @ f_ext - Y @ Bc.T @ (Z[i] * q_rel))[data["out_full"]]
    return np.abs(out)


def pyhbm_response(ctx, n_modes, freq, k, c, modal_damping, F0):
    """|u_out|(omega) of the same system, Craig-Bampton route.

    The reduced ODE is M q'' + C q' + K q + Bc_load sum_e f_e(x, xdot) = f_r F0
    with x = Bc q, so a linear element k + i w c gives

        [-w^2 M + i w C + K + Bc_load diag(k + i w c) Bc] q = f_r F0

    read out through substructure A's recovery row for the plotted channel.
    """
    cfg = dict(pyhbm_main.CONFIG, n_modes=n_modes, modal_damping=modal_damping)
    M, C, K, Bc, Bc_load, f_r, sub_A, sub_B = pyhbm_main.build_reduced(cfg, ctx)
    t_out = np.concatenate([
        sub_A.recovery_row(np.array(ctx["descriptor"]["output"]["position"]),
                           np.array(ctx["descriptor"]["output"]["direction"])),
        np.zeros(sub_B.M_r.shape[0])])

    omega = 2 * np.pi * freq
    Z = joint_impedance(omega, k, c)
    out = np.empty(len(omega), dtype=complex)
    for i, w in enumerate(omega):
        A = -w**2 * M + 1j * w * C + K + Bc_load @ (Z[i][:, None] * Bc)
        out[i] = t_out @ np.linalg.solve(A, f_r * F0)
    return np.abs(out)


def rel_error(curve, reference):
    """Relative L2 distance of a curve to the richest basis of its own family."""
    return float(np.linalg.norm(curve - reference) / np.linalg.norm(reference))


def read_reference(path):
    """(freq_hz, uout_h1_abs_m) of a saved continuation result."""
    frame = pd.read_csv(path, comment="#")
    return frame["freq_hz"].to_numpy(), frame["uout_h1_abs_m"].to_numpy()


def modes_arg(text):
    return [int(v) for v in str(text).split(",") if v.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=measurements.DEFAULT,
                        help="one of: " + ", ".join(measurements.available()))
    parser.add_argument("--pyfbs-modes", default="20,40,60,100",
                        help="free-interface mode counts (default: %(default)s)")
    parser.add_argument("--pyhbm-modes", default="20,32,60",
                        help="Craig-Bampton mode counts (default: %(default)s)")
    parser.add_argument("--no-modes", type=int, default=None,
                        help="size of the pyFBS eigensolve; defaults to the "
                             "largest --pyfbs-modes (min 100). Raising it past "
                             "an earlier run re-solves the 20370-DoF "
                             "eigenproblem once -- the .full.pkl cache is keyed "
                             "on this number")
    parser.add_argument("--f-lo", type=float, default=1.0)
    parser.add_argument("--f-hi", type=float, default=1000.0)
    parser.add_argument("--df", type=float, default=0.1,
                        help="frequency step [Hz]. A lightly damped resonance is "
                             "only ~2*zeta*f wide, so a coarse grid steps over "
                             "the crest and the reported peak jumps to whichever "
                             "broader mode happens to be sampled (default: "
                             "%(default)s)")
    parser.add_argument("--k", type=float, default=1.0e6, help="joint stiffness")
    parser.add_argument("--c", type=float, default=0.5, help="joint damping")
    parser.add_argument("--modal-damping", type=float, default=0.005)
    parser.add_argument("--f0", type=float, default=80.0, help="force amplitude [N]")
    parser.add_argument("--reference", nargs="*", default=(),
                        help="continuation CSVs to overlay (alpha=0 runs)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    pyfbs_modes = modes_arg(args.pyfbs_modes)
    pyhbm_modes = modes_arg(args.pyhbm_modes)
    no_modes = args.no_modes or max(100, max(pyfbs_modes))
    out_dir = Path(args.out_dir) if args.out_dir else (
        HERE.parent / "results" / "linear_convergence")
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = {}
    freq = np.arange(args.f_lo, args.f_hi + args.df, args.df)
    for m in pyfbs_modes:
        curves[("pyFBS", m)] = pyfbs_response(
            args.workbook, m, no_modes, freq, args.k, args.c,
            args.modal_damping, args.f0, args.f_hi)
        peak = freq[np.argmax(curves[("pyFBS", m)])]
        print(f"pyFBS  limit_modes={m:<5d} peak {peak:8.3f} Hz")

    ctx = pyhbm_main.load_substructures(args.workbook)
    for m in pyhbm_modes:
        curves[("pyhbm", m)] = pyhbm_response(ctx, m, freq, args.k, args.c,
                                              args.modal_damping, args.f0)
        peak = freq[np.argmax(curves[("pyhbm", m)])]
        print(f"pyhbm  n_modes={m:<9d} peak {peak:8.3f} Hz")

    # a resonance is only ~2*zeta*f wide, and below ~4 samples across that the
    # crest is badly undersampled -- which shifts the reported peak onto
    # whichever broader mode happens to be sampled near its top
    lowest_peak = min(freq[np.argmax(mag)] for mag in curves.values())
    width = 2.0 * args.modal_damping * lowest_peak
    if args.df > width / 4.0:
        print(f"\n"
              f"WARNING: df={args.df:g} Hz vs a {width:.3g} Hz half-power "
              f"width at the lowest peak ({lowest_peak:g} Hz) -- crests are "
              f"undersampled; use --df {width / 4.0:.3g} or smaller")

    # convergence is measured WITHIN each family, against its own richest basis:
    # the two bases are different objects, so a cross-family difference is only
    # meaningful once both have flattened
    richest = {"pyFBS": max(pyfbs_modes), "pyhbm": max(pyhbm_modes)}
    print("\nrelative L2 distance to the richest basis of the same family")
    for family, mode_list in (("pyFBS", pyfbs_modes), ("pyhbm", pyhbm_modes)):
        ref = curves[(family, richest[family])]
        for m in mode_list:
            print(f"  {family:<6} {m:<5d} {rel_error(curves[(family, m)], ref):.3e}")
    cross = rel_error(curves[("pyFBS", richest["pyFBS"])],
                      curves[("pyhbm", richest["pyhbm"])])
    print(f"\npyFBS({richest['pyFBS']}) vs pyhbm({richest['pyhbm']}): {cross:.3e}")

    table = pd.DataFrame({"freq_hz": freq})
    for (family, m), mag in curves.items():
        table[f"{family}_{m}"] = mag
    csv_path = out_dir / (args.workbook + ".csv")
    table.to_csv(csv_path, index=False)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for (family, m), mag in curves.items():
        ax.semilogy(freq, mag, lw=1.2,
                    ls="-" if family == "pyFBS" else "--",
                    label=f"{family} {m}")
    for path in args.reference:
        f_ref, u_ref = read_reference(path)
        ax.semilogy(f_ref, u_ref, "k:", lw=1.0, label=Path(path).stem)
    ax.set(xlabel="frequency [Hz]", ylabel="|u_out| [m]",
           xlim=(args.f_lo, args.f_hi),
           title=f"linear coupling, {args.workbook}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    for family, mode_list, marker in (("pyFBS", pyfbs_modes, "o"),
                                      ("pyhbm", pyhbm_modes, "s")):
        ref = curves[(family, richest[family])]
        errors = [rel_error(curves[(family, m)], ref) for m in mode_list]
        # the richest basis sits at exactly zero, which a log axis drops
        ax2.semilogy(mode_list, [e if e > 0 else np.nan for e in errors],
                     marker=marker, label=family)
    ax2.set(xlabel="modes retained", ylabel="rel. L2 vs richest basis",
            title="basis convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / (args.workbook + ".png")
    fig.savefig(png, dpi=130)
    print(f"\nwrote {csv_path}\nwrote {png}")


if __name__ == "__main__":
    main()
