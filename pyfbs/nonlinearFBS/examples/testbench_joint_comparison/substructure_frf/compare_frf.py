"""
Virtual-point driving-point FRF of ONE substructure at a time, pyFBS vs pyhbm.

The coupled joint studies mix three effects -- the joint law, the coupling and
the interface treatment. This script removes the first two: each substructure is
left free and only its own 6x6 virtual-point admittance is looked at, so the
remaining pyFBS-vs-pyhbm difference is the interface treatment alone. Two
measurement tables are run side by side, a fully collocated one and a
non-collocated one, because that is the pair the mismatch question is about.

The two interface treatments:

  pyFBS   free-interface modes of the substructure, synthesized into channel
          FRFs and then projected with the VIRTUAL POINT TRANSFORMATION,
          Y_VP = Tu Y Tf. Tu is built from the Channels_<X> rows and Tf from the
          Impacts_<X> rows, so on a non-collocated table they come from
          different DoF sets and Y_VP stops being reciprocal.

  pyhbm   the RBE element is part of the FE MODEL, not a post-processing step.
          RBE2 is applied to K and M themselves (apply_rbe2 forms T_b.T K T_b)
          before the Craig-Bampton reduction, so the master's six DoFs are
          genuine DoFs of the reduced matrices and the admittance is just their
          block of the inverted dynamic stiffness. RBE3 ties a DEPENDENT,
          MASSLESS master to the still-free interface (u_m = G.T u_Gamma);
          eliminating that DoF is exact and reproduces the G / G.T pair the
          reduction code carries around the inverse. check_rbe3_elimination()
          verifies that on the assembled system rather than assuming it.

Both bases are truncated at the same number (default 100), but they are NOT the
same object: pyFBS keeps free-interface modes of the whole substructure, pyhbm
keeps fixed-interface modes on top of the static constraint modes. A residual
high-frequency gap is therefore truncation, not the interface -- see
measurements/interface_modes.py to quantify it.

Nothing here is shared with the continuation examples: the script only reads
their data and writes into its own results/ folder.

    python compare_frf.py
    python compare_frf.py --dof ux --f-hi 2000 --modes 60
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]                 # the checkout that holds the pyfbs package
if str(REPO_ROOT) not in sys.path:          # so a plain `python compare_frf.py` works
    sys.path.insert(0, str(REPO_ROOT))

from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import (  # noqa: E402
    dynamical_system as fbs_ds,
    export_substructure_descriptor as exporter,
)

# The pyhbm reduction infrastructure is loaded by explicit path, as
# measurements/interface_modes.py does. Reaching for the shared module directly
# instead of the pyhbm example's main.py is what keeps the pyhbm folder out of
# this: the workbook and the descriptor are read from THIS example, so the two
# folders' measurement tables need not be in sync.
THESIS_ROOT = HERE.parents[6]
CB_DIR = THESIS_ROOT / "code" / "pyhbm" / "examples" / "testbench_cubicSpring_CB"
_spec = importlib.util.spec_from_file_location("cb_infra", CB_DIR / "dynamical_system.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

FEM_DIR = CB_DIR / "lab_testbench" / "FEM"      # the npz cache next to it is used first
N_IF = fbs_ds.N_IF
VP_DOFS = fbs_ds.VP_DOFS
CONDENSATIONS = ("rbe2", "rbe3")

# The default pair is matched: both tables carry the SAME 12 impact rows on A
# and 9 on B; the collocated one additionally lists those rows as channels, so
# collocation is the only difference between the two rows of the figure.
DEFAULT_COLLOCATED = "collocated_impact_locations"
DEFAULT_NON_COLLOCATED = "non_collocated_original"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_descriptor(workbook):
    """
    The pyFBS descriptor of ``workbook``: VP position and the interface node ids
    the VPT sees. Both pipelines are anchored on it, so they see exactly the same
    node set.

    A missing descriptor is reported rather than built: the exporter asks for
    no_modes=100, and pyFBS overwrites its .full.pkl cache whenever that number
    differs from the cached one -- which would throw away the 4000-mode solve
    stored there. Running the exporter is therefore left as a deliberate step.
    """
    path = exporter.descriptor_path(workbook)
    if not path.is_file():
        raise SystemExit(
            f"no descriptor for {workbook!r}: {path}\n"
            f"build it with\n"
            f"    python ../export_substructure_descriptor.py {workbook}\n"
            f"NOTE the exporter loads the model with no_modes=100, which "
            f"re-solves and OVERWRITES lab_testbench/FEM/*.full.pkl if that "
            f"cache holds a different mode count (today: 4000).")
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# pyFBS side
# ---------------------------------------------------------------------------

def pyfbs_vp_blocks(workbook, modes, no_modes, f_resolution, f_end, modal_damping):
    """
    The uncoupled 6x6 virtual-point admittances of A and B, pyFBS route.

    build_testbench_data does the whole modal + VPT pipeline; the coupling it
    also returns is ignored here. In its block-diagonal layout
    [A ref | A VP(6) | B VP(6) | B ref] the VP rows are the LAST six of A and the
    FIRST six of B.

    ``modes`` is what the synthesis superposes; ``no_modes`` only sizes the
    eigensolve and must be left at whatever the .full.pkl cache holds -- see
    the --no-modes help text.

    :returns: (freq, {"A": (n_freq, 6, 6), "B": (n_freq, 6, 6)})
    """
    data = fbs_ds.build_testbench_data(
        f_resolution=f_resolution, modal_damping=modal_damping, f_end=f_end,
        limit_modes=modes, no_modes=no_modes, workbook=workbook)
    return data["freq"], {"A": data["Y_A"][:, -N_IF:, -N_IF:],
                          "B": data["Y_B"][:, :N_IF, :N_IF]}


# ---------------------------------------------------------------------------
# pyhbm side
# ---------------------------------------------------------------------------

def reduce_substructure(name, descriptor, condensation, n_modes, modal_damping):
    """One Craig-Bampton substructure with the RBE element built into K and M."""
    data = cb.load_or_export(name, FEM_DIR, CB_DIR)
    idx = cb.find_nodes_by_ansys_id(
        data["nodes"], data["nnum"],
        descriptor["substructures"][name]["interface_node_ids"])
    vp_xyz = np.array(descriptor["vp"]["position"])
    cb.report_interface(f"{name}/{condensation}", data["nodes"], idx, vp_xyz)
    return cb.ReducedSubstructure.build(
        name, data, idx, vp_xyz, n_modes, modal_damping,
        condensation=condensation)


def vp_blocks(sub, omega):
    """
    (n_freq, 6, 6) virtual-point admittance of one reduced substructure.

    For RBE2 vp_operator is a plain selection of the six master DoFs, so this is
    literally the master block of the inverted dynamic stiffness. For RBE3 it is
    the G.T / G pair of the eliminated massless master node -- same thing, one
    exact elimination further; see check_rbe3_elimination.
    """
    out = np.empty((len(omega), N_IF, N_IF), dtype=complex)
    for i, w in enumerate(omega):
        Z = -w**2 * sub.M_r + 1j * w * sub.C_r + sub.K_r
        out[i] = sub.vp_operator @ np.linalg.solve(Z, sub.vp_load)
    return out


def check_rbe3_elimination(sub, omega, tol=1e-10):
    """
    Show that the RBE3 master really is a physical element of the model.

    Keep the master node as its own six DoFs, give it no mass and no stiffness
    of its own, and tie it to the interface with the RBE3 constraint
    u_m = G.T q_r. Written as one transformation T = [I; G.T] over the
    independent DoFs q_r, the augmented system reduces to

        (T.T diag(Z, 0) T) q_r = T.T f_m   ->   u_m = G.T Z^-1 G w,

    which is the form vp_blocks evaluates. Raises if the two disagree, so the
    "element before the inversion" property is verified rather than claimed.
    """
    P = sub.vp_operator                                   # G.T on the boundary DoFs
    nr = sub.M_r.shape[0]
    T = np.vstack([np.eye(nr), P])                        # q_r -> [q_r; u_m]
    off = np.zeros((nr, N_IF))
    zero = np.zeros((N_IF, N_IF))
    M = np.block([[sub.M_r, off], [off.T, zero]])
    C = np.block([[sub.C_r, off], [off.T, zero]])
    K = np.block([[sub.K_r, off], [off.T, zero]])

    w = omega
    Z_aug = T.T @ (-w**2 * M + 1j * w * C + K) @ T
    load = T.T @ np.vstack([off, np.eye(N_IF)])           # unit wrench on u_m
    augmented = P @ np.linalg.solve(Z_aug, load)

    Z = -w**2 * sub.M_r + 1j * w * sub.C_r + sub.K_r
    implemented = P @ np.linalg.solve(Z, sub.vp_load)
    err = np.linalg.norm(augmented - implemented) / np.linalg.norm(implemented)
    assert err < tol, (f"[{sub.name}] RBE3 master elimination mismatch: {err:.2e} "
                       f"at {w / (2 * np.pi):.1f} Hz")
    print(f"[{sub.name}] RBE3 massless-master elimination verified "
          f"at {w / (2 * np.pi):.1f} Hz (rel. {err:.1e})")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_case(workbook, args, dof):
    """Every curve of one measurement table, keyed (substructure, method)."""
    print(f"\n{'=' * 70}\n=== {workbook}\n{'=' * 70}")
    descriptor = load_descriptor(workbook)
    print(f"collocated: {descriptor['collocated']}")

    freq, blocks = pyfbs_vp_blocks(workbook, args.modes, args.no_modes, args.df,
                                   args.f_hi, args.modal_damping)
    keep = freq >= args.f_lo
    freq = freq[keep]
    omega = 2 * np.pi * freq

    curves = {(name, "pyFBS VPT"): blocks[name][keep, dof, dof]
              for name in ("A", "B")}

    for condensation in CONDENSATIONS:
        for name in ("A", "B"):
            t0 = time.time()
            sub = reduce_substructure(name, descriptor, condensation,
                                      args.modes, args.modal_damping)
            if condensation == "rbe3":
                check_rbe3_elimination(sub, omega[len(omega) // 2])
            label = f"pyhbm {condensation.upper()}"
            curves[(name, label)] = vp_blocks(sub, omega)[:, dof, dof]
            print(f"[{name}] {condensation} done in {time.time() - t0:.1f} s")

    return freq, curves, bool(descriptor["collocated"])


def rel_l2(curve, reference):
    return float(np.linalg.norm(curve - reference) / np.linalg.norm(reference))


def report(freq, curves):
    """Distance of each pyhbm curve to the pyFBS one, per substructure."""
    for name in ("A", "B"):
        ref = np.abs(curves[(name, "pyFBS VPT")])
        for method in ("pyhbm RBE2", "pyhbm RBE3"):
            mag = np.abs(curves[(name, method)])
            print(f"  {name}: {method:<11} vs pyFBS VPT  "
                  f"|Y| rel. L2 {rel_l2(mag, ref):.3e}   "
                  f"peak {freq[np.argmax(mag)]:7.2f} Hz vs "
                  f"{freq[np.argmax(ref)]:7.2f} Hz")


STYLE = {"pyFBS VPT": dict(ls="-", lw=1.4, color="tab:blue"),
         "pyhbm RBE2": dict(ls="--", lw=1.2, color="tab:orange"),
         "pyhbm RBE3": dict(ls="-.", lw=1.2, color="tab:green")}


def figure(cases, dof_name, args):
    """One 2x2 figure: rows are the two tables, columns the two substructures."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for row, (workbook, freq, curves, collocated) in enumerate(cases):
        for col, name in enumerate(("A", "B")):
            ax = axes[row, col]
            for method, style in STYLE.items():
                ax.semilogy(freq, np.abs(curves[(name, method)]),
                            label=method, **style)
            tag = "collocated" if collocated else "NOT collocated"
            ax.set_title(f"substructure {name} -- {workbook} ({tag})", fontsize=9)
            ax.grid(alpha=0.3)
            if row == len(cases) - 1:
                ax.set_xlabel("frequency [Hz]")
            if col == 0:
                ax.set_ylabel(f"|Y_VP| {dof_name}/{dof_name} [m/N]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"uncoupled virtual-point driving-point FRF, {dof_name} -- "
                 f"{args.modes} modes per substructure, zeta={args.modal_damping}",
                 fontsize=11)
    fig.tight_layout()
    return fig


def save_csv(cases, path):
    """One long table: the complex curves of both cases, tagged by workbook."""
    import pandas as pd

    frames = []
    for workbook, freq, curves, _ in cases:
        frame = pd.DataFrame({"workbook": workbook, "freq_hz": freq})
        for (name, method), curve in curves.items():
            tag = f"{name}_{method.split()[-1]}"
            frame[f"{tag}_re"] = curve.real
            frame[f"{tag}_im"] = curve.imag
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collocated", default=DEFAULT_COLLOCATED,
                        help="collocated table (default: %(default)s)")
    parser.add_argument("--non-collocated", default=DEFAULT_NON_COLLOCATED,
                        help="non-collocated table (default: %(default)s)")
    parser.add_argument("--dof", default="uz", choices=VP_DOFS,
                        help="virtual-point DoF of the driving point")
    parser.add_argument("--modes", type=int, default=100,
                        help="free-interface modes (pyFBS) and fixed-interface "
                             "Craig-Bampton modes (pyhbm)")
    parser.add_argument("--no-modes", type=int, default=4000,
                        help="size of the pyFBS eigensolve. pyFBS keys its "
                             "lab_testbench/FEM/*.full.pkl cache on this number "
                             "and OVERWRITES the cache when it differs, so a "
                             "wrong value silently throws away the ~660 MB "
                             "4000-mode solve that is stored there today "
                             "(default: %(default)s)")
    parser.add_argument("--f-lo", type=float, default=1.0)
    parser.add_argument("--f-hi", type=float, default=1000.0)
    parser.add_argument("--df", type=float, default=0.5,
                        help="frequency step [Hz]; a lightly damped resonance is "
                             "only ~2*zeta*f wide, so a coarse grid steps over "
                             "the crest")
    parser.add_argument("--modal-damping", type=float, default=0.005)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    dof = VP_DOFS.index(args.dof)
    out_dir = Path(args.out_dir) if args.out_dir else HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for workbook in (args.collocated, args.non_collocated):
        freq, curves, collocated = run_case(workbook, args, dof)
        cases.append((workbook, freq, curves, collocated))

    print(f"\n{'=' * 70}\nvirtual-point driving point {args.dof}/{args.dof}, "
          f"{args.f_lo:g}..{args.f_hi:g} Hz")
    for workbook, freq, curves, collocated in cases:
        print(f"\n{workbook} (collocated={collocated})")
        report(freq, curves)

    stem = f"vp_driving_point_{args.dof}_{args.modes}modes"
    save_csv(cases, out_dir / f"{stem}.csv")
    fig = figure(cases, args.dof, args)
    fig.savefig(out_dir / f"{stem}.png", dpi=130)
    print(f"\nwrote {out_dir / (stem + '.csv')}\nwrote {out_dir / (stem + '.png')}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
