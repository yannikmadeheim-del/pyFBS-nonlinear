"""
Export the VPT-consistent substructure descriptor for the pyhbm RBE2 + Craig-
Bampton reference (code/pyhbm/examples/testbench_cubicSpring_CB).

Run this from the testbench_cubicSpring example directory. It reproduces the
EXACT interface DoFs that pyFBS's virtual-point transformation uses -- the FE
nodes onto which the Grouping==<VP> sensor channels (response side, Tu) and
reference impacts (force side, Tf) snap via Model.find_nearest_locations, each
with its own direction -- and writes them, with the joint / excitation / output
definition, to ``substructure_descriptor.json`` in the pyhbm example directory.

Two levels of detail are written per substructure, because the pyhbm side has two
families of condensation:

  * ``interface_node_ids``  -- the union of the snapped nodes. All the whole-node
    condensations ("rbe2", "rbe3") need, since they retain every node DoF.
  * ``interface_channels`` / ``interface_impacts`` -- one record per workbook row,
    IN ROW ORDER, carrying the snapped node and that row's direction. The
    directional condensations ("RBE_rigid", "RBE_average") rebuild Ru and Rf from
    these, so the order has to match vpt.ru / vpt.rf row for row.

pyFBS owns the mesh and the snapper, so it is the single source of truth for the
interface node choice; pyhbm re-reads the full M/K independently from the same
Ansys .full/.rst. Only the node ids (as ORIGINAL Ansys ids, invariant to pyhbm's
orphan-node drop and renumbering), the VP frame, the joint and the I/O DoFs cross
the boundary. New substructure -> rerun this script, no pyhbm code change.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from ansys.mapdl import reader as pymapdl_reader

import pyfbs

HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[5]                    # Fast_numerical_solution_...
PYHBM_DIR = (THESIS_ROOT / "code" / "pyhbm" / "examples"
             / "testbench_cubicSpring_CB")
FEM_DIR = HERE / "lab_testbench" / "FEM"
XLSX = HERE / "lab_testbench" / "Measurements" / "coupling_example.xlsx"
OUT_JSON = PYHBM_DIR / "substructure_descriptor.json"

# --- joint / excitation / output (identical to this example's main.py) --------
# NOTE: the virtual-point position is NOT hardcoded here. It used to be, as a
# 6-digit copy of the workbook value, which put the descriptor's VP 0.37 um away
# from the one pyFBS's VPT actually uses (VP_Channels holds 0.03889537,
# 0.34810701, 0.007). Every lever arm r = x_node - x_VP inherited that offset, so
# pyhbm's Ru/Rf differed from vpt.ru/vpt.rf by ~4e-7 and Tu/Tf by ~1e-5 relative.
# Reading it from the sheet instead makes the two agree to ~1e-18.
K_DIAG     = [1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6] #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] stiffness k
C_DIAG     = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]       #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] viscous damping c
ALPHA_DIAG = [1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e8] #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] stiffness alpha
BETA_DIAG  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
F0         = 200.0
EXCITATION = dict(position=[0.112567, 0.258976, -0.009414],   # impact H28 on B
                  direction=[-0.0871559, 0.9961947, 0.0])
OUTPUT     = dict(position=[-0.076519, 0.142987, 0.022],      # channel S1 X on A
                  direction=[0.7050572, 0.7091504, 0.0])

# substructure -> (rst, full, channel sheet, impact sheet)
SUBS = {
    "A": ("A.rst", "A.full", "Channels_A", "Impacts_A"),
    "B": ("B.rst", "B.full", "Channels_B", "Impacts_B"),
}


def vp_groupings(xlsx):
    """Interface grouping ids, read from the VP sheets (not hardcoded to 10)."""
    g = set()
    for sheet in ("VP_Channels", "VP_RefChannels"):
        g |= set(pd.read_excel(xlsx, sheet_name=sheet)["Grouping"].astype(int))
    return g


def vp_position(xlsx):
    """
    The virtual-point position, straight from the workbook -- the same numbers
    pyfbs.interface.VPT builds its lever arms from, at full precision.

    All six VP rows must share one position (a VPT with several virtual points
    would need a grouping argument here), and the response and force sheets must
    agree, since a single VP frame is exported.
    """
    pos = []
    for sheet in ("VP_Channels", "VP_RefChannels"):
        df = pd.read_excel(xlsx, sheet_name=sheet)
        p = df[["Position_1", "Position_2", "Position_3"]].to_numpy(float)
        assert np.ptp(p, axis=0).max() < 1e-12, \
            f"{sheet}: rows differ in position -- more than one virtual point?"
        pos.append(p[0])
    assert np.allclose(*pos, atol=1e-12), \
        "VP_Channels and VP_RefChannels disagree on the virtual-point position"
    return pos[0]


def snap_rows(MK, nnum, df):
    """
    One record per row of ``df``, snapped to the nearest FE node.

    Positions are snapped with the very routine the VPT pipeline uses
    (``Model.find_nearest_locations`` -> ``mesh.find_closest_point``), then mapped
    to the ORIGINAL Ansys node id so the ids survive pyhbm's orphan-node drop /
    renumber. Row ORDER is preserved: pyhbm rebuilds Ru / Rf row by row from these
    records, and they must line up with vpt.ru / vpt.rf.

    :return: list of dicts (name, node_id, direction, snap_mm)
    """
    pos = df[["Position_1", "Position_2", "Position_3"]].to_numpy(float)
    dirs = df[["Direction_1", "Direction_2", "Direction_3"]].to_numpy(float)
    idx = MK.find_nearest_locations(pos)
    snap = np.linalg.norm(MK.nodes[idx] - pos, axis=1) * 1e3          # [mm]
    return [dict(name=str(n), node_id=int(nnum[i]), direction=[float(v) for v in d],
                 snap_mm=float(s))
            for n, i, d, s in zip(df["Name"], idx, dirs, snap)]


def interface_records(rst_path, full_path, df_chn, df_imp, grp):
    """
    The interface rows VPT uses on one substructure, per kind and in workbook
    order, plus their sorted unique node ids.

    The channel rows drive Tu and the impact rows drive Tf; they are DIFFERENT
    DoF sets, which is exactly why pyhbm needs them separately rather than as one
    merged node list. ``interface_node_ids`` (their union) is kept as well, since
    the whole-node condensations still only need the nodes.

    :return: (channel records, impact records, sorted unique Ansys ids)
    """
    MK = pyfbs.mck.Model.from_ansys(str(rst_path), str(full_path),
                                    no_modes=100, allow_pickle=True,
                                    recalculate=False, mesh_scale=1)
    # find_nearest_locations indexes into mesh.points, and update_locations_df
    # then reads self.nodes[index]; that identity must hold for the id map below.
    assert np.allclose(MK.mesh.points, MK.nodes), \
        "mesh.points and nodes are not aligned -- the id mapping would be wrong"

    nnum = np.asarray(pymapdl_reader.read_binary(str(rst_path)).mesh.nnum)
    assert len(nnum) == len(MK.nodes), "nnum length != node count"

    chn = snap_rows(MK, nnum, df_chn[df_chn["Grouping"].isin(grp)])
    imp = snap_rows(MK, nnum, df_imp[df_imp["Grouping"].isin(grp)])
    ids = np.unique([r["node_id"] for r in chn + imp])
    return chn, imp, ids


def build_descriptor():
    grp = vp_groupings(XLSX)
    subs = {}
    for name, (rst, full, chn_sheet, imp_sheet) in SUBS.items():
        df_chn = pd.read_excel(XLSX, sheet_name=chn_sheet)
        df_imp = pd.read_excel(XLSX, sheet_name=imp_sheet)
        chn, imp, ids = interface_records(FEM_DIR / rst, FEM_DIR / full,
                                          df_chn, df_imp, grp)
        snap = [r["snap_mm"] for r in chn + imp]
        print(f"[{name}] interface rows {len(chn)} channels + {len(imp)} impacts "
              f"-> {len(ids)} unique nodes "
              f"| snap {min(snap):.2f}..{max(snap):.2f} mm")
        print(f"      ansys ids: {ids.tolist()}")
        subs[name] = dict(rst=f"FEM/{rst}", full=f"FEM/{full}",
                          interface_node_ids=ids.tolist(),
                          interface_channels=chn,      # -> Tu, in workbook order
                          interface_impacts=imp)       # -> Tf, in workbook order

    return dict(
        example="testbench_cubicSpring",
        vp=dict(position=vp_position(XLSX).tolist(), axes=np.eye(3).tolist()),
        joint=dict(k_diag=K_DIAG, c_diag=C_DIAG, alpha_diag=ALPHA_DIAG, beta_diag=BETA_DIAG),
        excitation=dict(**EXCITATION, F0=F0),
        output=OUTPUT,
        substructures=subs,
    )


if __name__ == "__main__":
    if not FEM_DIR.exists():                      # data is a download-on-demand copy
        pyfbs.io.download_lab_testbench()

    descriptor = build_descriptor()
    OUT_JSON.write_text(json.dumps(descriptor, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # acceptance: this testbench interface is 3 sensors + impacts on each side
    n_A = len(descriptor["substructures"]["A"]["interface_node_ids"])
    n_B = len(descriptor["substructures"]["B"]["interface_node_ids"])
    assert n_A == 15 and n_B == 12, \
        f"expected A:15 / B:12 interface nodes, got A:{n_A} / B:{n_B}"
    print(f"acceptance ok: A:{n_A} / B:{n_B} interface nodes")
