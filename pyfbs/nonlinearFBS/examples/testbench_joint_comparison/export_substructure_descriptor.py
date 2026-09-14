"""
Export the VPT-consistent substructure descriptor of a measurement table in
``measurements/``, for the pyhbm RBE2 + Craig-Bampton counterpart of this
example (code/pyhbm/examples/testbench_joint_comparison_CB).

pyFBS and pyhbm run on DIFFERENT machines, so the descriptor is written next to
this script and the resulting ``substructure_descriptor_<workbook>.json`` is
copied to the pyhbm example folder by hand. The file name is the one that
example's main.py looks up, so it needs no renaming on arrival; the workbook
itself has to be copied into its ``measurements/`` as well.

The descriptor reproduces the EXACT interface DoFs that pyFBS's virtual-point
transformation uses -- the FE nodes onto which the Grouping==<VP> sensor channels
(response side, Tu) and reference impacts (force side, Tf) snap via
Model.find_nearest_locations, each with its own direction -- plus the virtual
point frame and the excitation / output DoFs.

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
the boundary. New measurement table -> rerun this script, no pyhbm code change.

    python export_substructure_descriptor.py                 # every table missing one
    python export_substructure_descriptor.py my_new_table    # these, overwriting
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ansys.mapdl import reader as pymapdl_reader

import pyfbs
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison.dynamical_system \
    import DATA_DIR, N_IF, _resolve_data_dir

HERE = Path(__file__).resolve().parent

# The FE data is the shared lab_testbench copy next to the cubic-spring example
# and the workbooks are this example's own measurements/ -- the exact two sources
# build_testbench_data reads, so the descriptor cannot end up describing a
# different mesh or a different table than the pyFBS run it is compared against.
FEM_DIR = DATA_DIR / "FEM"

# --- joint --------------------------------------------------------------------
# NOT used by testbench_joint_comparison_CB: its main.py takes the joint element
# list and F0 from its own CONFIG. The block is written anyway so the JSON schema
# stays the one the cubic-spring example's main.py reads out of
# descriptor["joint"]; these values are that example's, not this study's.
K_DIAG     = [1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6] #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] stiffness k
C_DIAG     = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]       #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] viscous damping c
ALPHA_DIAG = [1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e8] #[x_trans, y_trans, z_trans, x_rot, y_rot, z_rot] stiffness alpha
BETA_DIAG  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
F0         = 200.0

# substructure -> (rst, full, channel sheet, impact sheet)
SUBS = {
    "A": ("A.rst", "A.full", "Channels_A", "Impacts_A"),
    "B": ("B.rst", "B.full", "Channels_B", "Impacts_B"),
}


def descriptor_path(workbook):
    """
    Where the descriptor of one table goes.

    The name is the convention the pyhbm side's main.py looks up
    (``substructure_descriptor_<workbook>.json``), so the file can be copied to
    the remote example folder unchanged.
    """
    return HERE / f"substructure_descriptor_{workbook}.json"


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


def vp_ordered(df, df_vp, grp):
    """
    The row order VPT gives this sheet.

    VPT rebuilds a sheet as its groupings in ASCENDING order, with the interface
    grouping's rows replaced by the six VP rows (VPT._get_vp_data_frame, with the
    default sort_grouping=True). dynamical_system then indexes the result
    directly -- out_full = 0 and inp_full = nA + N_IF -- so the excitation and
    output DoFs are read off this frame rather than hardcoded to one workbook's
    S1 X / H28, which a differently ordered table would silently move.
    """
    parts = [df_vp[df_vp["Grouping"] == g] if g in grp else df[df["Grouping"] == g]
             for g in sorted(set(df["Grouping"].astype(int)))]
    return pd.concat(parts, ignore_index=True)


def io_dof(df, row, kind):
    """
    Name, position and direction of one VPT-ordered row, at full precision.

    :param df: the frame ``vp_ordered`` returned for the sheet
    :param row: its row index (0 for the output channel, N_IF for the excitation)
    """
    if row >= len(df):
        raise SystemExit(f"{kind}: sheet has {len(df)} VPT rows, no row {row} -- "
                         f"too few non-interface DoFs in the table")
    r = df.iloc[row]
    assert not str(r["Name"]).startswith("VP"), \
        f"{kind} landed on a virtual-point row ({r['Name']}) -- the sheet has no " \
        f"non-interface DoF at the index dynamical_system uses"
    return dict(name=str(r["Name"]),
                position=[float(r[f"Position_{i}"]) for i in (1, 2, 3)],
                direction=[float(r[f"Direction_{i}"]) for i in (1, 2, 3)])


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


def build_descriptor(workbook=measurements.DEFAULT):
    """
    The descriptor of one measurement table in ``measurements/``.

    :param workbook: file name without the extension; ``measurements.available()``
        lists the choices
    """
    xlsx = measurements.resolve(workbook)
    fem = FEM_DIR if FEM_DIR.exists() else _resolve_data_dir() / "FEM"
    grp = vp_groupings(xlsx)

    sheets = {name: pd.read_excel(xlsx, sheet_name=name)
              for name in ("Channels_A", "Impacts_A", "Channels_B", "Impacts_B",
                           "VP_Channels", "VP_RefChannels")}

    subs = {}
    for name, (rst, full, chn_sheet, imp_sheet) in SUBS.items():
        chn, imp, ids = interface_records(fem / rst, fem / full,
                                          sheets[chn_sheet], sheets[imp_sheet], grp)
        snap = [r["snap_mm"] for r in chn + imp]
        print(f"[{name}] interface rows {len(chn)} channels + {len(imp)} impacts "
              f"-> {len(ids)} unique nodes "
              f"| snap {min(snap):.2f}..{max(snap):.2f} mm")
        print(f"      ansys ids: {ids.tolist()}")
        # A collocated workbook lists the SAME interface DoF set in both sheets, so
        # the two row counts agree; that is the property the RBE_average / VPT
        # comparison rests on. A non-collocated table is still exportable -- pyhbm
        # builds Ru and Rf from the two row lists independently, and RBE_average is
        # defined for Tu != Tf.T -- so this reports rather than fails, and the
        # answer travels with the descriptor.
        if len(chn) != len(imp):
            print(f"      NOT collocated: {len(chn)} channel rows vs {len(imp)} "
                  f"impact rows -- Tf is not Tu.T, so this table does not support "
                  f"the RBE_average / VPT equivalence check")
        subs[name] = dict(rst=f"FEM/{rst}", full=f"FEM/{full}",
                          interface_node_ids=ids.tolist(),
                          interface_channels=chn,      # -> Tu, in workbook order
                          interface_impacts=imp)       # -> Tf, in workbook order

    # The response DoF is the first row of vpt_A.df_chn and the force DoF the first
    # one after B's six VP rows -- dynamical_system's out_full / inp_full.
    output = io_dof(vp_ordered(sheets["Channels_A"], sheets["VP_Channels"], grp),
                    0, "output")
    excitation = io_dof(vp_ordered(sheets["Impacts_B"], sheets["VP_RefChannels"], grp),
                        N_IF, "excitation")
    print(f"i/o: output '{output['name']}' on A, "
          f"excitation '{excitation['name']}' on B")

    # Recorded so the pyhbm side, and anyone reading a descriptor later, can tell
    # which kind of table it came from without re-reading the workbook.
    collocated = all(len(s["interface_channels"]) == len(s["interface_impacts"])
                     for s in subs.values())

    return dict(
        example=f"testbench_joint_comparison ({workbook})",
        collocated=collocated,
        vp=dict(position=vp_position(xlsx).tolist(), axes=np.eye(3).tolist()),
        joint=dict(k_diag=K_DIAG, c_diag=C_DIAG, alpha_diag=ALPHA_DIAG, beta_diag=BETA_DIAG),
        excitation=dict(**excitation, F0=F0),
        output=output,
        substructures=subs,
    )


if __name__ == "__main__":
    names = [Path(n).stem for n in sys.argv[1:]]
    if not names:                               # bare call: fill in what is missing
        names = [n for n in measurements.available()
                 if not descriptor_path(n).is_file()]
        if not names:
            raise SystemExit("every table in measurements/ already has a "
                             "descriptor; name one to rebuild it")

    for name in names:
        print(f"\n=== {name} ===")
        path = descriptor_path(name)
        path.write_text(json.dumps(build_descriptor(name), indent=2))
        print(f"wrote {path}")
