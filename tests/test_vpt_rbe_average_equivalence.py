"""
The pyFBS VPT and the pyhbm RBE_average condensation must build the SAME
interface transformation.

Both compute Tu = pinv(Ru' Wu Ru) Ru' Wu and Tf = pinv(Wf) Rf pinv(Rf' pinv(Wf) Rf)
from their own copy of the measurement table, in their own source tree. Nothing
enforces that they stay in step, and a silent divergence would invalidate every
RBE_average-vs-VPT comparison in results/ -- so pin it here.

The collocated tables list the SAME interface DoF set in both sheets but in a
different row order (channels+impacts against impacts+channels), so Tf equals
Tu.T only after aligning rows by (position, direction). That alignment is the
real invariant; a naive allclose(Tf, Tu.T) passes on the sensor table and fails
on the all-rows one for reasons that have nothing to do with correctness.

Node snapping is NOT under test here -- both sides are fed the same snapped
nodes on purpose, so a failure points at the R/T algebra. pyhbm's own
check_vpt_rows guards snapping against the pyFBS-written descriptor at runtime.

Run with --noconftest: tests/conftest.py imports the legacy capitalized package
and dies at collection.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyfbs.interface import VPT

REPO = Path(__file__).resolve().parents[1]
THESIS_ROOT = REPO.parents[1]
PYFBS_MEASUREMENTS = (REPO / "pyfbs" / "nonlinearFBS" / "examples"
                      / "testbench_joint_comparison" / "measurements")
PYHBM_EXAMPLE = (THESIS_ROOT / "code" / "pyhbm" / "examples"
                 / "testbench_joint_comparison_CB")
NPZ_DIR = PYHBM_EXAMPLE.parent / "testbench_cubicSpring_CB"

WORKBOOKS = ("collocated_all_coupling_example",
             "collocated_sensor_coupling_example")
SIDES = ("A", "B")
GROUPING = 10

pytestmark = pytest.mark.skipif(
    not (PYHBM_EXAMPLE.is_dir() and NPZ_DIR.is_dir()),
    reason="pyhbm checkout or its cached .npz substructures not available")


@pytest.fixture(scope="module")
def cb():
    """pyhbm's shared CB infrastructure, via the example's re-export shim."""
    sys.path.insert(0, str(PYHBM_EXAMPLE))
    from dynamical_system import cb as module
    return module


def nodes_of(side):
    return np.load(NPZ_DIR / f"{side}_full.npz", allow_pickle=True)["nodes"]


def vp_position(xlsx):
    return pd.read_excel(xlsx, sheet_name="VP_Channels")[
        ["Position_1", "Position_2", "Position_3"]].to_numpy(float)[0]


def vp_columns(df_vp, df):
    """The six VP columns of ru/rf, after VPT's ascending-grouping sort.

    Non-interface rows are appended as an identity pass-through block, and the
    columns are then sorted by grouping, so where the VP block lands depends on
    the other grouping numbers present.
    """
    order = np.concatenate((
        df_vp.sort_values(["Grouping"], kind="stable")["Grouping"].to_numpy(),
        df.loc[df["Grouping"] != GROUPING, "Grouping"].to_numpy()))
    return np.argsort(np.argsort(order, kind="stable"), kind="stable")[:6]


def dof_key(row):
    """(position, unit direction) identifying the physical DoF of a row."""
    pos = np.round([row.Position_1, row.Position_2, row.Position_3], 9)
    d = np.array([row.Direction_1, row.Direction_2, row.Direction_3], float)
    return tuple(pos), tuple(np.round(d / np.linalg.norm(d), 6))


@pytest.mark.parametrize("workbook", WORKBOOKS)
def test_workbook_copies_identical(workbook):
    """The two projects keep their own copy; they must not drift."""
    paths = (PYFBS_MEASUREMENTS / f"{workbook}.xlsx",
             PYHBM_EXAMPLE / "measurements" / f"{workbook}.xlsx")
    digests = {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in paths}
    assert len(set(digests.values())) == 1, digests


@pytest.fixture(scope="module")
def built(cb):
    """{(workbook, side): Ru/Rf/Tu/Tf from each pipeline, plus the row keys}."""
    out = {}
    for workbook in WORKBOOKS:
        xlsx = PYFBS_MEASUREMENTS / f"{workbook}.xlsx"
        vp = vp_position(xlsx)
        df_vp = pd.read_excel(xlsx, sheet_name="VP_Channels")
        df_vpref = pd.read_excel(xlsx, sheet_name="VP_RefChannels")
        for side in SIDES:
            nodes = nodes_of(side)
            rows = cb.read_vpt_rows(xlsx, side, nodes)
            Ru_h = cb.vpt_idm(nodes, rows["channels"], vp)
            Rf_h = cb.vpt_idm(nodes, rows["impacts"], vp)
            Tu_h, Tf_h = cb.vpt_transformations(Ru_h, Rf_h)

            chn = pd.read_excel(xlsx, sheet_name=f"Channels_{side}")
            imp = pd.read_excel(xlsx, sheet_name=f"Impacts_{side}")
            for frame in (chn, imp):          # same snapped nodes on both sides
                p = frame[["Position_1", "Position_2",
                           "Position_3"]].to_numpy(float)
                near = np.argmin(
                    np.linalg.norm(p[:, None, :] - nodes[None], axis=2), axis=1)
                frame[["Position_1", "Position_2", "Position_3"]] = nodes[near]
            vpt = VPT(chn, imp, df_vp, df_vpref)
            ci = np.flatnonzero(chn["Grouping"].to_numpy() == GROUPING)
            ii = np.flatnonzero(imp["Grouping"].to_numpy() == GROUPING)
            out[(workbook, side)] = dict(
                Ru_h=Ru_h, Rf_h=Rf_h, Tu_h=Tu_h, Tf_h=Tf_h,
                Ru_p=vpt.ru[np.ix_(ci, vp_columns(df_vp, chn))],
                Rf_p=vpt.rf[np.ix_(ii, vp_columns(df_vpref, imp))],
                Tu_p=vpt.tu[np.ix_(vp_columns(df_vp, chn), ci)],
                Tf_p=vpt.tf[np.ix_(ii, vp_columns(df_vpref, imp))],
                chn_keys=[dof_key(chn.iloc[i]) for i in ci],
                imp_keys=[dof_key(imp.iloc[i]) for i in ii])
    return out


CASES = [(w, s) for w in WORKBOOKS for s in SIDES]


@pytest.mark.parametrize("workbook,side", CASES)
def test_idm_matrices_agree(built, workbook, side):
    b = built[(workbook, side)]
    assert np.allclose(b["Ru_h"], b["Ru_p"], atol=1e-15)
    assert np.allclose(b["Rf_h"], b["Rf_p"], atol=1e-15)


@pytest.mark.parametrize("workbook,side", CASES)
def test_transformations_agree(built, workbook, side):
    b = built[(workbook, side)]
    for name in ("Tu", "Tf"):
        h, p = b[f"{name}_h"], b[f"{name}_p"]
        assert np.abs(h - p).max() / np.abs(p).max() < 1e-10, name


@pytest.mark.parametrize("workbook,side", CASES)
def test_transformations_invert_the_idm(built, workbook, side):
    """Tu Ru = I and Rf' Tf = I: both are exact on rigid interface motion."""
    b = built[(workbook, side)]
    assert np.allclose(b["Tu_p"] @ b["Ru_p"], np.eye(6), atol=1e-9)
    assert np.allclose(b["Rf_p"].T @ b["Tf_p"], np.eye(6), atol=1e-9)


@pytest.mark.parametrize("workbook,side", CASES)
def test_collocated_up_to_row_order(built, workbook, side):
    """Channels and impacts span the same DoF set, and Tf = Tu' once aligned."""
    b = built[(workbook, side)]
    assert set(b["chn_keys"]) == set(b["imp_keys"])
    perm = [b["chn_keys"].index(k) for k in b["imp_keys"]]
    aligned = b["Tf_p"][np.argsort(perm), :]
    assert np.abs(aligned - b["Tu_p"].T).max() / np.abs(b["Tu_p"]).max() < 1e-10
