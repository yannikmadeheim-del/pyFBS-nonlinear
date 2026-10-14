"""
3D view of the measurement layout: lab testbench A + B.

(a) Substructure A and (b) substructure B: sensor channels (blue), impacts (red) and
the virtual point (green) used for the VPT; the other channels and impacts are the
reference DoFs.
(c) Coupled structure: virtual point, excitation F (first B reference impact, inp_full),
output channel (first A reference channel, out_full) and the global coordinate system.
Positions in the workbook are in m, the STL files in mm.
"""

import os
from pathlib import Path

import pandas as pd

import pyfbs

HERE = Path(__file__).resolve().parent
LAB = HERE.parents[1] / "lab_testbench"   # shared copy in pyfbs/nonlinearFBS

cwd = os.getcwd()
os.chdir(LAB.parent)                    # download_lab_testbench writes to the cwd
try:
    pyfbs.io.download_lab_testbench()   # skips files that are already there
finally:
    os.chdir(cwd)

xlsx = LAB / "Measurements" / "coupling_example.xlsx"
SCALE = 1000                            # m -> mm
G_REF_A, G_REF_B = 1, 100               # reference groupings of A (output) and B (excitation)
POS = ["Position_1", "Position_2", "Position_3"]
DIR = ["Direction_1", "Direction_2", "Direction_3"]

sheets = {name: pd.read_excel(xlsx, sheet_name=name)
          for name in ("Channels_A", "Impacts_A", "Channels_B", "Impacts_B", "VP_Channels")}
df_vp = sheets["VP_Channels"]

view = pyfbs.display.View3D(show_origin=False, shape=(1, 3), window_size=(1800, 700),
                            title="Lab testbench A + B")


def new_subplot(idx, title, stl):
    """Activate one subplot, draw its STL and start empty channel/impact/VP collections."""
    view.plot.subplot(0, idx)
    view.plot.add_text(title, position="upper_left", font_size=10, color="k")
    view.add_stl(str(LAB / "STL" / f"{stl}.stl"), name=f"{stl}_{idx}", opacity=0.6)
    # show_chn/show_imp/show_vp merge everything into one collection: reset it per subplot
    view.global_chn, view.global_imp, view.global_vps = None, None, []


# --- (a), (b) substructures with the VPT layout -----------------------------------
for idx, sub in enumerate(["A", "B"]):
    new_subplot(idx, f"({'ab'[idx]}) Substructure {sub}", sub)
    view.show_chn(sheets[f"Channels_{sub}"], scale=SCALE, size=15)
    view.show_imp(sheets[f"Impacts_{sub}"], scale=SCALE, size=15)
    view.show_vp(df_vp, scale=SCALE, size=6)
    view.plot.isometric_view()

# --- (c) coupled structure with excitation and output ----------------------------
new_subplot(2, "(c) Coupled structure", "AB")
chn_out = sheets["Channels_A"].query("Grouping == @G_REF_A").iloc[[0]]   # out_full
imp_exc = sheets["Impacts_B"].query("Grouping == @G_REF_B").iloc[[0]]    # inp_full
size = 40
view.show_vp(df_vp, scale=SCALE, size=6)
view.show_chn(chn_out, scale=SCALE, size=size)
view.show_imp(imp_exc, scale=SCALE, size=size)
# labels at the free arrow ends: channels start at the sensor, impacts end at the impact point
view.plot.add_point_labels(SCALE * chn_out[POS].to_numpy() + size * chn_out[DIR].to_numpy(),
                           ["out"], font_size=14, show_points=False,
                           shape=None, always_visible=True)
view.plot.add_point_labels(SCALE * imp_exc[POS].to_numpy() - size * imp_exc[DIR].to_numpy(),
                           ["F"], font_size=14, show_points=False,
                           shape=None, always_visible=True)
view.plot.add_axes()                 # global coordinate system
view.plot.isometric_view()

view.plot.app.exec_()
