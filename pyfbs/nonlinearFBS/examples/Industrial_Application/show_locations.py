"""
3D view of the measurement layout: Subframe + ControlArm.

(a) Subframe and (b) ControlArm: sensor channels (blue), impacts (red) and the two
virtual points (green) used for the VPT.
(c) Coupled structure: virtual points, excitation F (ControlArm, group 20), output
sensor channels x, y, z (Subframe, group 10) and the global coordinate system.
Units: mm.
"""

from pathlib import Path

import pandas as pd

import pyfbs

HERE = Path(__file__).resolve().parent
DATA = HERE / "Subframe_Control_Arm"
xlsx = DATA / "Measurements" / "measurements.xlsx"

G_RESPONSE, G_EXCITE = 10, 20        # output sensor (Subframe), excitation (ControlArm)
POS = ["Position_1", "Position_2", "Position_3"]
DIR = ["Direction_1", "Direction_2", "Direction_3"]

sheets = {name: pd.read_excel(xlsx, sheet_name=name)
          for name in ("Channels_Subframe", "Impacts_Subframe",
                       "Channels_ControlArm", "Impacts_ControlArm", "VP_Channels")}
df_vp = sheets["VP_Channels"]

view = pyfbs.display.View3D(show_origin=False, shape=(1, 3), window_size=(1800, 700),
                            title="Subframe + ControlArm")


def new_subplot(idx, title, parts):
    """Activate one subplot, draw its STLs and start empty channel/impact/VP collections."""
    view.plot.subplot(0, idx)
    view.plot.add_text(title, position="upper_left", font_size=10, color="k")
    for part in parts:
        view.add_stl(str(DATA / "STL" / f"{part}.STL"), name=f"{part}_{idx}", opacity=0.6)
    # show_chn/show_imp/show_vp merge everything into one collection: reset it per subplot
    view.global_chn, view.global_imp, view.global_vps = None, None, []


# --- (a), (b) substructures with the VPT layout -----------------------------------
for idx, part in enumerate(["Subframe", "ControlArm"]):
    new_subplot(idx, f"({'ab'[idx]}) {part}", [part])
    view.show_chn(sheets[f"Channels_{part}"], size=15)
    view.show_imp(sheets[f"Impacts_{part}"], size=15)
    view.show_vp(df_vp, size=6)
    view.plot.isometric_view()

# --- (c) coupled structure with excitation and output ----------------------------
new_subplot(2, "(c) Coupled structure", ["Subframe", "ControlArm"])
chn_out = sheets["Channels_Subframe"].query("Grouping == @G_RESPONSE")
imp_exc = sheets["Impacts_ControlArm"].query("Grouping == @G_EXCITE")
size = 40
view.show_vp(df_vp, size=6)
view.show_chn(chn_out, size=size)
view.show_imp(imp_exc, size=size)
# labels at the free arrow ends: channels start at the sensor, impacts end at the impact point
view.plot.add_point_labels(chn_out[POS].to_numpy() + size * chn_out[DIR].to_numpy(),
                           ["x", "y", "z"], font_size=14, show_points=False,
                           shape=None, always_visible=True)
view.plot.add_point_labels(imp_exc[POS].to_numpy() - size * imp_exc[DIR].to_numpy(),
                           ["F"], font_size=14, show_points=False,
                           shape=None, always_visible=True)
view.plot.add_axes()                 # global coordinate system
view.plot.isometric_view()

view.plot.app.exec_()
