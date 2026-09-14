"""
3D overview of one measurement table: STL, sensors, impacts, channels, VP.

Run it to see which DoFs a table actually feeds the virtual point -- that is the
only thing that differs between the tables in this folder, and it is what sets
the size of Ru/Rf and hence how over-determined the rigid interface fit is.

    python -m pyfbs.nonlinearFBS.examples.testbench_joint_comparison.measurements.view_measurements
    ... --workbook collocated_sensor_coupling_example --screenshot overview.png
"""

import argparse

import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements
# the STL sits next to the shared FE data; reuse the example's resolver so the
# download-on-demand fallback stays in one place
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison.dynamical_system import (
    _resolve_data_dir,
)

VPT_GROUPING = 10           # the grouping that the VP transformation consumes
SCALE = 1000                # workbook is in metres, the STL meshes in millimetres
MUTED = "#9a9a9a"           # rows outside the interface
# the accelerometer bodies are drawn to physical size and swamp the default
# size=10 arrows, which is exactly the detail these tables differ in
ARROW = 30


def _split(df):
    """(interface rows, remaining rows) of a channel/impact sheet."""
    if "Grouping" not in df:
        return df.iloc[:0], df
    interface = df["Grouping"] == VPT_GROUPING
    return df[interface], df[~interface]


def show(view, idx, title, stl, df_acc, df_chn, df_imp, df_vp):
    view.plot.subplot(0, idx)
    view.plot.isometric_view()
    view.plot.add_text(title, position="upper_left", font_size=10, color="k",
                       font="times", name=f"title_{idx}")
    view.add_stl(str(stl), color="#83afd2", name=f"stl_{idx}")

    # show_chn/show_imp keep ONE merged mesh and re-add it to whichever subplot
    # is active, so appending across subplots would pile every structure's arrows
    # onto the last one. The collection has to be emptied once per subplot.
    #
    # overwrite=True is the API for that, but it routes through show_hide_*,
    # and show_hide_impacts still treats global_imp as a LIST of triples while
    # show_imp assigns the flat [mesh, actor, colors] -- so it raises. Clearing
    # the attribute directly reaches the same fresh-collection branch without
    # going through it. Empty frames are skipped as well: pyvista rejects a
    # (0, 3) vectors array, and the AB sheets carry no interface rows at all.
    chn_if, chn_rest = _split(df_chn)
    imp_if, imp_rest = _split(df_imp)

    def draw(attr, empty, groups, show, **shared):
        setattr(view, attr, empty)
        for df, kwargs in groups:
            if df is not None and len(df):
                show(df, scale=SCALE, overwrite=False, **shared, **kwargs)

    # the non-interface rows are drawn muted, so the DoFs that reach the virtual
    # point are the ones that stand out
    draw("global_acc", [], [(df_acc, {})], view.show_acc)
    draw("global_chn", None, [(chn_rest, {"color": MUTED}), (chn_if, {})],
         view.show_chn, size=ARROW)
    draw("global_imp", None, [(imp_rest, {"color": MUTED}), (imp_if, {})],
         view.show_imp, size=ARROW)
    draw("global_vps", [], [(df_vp if len(chn_if) else None, {})], view.show_vp)
    return len(chn_if), len(imp_if)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=measurements.DEFAULT,
                        help=f"one of: {', '.join(measurements.available())}")
    parser.add_argument("--screenshot", help="save this PNG and exit; the "
                                             "window still opens briefly, "
                                             "off_screen renders at 100x30")
    parser.add_argument("--size", default="1800x700",
                        help="window size WxH (default: %(default)s)")
    args = parser.parse_args()

    xlsx = measurements.resolve(args.workbook)
    stl_dir = _resolve_data_dir() / "STL"
    sheets = {name: pd.read_excel(xlsx, sheet_name=name)
              for name in ("Sensors_A", "Channels_A", "Impacts_A",
                           "Sensors_B", "Channels_B", "Impacts_B",
                           "Sensors_AB", "Channels_AB", "Impacts_AB",
                           "VP_Channels")}

    width, height = (int(v) for v in args.size.lower().split("x"))
    view = pyfbs.display.View3D(show_origin=False, show_axes=False,
                                shape=(1, 3), title=f"Overview - {args.workbook}",
                                window_size=(width, height))
    for idx, sub in enumerate(("A", "B", "AB")):
        n_chn, n_imp = show(
            view, idx, f"{sub} structure", stl_dir / f"{sub}.stl",
            sheets[f"Sensors_{sub}"], sheets[f"Channels_{sub}"],
            sheets[f"Impacts_{sub}"], sheets["VP_Channels"])
        print(f"[{sub}] {n_chn} interface channels / {n_imp} interface impacts "
              f"(Grouping == {VPT_GROUPING})")

    if args.screenshot:
        # the Qt widget only takes its real size once the window has been shown,
        # so pump the event loop before grabbing the frame
        view.plot.app.processEvents()
        view.plot.render()
        view.plot.screenshot(args.screenshot)
        view.plot.close()
        print(f"wrote {args.screenshot}")
    else:
        view.plot.app.exec_()


if __name__ == "__main__":
    main()
