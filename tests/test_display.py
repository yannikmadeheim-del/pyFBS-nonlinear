import pytest
import pyvista
import pyvistaqt
import pyFBS
import pandas as pd
from pyvistaqt import BackgroundPlotter, MainWindow, QtInteractor


#can't test with showing axes - segmentation fault on travis
#@pytest.mark.parametrize("show_axes", [False, True])
@pytest.mark.parametrize("show_origin", [False, True])
def test_display(show_origin):
    view3D = pyFBS.view3D(show_origin = show_origin,show_axes = False, title = "test")
    assert(view3D.plot is not None)
    view3D.plot.close()

def test_add_stl():
    view3D = pyFBS.view3D()
    stl_file = "./" + pyFBS.IO.LAB_FOLDER + "/" + "STL" + "/" + pyFBS.IO.LAB_FILES["STL"][0]
    mesh = view3D.add_stl(stl_file)
    assert(mesh is not None)
    view3D.plot.close()

def test_show_accelerometers():
    view3D = pyFBS.view3D()
    path_to_xlsx = "./" + pyFBS.IO.LAB_FOLDER + "/" + "Measurements" + "/" + "AM_measurements.xlsx"
    df_acc = pd.read_excel(path_to_xlsx, sheet_name='Sensors_AB')
    view3D.show_acc(df_acc)
    assert(view3D.global_acc)
    view3D.plot.close()

def test_show_impacts():
    view3D = pyFBS.view3D()
    path_to_xlsx = "./" + pyFBS.IO.LAB_FOLDER + "/" + "Measurements" + "/" + "AM_measurements.xlsx"
    df_imp = pd.read_excel(path_to_xlsx, sheet_name='Channels_AB')
    view3D.show_imp(df_imp)
    assert(view3D.global_imp)
    view3D.plot.close()

def test_show_channels():
    view3D = pyFBS.view3D()
    path_to_xlsx = "./" + pyFBS.IO.LAB_FOLDER + "/" + "Measurements" + "/" + "AM_measurements.xlsx"
    df_acc = pd.read_excel(path_to_xlsx, sheet_name='Sensors_AB')
    df_chn = pyFBS.generate_channels_from_sensors(df_acc)
    assert(df_chn.empty is False)
    view3D.show_chn(df_chn)
    assert(view3D.global_chn)
    view3D.plot.close()

