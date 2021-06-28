import pytest
import pyvista
import pyvistaqt
import numpy as np
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
    pyFBS.download_lab_testbench()
    stl_file = "./" + pyFBS.IO.LAB_FOLDER + "/" + "STL" + "/" + pyFBS.IO.LAB_FILES["STL"][0]
    mesh = view3D.add_stl(stl_file)
    assert(mesh is not None)
    view3D.plot.close()

def test_show_accelerometers():
    view3D = pyFBS.view3D()
    data_acc = np.asarray([["T", None, None, None, 0,0,0,0,0,0]])
    df_acc = pd.DataFrame(data=data_acc, columns=pyFBS.display.COLUMNS_ACC)
    view3D.show_acc(df_acc)
    assert(view3D.global_acc)
    view3D.plot.close()

def test_show_impacts():
    view3D = pyFBS.view3D()
    data_imp = np.asarray([["T", None, None, None, 0,0,0,0,0,1]])
    df_imp = pd.DataFrame(data=data_imp, columns=pyFBS.display.COLUMNS_CHN)
    view3D.show_imp(df_imp)
    assert(view3D.global_imp)
    view3D.plot.close()

def test_show_channels():
    view3D = pyFBS.view3D()
    data_acc = np.asarray([["T", None, None, None, 0,0,0,0,0,0]])
    df_acc = pd.DataFrame(data=data_acc, columns=pyFBS.display.COLUMNS_ACC)
    df_chn = pyFBS.generate_channels_from_sensors(df_acc)
    assert(df_chn.empty is False)
    view3D.show_chn(df_chn)
    assert(view3D.global_chn)
    view3D.plot.close()

