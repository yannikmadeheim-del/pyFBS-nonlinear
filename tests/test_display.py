import pytest
import pyvista
import pyvistaqt
import pyFBS
from pyvistaqt import BackgroundPlotter, MainWindow, QtInteractor



@pytest.mark.parametrize("show_origin", [False, True])
@pytest.mark.parametrize("show_axes", [False, True])
def test_display(show_origin,show_axes):
    view3D = pyFBS.view3D(show_origin = show_origin,show_axes = show_axes, title = "test")
    assert(view3D.plot is not None)
    view3D.plot.close()

def test_add_stl():
    view3D = pyFBS.view3D()
    stl_file = "./" + pyFBS.IO.LAB_FOLDER + "/" + "STL" + "/" + pyFBS.IO.LAB_FILES["STL"][0]
    mesh = view3D.add_stl(stl_file)
    assert(mesh is not None)
    view3D.plot.close()
