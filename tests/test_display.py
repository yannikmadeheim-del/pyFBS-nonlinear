import pytest
import pyvista
import pyvistaqt
import pyFBS
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

def test_add_stl():
    view3D = pyFBS.view3D()
    stl_file = "./" + pyFBS.IO.LAB_FOLDER + "/" + "STL" + "/" + pyFBS.IO.LAB_FILES["STL"][0]
    mesh = view3D.add_stl(stl_file)
    assert(mesh is not None)
    view3D.plot.close()