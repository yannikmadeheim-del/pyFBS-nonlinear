import pytest

import pyvista
import pyvistaqt

import pyFBS

from pyvistaqt import BackgroundPlotter, MainWindow, QtInteractor


def test1(qtbot):
    plotter = BackgroundPlotter()



def test2(qtbot):
    plotter = BackgroundPlotter(off_screen=False)



def test3(qtbot):
    plotter = BackgroundPlotter(show = True)

def test4(qtbot):
    plotter = pyFBS.view3D()
    
def test5(qtbot):
    plotter = pyFBS.download_lab_testbench()
    
def test5(qtbot):
    plotter = pyFBS.download_automotive_testbench()