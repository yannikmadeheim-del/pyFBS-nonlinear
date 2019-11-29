# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:16:50 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
Acknowledgment:
    This function uses the library vtki created by Alex Kaszynski. However, the author modifed its functions to mould them according to his requirements.
    
Accepts:
    1. stl data that is created in function "data_reader"
    2. xyz list created in function "data_reader"
    
Returns:
    1. The coordinates of the nodes interactively selected by the user.
    
When is the function called:
    This function is called for systems where the input is not read from ansys full file. 
    
Logic:
    Creates a vtki plotter object and adds geometry and nodes on the same vtk figure.
# =============================================================================
"""

import vtki

def plotter(stl_data, xyz, x):
    
    plotter = vtki.Plotter()
    plotter.add_mesh(stl_data)
    plotter.add_mesh(xyz, color = "orange")
#    plotter.add_point_labels(xyz[0,:,:],x)
    plotter.plot()
    
    selected_nodes_hammer = plotter.selected_nodes_hammer
    selected_nodes_sensor = plotter.selected_nodes_sensor

    return selected_nodes_hammer, selected_nodes_sensor