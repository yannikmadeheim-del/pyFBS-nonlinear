# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts:
    1. DOFs per node.
    2. Stiffness matrix file in .mat format (MATLAB output format).
    3. Mass matrix file in .mat format (MATLAB output format).
    4. STL file for creation of geometry.
    5. Excel file containing geoemtry's nodes and their coordinates.

Returns:
    1. Total DOFs per node.
    2. The total nodes of the substructure.
    3. Stiffness Matrix.
    4. Mass Matrix.
    5. stl_data that will be used subsequently for creation of geometry through vtk.
    6. Nodal x coordiantes.
    7. Nodal y coordiantes.
    8. Nodal z coordiantes.
    9. Horizontally concatenated nodal xyz coordiantes.
    
When is the function called:
    This function is called for the systems where the input is not read from the ansys full file. 
    
# =============================================================================
"""


import pandas as pd
import scipy.io as sp
import vtki
import numpy as np

def data_reader(dofs_per_node, K_file_name, M_file_name, stl_file_name, xl_file_name):
    ndof_per_node = dofs_per_node
    
    K = sp.loadmat(K_file_name)
    K = K["K"]
    
    M = sp.loadmat(M_file_name)
    M = M["M"]
    
    stl_data = vtki.PolyData(stl_file_name)
    
    xls = pd.ExcelFile(xl_file_name)
    data = pd.read_excel(xls, 'Tabelle1')
    
    nodes = data.iloc[:,0].tolist()
    x = data.iloc[:,1].tolist()
    y = data.iloc[:,2].tolist()
    z = data.iloc[:,3].tolist()
    
    xyz = np.dstack((x,y,z))
    
    return ndof_per_node, nodes, K, M, stl_data, x, y, z, xyz