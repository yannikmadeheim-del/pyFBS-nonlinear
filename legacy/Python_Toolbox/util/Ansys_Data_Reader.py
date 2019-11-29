# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
Acknowledgment:
    This function uses the library pyansys created by Alex Kaszynski.

read_ansys_full_file: 
    
    1. Reads ansys full file using reader from pyansys.
    2. Reads ansys result file if ansys result file (with extension .rst) is provided. This file contains alot of useful information
    but here only the node order is imported. Node order means the order of noddes in global stiffness and mass matrices.
    3. If nodes_from_matlab are provided, the read stiffness and mass matrices are rearranged in this order.
    
sort_matrices_from_nodes:
    
    1. Rearranges read stiffness and mass matrices in the given node order.
    
sort_matrices_from_nodal_coordinates:
    
    1. This function is incomplete at the moment. It compares the nodal coordinates from ansys result file and given coordinates
    (The read stiffness and mass matrices should be sorted according to this list). After comparision, node numbers are 
    assigned to the list of given coordinates after which rearrangment is performed. 
# =============================================================================

"""


from scipy import sparse
from xarray import DataArray

import pyansys

def read_ansys_full_file(full_file_name, result_file_name, nodes_from_matlab = None):
    
    fobj = pyansys.read_binary(full_file_name)
    dof_reference_table, k, m = fobj.load_km(sort=False)  # returns upper triangle only
    ndof_per_node = fobj.ndof[0]
    
    k += sparse.triu(k, 1).T.todense()
    m += sparse.triu(m, 1).T.todense()
    
#    k, m = sort_matrices_from_nodes(k.todense(), m.todense(), dof_reference_table, nodes_from_matlab, ndof_per_node)
        
    nodes_and_coords_from_rst_file = {}
    
    result = pyansys.read_binary(result_file_name)
    coords_from_rst_file = result.geometry["nodes"]
    
    nodes_and_coords_from_rst_file = {key:coords_from_rst_file[key-1][0:3] for key in dof_reference_table[::3][:,0]\
                                       if key not in nodes_and_coords_from_rst_file}
    
    if nodes_from_matlab is not None:
        
        nodes_and_coords_mat_file = {key[1]:nodes_and_coords_from_rst_file[key_1] for key in nodes_from_matlab for key_1 in \
                                     nodes_and_coords_from_rst_file if key[1] == key_1}
        
    else:
        
        nodes_and_coords_mat_file = nodes_and_coords_from_rst_file
        
    
    return k, m, dof_reference_table, ndof_per_node, nodes_and_coords_mat_file

def sort_matrices_from_nodes(k, m, dof_reference_table, nodes_from_matlab, ndof_per_node):
    
    dim_names = ["nodes_and_dofs", "subcases"] 
    
    original_nodes_with_DOFs = [",".join([str(node), str(dof)]) for node, dof in zip(dof_reference_table[:,0], dof_reference_table[:,1])]
    
    matlab_nodes_with_DOFs = [",".join([str(node), str(dof-1)]) for node, dof in zip(nodes_from_matlab[:,1], nodes_from_matlab[:,2])]
    
    column_names = [original_nodes_with_DOFs, original_nodes_with_DOFs]
    
    K_xr = DataArray(k, coords=column_names, dims=dim_names)
    
    K_xr = K_xr.loc[dict(nodes_and_dofs = matlab_nodes_with_DOFs)]
    K_xr = K_xr.loc[dict(subcases = matlab_nodes_with_DOFs)]
    
    M_xr = DataArray(m, coords=column_names, dims=dim_names)
    
    M_xr = M_xr.loc[dict(nodes_and_dofs = matlab_nodes_with_DOFs)] #betrachten
    M_xr = M_xr.loc[dict(subcases = matlab_nodes_with_DOFs)]
    
    return K_xr, M_xr
    
    