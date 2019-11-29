# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""

# =============================================================================

This function is very similar function to the function create_mass_and_stiffness_matrix_xarrays
present also in the folder MCK_Model. This function creates the mass and stiffness matrices
when the boundary condition are already applied on the system while the other function
"create_mass_and_stiffness_matrix_xarrays", creates the constrained system matrices. 
(Notice the .loc line at the end of the function.)

# =============================================================================

"""


import xarray as xr

def create_mass_and_stiffness_matrix_xarrays_direct(self, k, m):
    
    dim_names = ["nodes_and_dof_without_BCs", "subcases"]
    column_names = [self.oNode.nodes_and_dofs_without_BCs, self.oNode.nodes_and_dofs_without_BCs]
    
    self.stiffness_matrix = xr.DataArray(data=k, coords=column_names, dims=dim_names)
    self.mass_matrix = xr.DataArray(data=m, coords=column_names, dims=dim_names)
