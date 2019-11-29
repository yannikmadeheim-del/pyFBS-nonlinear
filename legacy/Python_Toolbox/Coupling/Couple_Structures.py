# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""
import numpy as np
import xarray as xr

"""
# =============================================================================
# Couples any given number of sub structures.
# If the user does not specify the force boolean matrix, then it is equal to the displacement boolean matrix.
# =============================================================================
"""
def couple_structures(self):
    
    dim_names = ["total_nodes_with_dof", "subcases"]
    column_names = [self.total_nodes_and_dofs_without_BCs, self.total_subcases]
    
    if not isinstance(self.boolean_matrix_force, np.ndarray):
        self.boolean_matrix_force = self.boolean_matrix_displacement
    
    for freq in self.frequencies:
        
        y_uncoupled = self.y_uncoupled[freq].values
        # first term ist Y*transp(B)
        first_term = y_uncoupled.dot(self.boolean_matrix_force.transpose())
        # second term is B*Y*transp(B)
        interface_flexibility_without_inverse = self.boolean_matrix_displacement.dot(y_uncoupled).dot(self.boolean_matrix_force.transpose())
        
        interface_flexibility = np.linalg.inv(interface_flexibility_without_inverse)
        
        del interface_flexibility_without_inverse
        
        y_coupling = first_term.dot(interface_flexibility).dot(self.boolean_matrix_displacement).dot(y_uncoupled)
        
        del first_term
        del interface_flexibility
        
        y_assembled = (y_uncoupled - y_coupling)
    
        del y_uncoupled
        del y_coupling
        
        y_assembled_as_xarray = xr.DataArray(data=y_assembled, coords=column_names, dims=dim_names)
        
        self.y_coupled[freq] = y_assembled_as_xarray
        del y_assembled_as_xarray
        