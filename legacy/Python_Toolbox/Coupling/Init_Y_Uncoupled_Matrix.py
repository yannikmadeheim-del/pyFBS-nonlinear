# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================
# Creates a block diagonal y_uncoupled for the complete structure.
Steps:
    1. Creates a list "total_nodes_and_dofs_without_BCs" for the block diagonal uncoupled system. For more information about this list consult the definition of class "Coupling"
    2. At each frequency, creates a block diagonal y_uncoupled matrix for the complete system to be coupled. It is stored in the form of a dictionary with frequency as the key and uncoupled y matrix in the form of a xarray.
# =============================================================================
"""
from scipy.sparse import block_diag
import xarray as xr
import copy

def init_y_uncoupled_matrix(self):
    
    create_total_dof_and_subcase_list(self)
    
    dim_names = ["total_nodes_and_dofs_without_BCs", "subcases"]
    column_names = [self.total_nodes_and_dofs_without_BCs, self.total_subcases]
    
    block_matrices_list = list()
    
    for freq in self.frequencies:
        for i in range (0,len(self.oFRF)):
            block_matrices_list.append(self.oFRF[i].Y_uncoupled[freq])
        Temp = block_diag(block_matrices_list).toarray()   
        y_uncoupled_as_xarray = xr.DataArray(data=Temp, coords=column_names, dims=dim_names)
        self.y_uncoupled[freq] = y_uncoupled_as_xarray
        block_matrices_list = list()
        
def create_total_dof_and_subcase_list(self):
    
    for i in range (0, len(self.oFRF)):
        Nodes_and_dofs = copy.deepcopy(self.oFRF[i].oNode.nodes_and_dofs_without_BCs)
        for j in range (0,len(Nodes_and_dofs)):
            Nodes_and_dofs[j] = Nodes_and_dofs[j] + "," + str(i+1) 
        self.total_nodes_and_dofs_without_BCs = self.total_nodes_and_dofs_without_BCs + Nodes_and_dofs
        del Nodes_and_dofs
    
    for i in range (0, len(self.oFRF)):
        subcases = copy.deepcopy(self.oFRF[i].subcases)
        for j in range (0,len(subcases)):
            subcases[j] = str(subcases[j]) + "," + str(i+1)
        self.total_subcases = self.total_subcases + subcases
        del subcases
    