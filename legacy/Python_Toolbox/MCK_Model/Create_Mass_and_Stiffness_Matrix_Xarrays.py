# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""

"""

# =============================================================================

Applies the boundary conditions on the free system to make it a constrained system. Notice the 
.loc line a the end of the function.

# =============================================================================

"""

import scipy
import xarray as xr

def create_mass_and_stiffness_matrix_xarrays(self, k, m):
    # make k, m full, symmetric matricies
    k += scipy.sparse.triu(k, 1).T
    m += scipy.sparse.triu(m, 1).T
                          
    k = k.todense()
    m = m.todense()
    
    dim_names = ["nodes_and_dofs_without_BCs", "subcases"]
    column_names = [self.oNode.nodes_and_dofs, self.oNode.nodes_and_dofs]
    
    k_complete = xr.DataArray(data=k, coords=column_names, dims=dim_names)
    m_complete = xr.DataArray(data=m, coords=column_names, dims=dim_names)
    
    self.stiffness_matrix = k_complete.loc[self.oNode.nodes_and_dofs_without_BCs, self.oNode.nodes_and_dofs_without_BCs]
    self.mass_matrix = m_complete.loc[self.oNode.nodes_and_dofs_without_BCs, self.oNode.nodes_and_dofs_without_BCs]