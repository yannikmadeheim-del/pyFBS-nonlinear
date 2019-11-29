# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:15:25 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
Accepts:
    1. Node object.
    
Returns:
    1. DOF reference table.
    
When is the function called:
    This function is called for the systems where the input is not read from the ansys full file. 
    
Purpose:
    As the input is not read from the ansys full file, the dof reference table is created manually instead. This is done to keep the subsequent structure of the code same. (for creation of system's coupled FRF matrices). For more explanation about dof reference table consult the "Node" class.
# =============================================================================

"""

import numpy as np

def create_data_table(self):
    dofs_per_node = self.oDOF.ndof_per_node
    dofs = np.arange(0,dofs_per_node)
    iterator = 0
    self.dof_reference_table = np.zeros((len(self.nodes)*dofs_per_node,2),int)
    for i in range (0,len(self.nodes)):
        for j in range (0,dofs_per_node):    
            self.dof_reference_table[iterator + j][0] = self.nodes[i] 
            self.dof_reference_table[iterator + j][1] = dofs[j]
        iterator = iterator + j + 1