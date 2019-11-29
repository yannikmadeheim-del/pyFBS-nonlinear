# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:40:04 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================
Accepts:
    1. Object of class MCK.
    2. Index of the selected node in the global stiffness and mass matricies.
Returns:
    1. Concatenates the DOF indices of the input node/s so that respective rows and columns can be picked from the stiffness and mass xarrays.
    
When is the function called:
    This function is called for systems where the input is not read from ansys full file. 
    
# =============================================================================
"""

def get_index_with_dofs(self, index):
    index_with_dofs = list()
    Temp = list()
    for item in index:
        for i in range (0,self.oNode.oDOF.ndof_per_node):
            Temp.append(str(item))
            Temp.append(str(i))
            index_with_dofs.append(",".join(Temp))
            Temp = list()
    
    return index_with_dofs