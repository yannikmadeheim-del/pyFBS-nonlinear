# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:38:51 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

Accepts:
    1. MCK object.
    2. Nodal (x,y,z) coordinates that are interactively selected by the user.
    3. xyz_list created in the function "XYZ_List" in folder util.
    
Returns:
    1. Index of selected nodes in global stiffness and mass matrices which are themselves formed in function "Data_Reader" in folder util.
    
When is the function called:
    This function is called for systems where the input is not read from ansys full file. 
    
Logic:
    Compares xyz coordinates in variable selected_nodes to the xyz coordinates in variable xyz_list. Upon finding common entry, its index is searched in the list "self.nodes_without_BCs" to find the position of node in the global stiffness and mass matrices.
    
# =============================================================================
"""

def get_index_of_selected_nodes(self,selected_nodes,xyz_list):
    indices = list()
    for item in selected_nodes:
        item = str(item)
        if (item in list(xyz_list)):
            indices.append(self.oNode.nodes_without_BCs[xyz_list.index(item)*self.oNode.oDOF.ndof_per_node])
            
    return indices