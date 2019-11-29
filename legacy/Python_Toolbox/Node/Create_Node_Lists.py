# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
If Type = Direct, then the BCs are already applied on the stiffness and mass matrices.
If Type = None, then the BCs need to be applied on the system.
# =============================================================================
"""

def create_node_list(self, Type = None):
    self.nodes_and_dofs = stack_nodes_and_dofs(self.dof_reference_table)
    if Type == "Direct":
        self.nodes_and_dofs_without_BCs = self.nodes_and_dofs
    else:
        BC_nodes_and_dofs = stack_nodes_and_dofs(self.BC_nodes_and_dofs)
        self.nodes_and_dofs_without_BCs = [item for item in self.nodes_and_dofs if item not in                                  BC_nodes_and_dofs]
        
    
    for item in self.nodes_and_dofs_without_BCs:
        item = item.split(",")
        self.nodes_without_BCs.append(item[0])

def stack_nodes_and_dofs(array):
    Temp = list()
    Temp_list = list()
    for i in range (0,len(array)):
        Temp.append(str(array[i][0]))
        Temp.append(str(array[i][1]))
        string = ",".join(Temp)
        Temp_list.append(string)
        Temp = list()
        
    return Temp_list