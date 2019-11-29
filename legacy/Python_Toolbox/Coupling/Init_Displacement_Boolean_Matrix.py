# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""
"""

# =============================================================================

# Creates boolean matrix on the basis of variable coupled_nodes provided by the user. 
Example:
    coupled_nodes  = [[(1,2),(2,1)],[(1,2),(3,1)]]
    dofs_per_node = 6
    total_dofs_in_system = 54
    self.boolean_matrix = 12 x 54
    
    counter is used for controlling sign of 1's in boolean matrix. So for each coupled node pair, 1 is assigned for dofs of one node and -1 for correspoding dof of the second node.
    
    iterator is used for controlling row number in the boolean_matrix. When boolean matrix for one coupled node pair has been created ([(1,2),(2,1)]), iterator increases the row number for next coupled node pair ([(1,2),(3,1)]) by 6. (dofs_per_node)
    
    create_index_list_of_coupled_nodes(coupled_nodes, oFRF)
        Checks the location/index of coupled nodes in the y_uncoupled of the complete system.
    
# =============================================================================

"""


import numpy as np

def init_displacement_boolean_matrix(self, VPoint = None):
    if VPoint is not None:
        nDOF = 6*VPoint
    else:
        nDOF = self.nDOF_per_node
    self.boolean_matrix_displacement = np.zeros((len(self.coupled_nodes)*nDOF, len(self.y_uncoupled[self.frequencies[0]])),int)
    index = create_index_list_of_coupled_nodes(self.coupled_nodes, self.oFRF)
    counter = 0
    iterator = 0
    for item in index:  
        for i in range (0,nDOF):
            self.boolean_matrix_displacement[i + iterator][index[counter] + i] = 1*pow(-1,counter)
        counter = counter + 1
        if counter%2 == 0:
            iterator = iterator + self.nDOF_per_node
    
def create_index_list_of_coupled_nodes(coupled_nodes, oFRF):
    index = list()
    for i in range (0,len(coupled_nodes)):
        for item in coupled_nodes[i]:
            item = str(item)
            structure, node = item.split(",")
            structure = structure.replace("(","")
            node = node.replace(")","")
            node = node.replace(" ", "")
            if (int(structure) > 0): #For other than first structures, we need to add the number of nodes of previous structures to get correct DOF in combined y_uncoupled matrix
                Temp = oFRF[int(structure) - 1].oNode.nodes_without_BCs.index(node)
                for i in range (0,int(structure)-1):
                    Temp = Temp +  len(oFRF[i].oNode.nodes_and_dofs_without_BCs)
                index.append(Temp)
            elif (int(structure) == 0):
                index.append(oFRF[int(structure) - 1].oNode.nodes_without_BC.index(node))
        
    return index
        