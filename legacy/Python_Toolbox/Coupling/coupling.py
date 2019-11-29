# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018
@author: Umer Sherdil Paracha
"""
"""

# =============================================================================

Accepts:    
    1. List of frequencies for which coupled FRF matrix will be created. 
    
    2. List of coupled nodes: [[(Structure number),(Node number)],[(Structure number),(Node number)]] e.g. if there are 3 structures and node 2 of first structure needs to be coupled with the node 1 of second structure and node 2 of second structure needs to be coupled with the node 1 of the third structure then the input should be [[(1,2),(2,1)],[(1,2),(3,1)]].
    
    3. oFRF: List of objects of class FRF which will be used during the coupling procedure.
    
    4. oVP: List of objects of class Virtual Point. This list remains empty till the user explicitly specifies it.
    
Class variables:
    
    1. self.nDOF_per_node: Number of DOFs per node. 
    
    2. self.frequencies: Frequencies for which the coupled FRF matrix will be created.
    
    3. self.total_nodes_and_dofs_without_BCs: List containing the free nodes and dofs of all the structures which will be coupled. So e.g if two structures that will be coupled have a stacked list of 24 free nodes and dof each, then the list "self.total_nodes_and_dofs_without_BCs" will contain 48 entries. For more explanation about free and total nodes consult the Node class.
    
    4. total_subcases: Complete list of subcases of the coupled structure.
    
    5. self.oFRF: List of objects of class FRF which will be used during the coupling procedure.
    
    6. self.oVP: List of objects of class Virtual Point. This list remains empty till the user explicitly specifies it.
    
    7. self.coupled_nodes: Structures and respective nodes that will be coupled.
    
    8. self.boolean_matrix_force: For the general case, where the force boolean matrix is not equal to the displacement boolean matrix. 
    
    9. self.boolean_matrix_displacement: This boolean matrix is a must for coupling. The matrix is created after the user has specified the structures and the nodes to be coupled. 
    
    10. self.y_uncoupled: Dictionary containing the uncoupled FRF matrices of all the structures that will be coupled.
    
    11. self.y_coupled: Dictionary containing the coupled FRF matrix of the complete structure.
    

# =============================================================================


"""

import copy

class coupling(object):
    def __init__(self, frequencies, coupled_nodes, oFRF, oVP = None):

        
        self.nDOF_per_node = oFRF[0].oNode.oDOF.ndof_per_node
        self.frequencies = frequencies
        self.total_nodes_and_dofs_without_BCs = list()
        self.total_subcases = list()
        self.oFRF = list()
        self.oVP = list()
        for i in range (0,len(oFRF)):
            self.oFRF.append(copy.deepcopy(oFRF[i]))
        if oVP is not None:
            for i in range (0,len(oVP)):
                self.oVP.append(copy.deepcopy(oVP[i]))
        self.coupled_nodes = coupled_nodes
        self.boolean_matrix_force = None
        self.boolean_matrix_displacement = None
        self.y_uncoupled = {}
        self.y_coupled = {}
       
    
        