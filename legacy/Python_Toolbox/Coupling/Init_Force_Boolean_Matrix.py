# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""
"""

# =============================================================================

# Creates a force boolean matrix on the basis of the dof list provided by the user.

This function is not so sophisticated as the displacement boolean matrix function. For the DOFs to be coupled, user specifies their
position in the uncoupled Y matrix after which this boolean matrix is created. 

    
# =============================================================================

"""


import numpy as np

def init_force_boolean_matrix(self, array):
    self.boolean_matrix_force = np.zeros((len(array),len(self.total_subcases)),int)
    counter = 0
    for i in range (0,len(array)):  
        for item in array[i][:]:
            self.boolean_matrix_force[i][item - 1] = 1*pow(-1,counter)
            counter = counter + 1
    
        