# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 
    1. FRF object.
    2. MCK object.
    3. Frequencies for which FRF matrix needs to be created.
    
Returns the uncoupled FRF matrix in the form of a dictionary.

# =============================================================================

"""

import numpy as np

def to_FRF(self,oMCK,frequencies):
    for freq in frequencies:
        omega = 2*np.pi*freq
        Z_Matrix = -omega*omega*oMCK.mass_matrix.values + oMCK.stiffness_matrix.values
        self.Y_uncoupled[freq] = np.linalg.inv(Z_Matrix)