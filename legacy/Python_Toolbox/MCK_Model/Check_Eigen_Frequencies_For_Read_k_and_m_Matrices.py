# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""

# =============================================================================

This is a test function to check if the the mass and stiffness matrix are read correctly
from the ansys full file. Basically, it does an eigen value analysis of the read stiffness and 
mass matrices. Afterwards, the author compared these results with the ansys results and
ascertained the validity of the read data.

# =============================================================================

"""


import numpy as np
import scipy

def check_eigen_frequencies_for_read_k_and_m_matrices(self):
    eig_freq, eig_mode = scipy.linalg.eig(self.stiffness_matrix,self.mass_matrix)
    f = (np.real(eig_freq))**0.5/(2*np.pi)
    f = sorted(f)
    
    return f
        