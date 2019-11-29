# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""
import scipy.io as sp_io

"""
# =============================================================================
# Loads the .mat data.
# =============================================================================
"""
def load(mat_file):
    
    data = sp_io.loadmat(mat_file)
    
    return data 
    
    
        