# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""
from time import time
from math import sqrt
from scipy.sparse.linalg import eigsh
from numpy import zeros
from Python_Toolbox.Modal_Model.Modal_Model import Modal_Model

def to_modal_model(oMCK, nModes):
    
    oModal_Model = Modal_Model()
    oModal_Model.Nodes = oMCK.Nodes
    oModal_Model.nNodes = oMCK.nNodes
    oModal_Model.DOFMapping = oMCK.DOFMapping
    
    if oMCK.K is not None and oMCK.M is not None:
        print("Calculating first " + str(nModes) + " eigen frequencies and eigen vectors ")
        start = time()
        
        omega_squared, oModal_Model.X = eigsh(oMCK.K, k=nModes, M=oMCK.M, sigma = 0, tol = 1e-3)
        
        end = time()
        print("The calcualtion of eigen frequencies took " + str(end - start) + " seconds")
        
        oModal_Model.omega = [round(sqrt(abs(item))) for item in omega_squared]
        oModal_Model.omegad = oModal_Model.omega
        
        oModal_Model.freq = [round(item/(2*3.14)) for item in oModal_Model.omega]
        oModal_Model.freqd = oModal_Model.freq
        
        oModal_Model.zeta = zeros(nModes, float)
        
        oModal_Model.nModes = nModes
    
    return oModal_Model
    
        