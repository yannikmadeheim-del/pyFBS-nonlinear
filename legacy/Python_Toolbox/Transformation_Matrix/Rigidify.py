# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:43:44 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts:


# =============================================================================

"""

def rigidify(IDM, dofs_i, dofs_j):
    
    if "f" in IDM.Virtual_Channels[0].Name:
        
        P = IDM.R.dot(IDM.T)   
        P = P[dofs_i, :]
        P = P[:, dofs_j]
        
    elif "u" in IDM.Virtual_Channels[0].Name:
        
        P = IDM.R.dot(IDM.T)   
        P = P[dofs_i, :]
        P = P[:, dofs_i]
        

    return P
    