# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""

def add_damping(oModal_Model, type_damping, value):
    
    if type_damping == 'modal':       
        for i in range (0,len(oModal_Model.zeta)):
            oModal_Model.zeta[i] = value