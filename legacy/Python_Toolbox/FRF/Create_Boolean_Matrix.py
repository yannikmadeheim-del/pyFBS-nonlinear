# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================
This is an improved file with a new way to implement the boolean matrices.
# =============================================================================
"""
import numpy as np

def create_boolean_matrix(oFRF, subsA, subsB):
    
    
    Matching_Channels = [[item1, item2, oFRF.Channels.index(item1), oFRF.Channels.index(item2)] for \
                          item1 in subsA.Channels for item2 in subsB.Channels if \
                         item1.Grouping == item2.Grouping and item1.Node_Number == item2.Node_Number \
                         and item1.Direction_Number == item2.Direction_Number and item1.Name == item2.Name]
    
    Matching_RefChannels = [[item1, item2, oFRF.RefChannels.index(item1), oFRF.RefChannels.index(item2)] for \
                             item1 in subsA.RefChannels for item2 in subsB.RefChannels if \
                     item1.Grouping == item2.Grouping and item1.Node_Number == item2.Node_Number \
                     and item1.Direction_Number == item2.Direction_Number and item1.Name == item2.Name]
    
    B_c = np.zeros((len(Matching_Channels),oFRF.nChannels),int)
    B_e = np.zeros((len(Matching_RefChannels),oFRF.nRefChannels),int)
        
    for i, j in zip(range(len(Matching_Channels)),[item[2] for item in Matching_Channels]):
        B_c[i][j] = 1
    
    for i, j in zip(range(len(Matching_Channels)),[item[3] for item in Matching_Channels]):
        B_c[i][j] = -1
        
    for i, j in zip(range(len(Matching_RefChannels)),[item[2] for item in Matching_RefChannels]):
        B_e[i][j] = 1
    
    for i, j in zip(range(len(Matching_RefChannels)),[item[3] for item in Matching_RefChannels]):
        B_e[i][j] = -1
        
    return B_c, B_e
    
        