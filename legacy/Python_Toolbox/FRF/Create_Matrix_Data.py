# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 
    1. FRF Series object.
    2. Number of frequencies.
    
Returns FRF matrix in raw form.

# =============================================================================

"""

import numpy as np
import re

def create_matrix_data(sub_structure, nFreq):
    
    iterator = 0
    
    Names = set([d.DataSets.Y_Channels.Channel_Info.Name for d in sub_structure])
    Nodes_Sensor = sorted([int(re.findall('\d+', item)[0]) for item in Names])
    Nodes_Impact = sorted(set([d.DataSets.Y_Channels.Ref_Channel_Info.Node_Number for d in sub_structure]))
    
    FRF_Matrix = np.zeros((len(Nodes_Sensor), len(Nodes_Impact), nFreq),complex)
    
    Data = [d.DataSets.Y_Channels.Data for d in sub_structure]
    
    if sub_structure[0].DataSets.Y_Channels.Ref_Channel_Info.Name != sub_structure[1].DataSets.Y_Channels.Ref_Channel_Info.Name:
                               
        for i in range (0,len(Nodes_Sensor)):
            for j in range (0,len(Nodes_Impact)):
                FRF_Matrix[i,j,:] = Data[j+iterator]
            iterator = iterator + len(Nodes_Impact)
            
    else:
        
        for j in range (0,len(Nodes_Impact)):
            for i in range (0,len(Nodes_Sensor)):
                FRF_Matrix[i,j,:] = Data[i+iterator]
            iterator = iterator + len(Nodes_Sensor)
                
    return FRF_Matrix
    
    
    
    
        
        
        
        
        