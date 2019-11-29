# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""
from Python_Toolbox.Blocks.Blocks import Blocks

def create_block_list(t, length, Name = None, overlap = None):
    
#%% Inputs    
    
    i = 0
    List_of_Blocks = list()

    if overlap == None:
        overlap = 1
        
    spacing = length/overlap
    
#%% For creating a list of blocks    
    
    if isinstance(t, (list,)):
        t2 = min(t)
        t1 = min(t) - spacing
        while(t2 < (max(t) - spacing)):
            
            obj = Blocks(t1 + spacing, length)
            obj.Name = "Block_" + str(i) + "sec"
            List_of_Blocks.append(obj)
            t2 = obj.t2
            t1 = obj.t1
            i = i + 1
#%% For creating only one block
            
    else:
        obj = Blocks(t, length, Name)
        List_of_Blocks.append(obj)
        
    return List_of_Blocks