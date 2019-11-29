# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""
from copy import deepcopy

def symmetrize(Data):

    if isinstance(Data,object):
        Created_Object = deepcopy(Data)
        Data = Data.Data

        if len(Data) == len(Data[0,:]):
            Created_Object.Data = (Data + Data.transpose(1,0,2))/2
        else:
            print("Error: Symmetrize function works only for square matrices")
            
    else:
        
        if len(Data) == len(Data[0,:]):
            Created_Object = (Data + Data.transpose(1,0,2))/2
        else:
            print("Error: Symmetrize function works only for square matrices")
        
    return Created_Object