# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""

import numpy as np
from math import floor

from Python_Toolbox.Time_Blocks.Time_Blocks import Time_Blocks

def from_time_series(oTime_Series, oBlock_List):
 
#%% Assigning inputs
    
    n_b = len(oBlock_List)
    obj_fs = oTime_Series.fs
    obj_Time = oTime_Series.Time
    obj_nTime = oTime_Series.nTime 
    L = oBlock_List[0].l
    N = floor(obj_fs*L)
    tb = np.arange(0,N)/obj_fs
    
#%% Pre-allocate Data
    
    Data = np.zeros((oTime_Series.nChannels, n_b, N),float)
    Real_Time_List = list()
    for item in oBlock_List:
        idx1 = np.argmax(obj_Time >= item.t1)
        idx2 = min(idx1 + N-1, obj_nTime)
        idx = np.arange(idx1, idx2 + 1)
        Real_Time = np.arange(item.t1,item.t2,oTime_Series.dt)
        Real_Time_List.append(Real_Time)
        Data[:,oBlock_List.index(item),0:len(idx)] = oTime_Series.Data[:,0,idx]
        
#%% Create time blocks
    
    oTime_Blocks = Time_Blocks()
    oTime_Blocks.__dict__ = oTime_Series.__dict__.copy()
    oTime_Blocks.Blocks = oBlock_List
    oTime_Blocks.nBlocks = n_b
    oTime_Blocks.Data = Data
    oTime_Blocks.Time = tb
    oTime_Blocks.RealTime = Real_Time_List
    
    return oTime_Blocks