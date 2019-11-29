# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================


# =============================================================================

"""

def selection_through_grouping(Data, x):
    
    if x[1] == '<=':
        
        Data.Channels = [item for item in Data.Channels if item.Grouping <= x[2]]
        Data.nChannels = len(Data.Channels)
        
        if Data.Data.ndim == 2:   
            Data.Data = Data.Data[0:Data.nChannels,:]
        elif Data.Data.ndim == 3:
            Data.Data = Data.Data[0:Data.nChannels,:,:]
        
    elif x[1] == '>=':
        
        New_Channels = [item for item in Data.Channels if item.Grouping >= x[2]]
        New_nChannels = len(New_Channels)
        
        Difference_nChannels = Data.nChannels - New_nChannels
        
        Data.Channels = New_Channels
        Data.nChannels = New_nChannels
        
        if Data.Data.ndim == 2:   
            Data.Data = Data.Data[Difference_nChannels:,:]
        elif Data.Data.ndim == 3:
            Data.Data = Data.Data[Difference_nChannels:,:,:]
        
    return Data