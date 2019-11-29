# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================



# =============================================================================
    
"""
from numpy import arange
from numpy import asarray
import matplotlib.pyplot as plt
from Python_Toolbox.Channel.Channel import Channel

def plot(Data, dofs_i, Time_Block):
    
    if isinstance(dofs_i[0], Channel):
        DOFs_i = [Data.Channels.index(item) for item in dofs_i]
        ylabel = Data.Channels[0].Quantity
    else:
        DOFs_i = dofs_i
        
    plt.figure()   
    plot_data = Data.Data[DOFs_i, Time_Block, :].transpose()
    plt.plot(plot_data)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)    
    