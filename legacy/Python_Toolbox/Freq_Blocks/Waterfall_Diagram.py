# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""
from numpy import log10, meshgrid, arange
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from Python_Toolbox.Math.Weighting import weighting

def waterfall_diagram(Data, ch_i):
    
    data_to_plot = Data.Data[ch_i, : , :]
    freq = Data.Freq
    
    if Data.Channels[ch_i].Unit == "m/s^2":
        dBref = 1e-6
    elif Data.Channels[ch_i].Unit == "Pa":
        dBref = 20e-6
        A = weighting(Data, "A")
        A[A < -1e300] = 0
        A[A > 1e300] = 0
        
        if A[0] == 0:
            data_to_plot = data_to_plot[1:] + A[1:]
            freq = freq[1:]
            
    data_to_plot_db = 20*log10(abs(data_to_plot/dBref))
    z = [item.z for item in Data.Blocks]
    
    fig = plt.figure()
    ax = fig.gca(projection = "3d")
    freq, z_mesh = meshgrid(freq,z)
    surf = ax.plot_surface(freq, z_mesh, data_to_plot_db, cmap=cm.coolwarm)
    ax.view_init(azim=0, elev=90)
    plt.xlabel("Frequency")
    plt.ylabel("Time")
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 10
    ax.grid(False)
    ax.set_zticks([])
    ax.tick_params(axis='x', which='major', pad=10)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("WaterFall", dpi=300)
    
    