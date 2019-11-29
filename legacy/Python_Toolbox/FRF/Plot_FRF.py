# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================



# =============================================================================
    
"""
from numpy import ndarray
import matplotlib.pyplot as plt
import math as mt
import cmath as cmt
import numpy as np

def plot_FRF(Data, dofs_i, dofs_j, Freq = None, PlotType = None):
    
    if Freq is None:      
        Freq = np.arange(0,Data.nFreq)
        
    if PlotType is None:
        if isinstance(Data, ndarray):
            db = 20*np.log10(abs(Data[dofs_i, dofs_j, Freq[0]:len(Freq)])) 
        else:
            db = 20*np.log10(abs(Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)])).transpose()
                     
        plt.plot(db)       
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Admittance (dB)")
        plt.grid()
        
    elif PlotType == "log": 
        plt.figure()
        plt.subplot(2,1,1)
        plt.semilogy(abs(Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)]))
        plt.xlabel("Frequency (Hz)")
        if Data.Channels[0].Unit is not None: 
            if type(Data.Channels[0].Unit)  == str:
                plt.ylabel("Admittance in " + Data.Channels[0].Unit)
            else:
                plt.ylabel("Admittance in " + Data.Channels[0].Unit.Name)  
        else:
            plt.ylabel("Admittance")
            
        plt.grid(which = "both")
        
    elif PlotType == "phase":
        plt.subplot(2,1,2)
        Data = Data.Data[dofs_i, dofs_j, Freq[0]:len(Freq)]
        Data = [mt.degrees(cmt.phase(d)) for d in Data]
        plt.plot(Data)
        plt.ylim(-180, +180)
        plt.yticks([-180, -90, 0, 90, 180])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase Angle in Degrees")
        plt.grid()