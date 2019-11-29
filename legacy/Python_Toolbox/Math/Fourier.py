# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""
from numpy import reshape, tile, fft
from math import ceil, floor

from Python_Toolbox.Math.Window import window

def fourier(Time_Block_Data, Time, Window, zp):
    
    Number_of_Time_Steps = len(Time_Block_Data[0,0,:])
    dt = Time[1] - Time[0]
    fs = 1/dt
    
    fnyq = ceil(fs/2)
    df_org = fs/Number_of_Time_Steps #Original frequency axis
    
    if any([item > fnyq for item in zp]):   
        print('Warning! Frequency axis clipped to Nyquist frequency of ', fnyq, ' Hz.')
        
    freq = [item for item in zp if item <= fnyq] #Targetted frequency axis
    df_eff = freq[1] - freq[0]
    
    zp = df_org / df_eff #Actual zero padding
    n_eff = floor(zp * Number_of_Time_Steps)
        
    Number_of_Frequencies = len(freq)
    win, winSpecs = window(Number_of_Time_Steps, Window)
    
    win = reshape(win, (1,1,len(win)))
    win = tile(win, (len(Time_Block_Data[:,0,0]), 1, 1))
    
    winSpecs.Broadband_Correction_Factor = winSpecs.Broadband_Correction_Factor / zp #Correct ENBW for zero padding
    winSpecs.Zero_Padding_Factor = zp
    
    Time_Block_Data = Time_Block_Data * win #Performing windowing
    
    FData = fft.fft(Time_Block_Data, n_eff, 2) #Performing FFT #Taking single sided spectrum
    FData = (2 / Number_of_Time_Steps) * FData[:,:,0:Number_of_Frequencies]
    FData[:,:,0] = 0.5 * FData[:,:,0] #Correct DC offset
    FData = FData / winSpecs.Coherent_Gain #Correct for coherent gain
    
    return FData, freq, winSpecs
    
     