# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""

from Python_Toolbox.Math.Fourier import fourier
from Python_Toolbox.Freq_Blocks.Freq_Blocks import Freq_Blocks

def from_time_blocks(oTime_Blocks, Window, Zero_Padding = None):
    
#%% Fourier transformation function similar to MATLAB toolbox
    
    if Zero_Padding is None:
        Zero_Padding = 1
    
    FData, freq, winSpecs = fourier(oTime_Blocks.Data, oTime_Blocks.Time, Window, Zero_Padding)

#%% Frequency Blocks objects
    
    oFrequency_Blocks = Freq_Blocks()
    oFrequency_Blocks.__dict__ = oTime_Blocks.__dict__.copy()
    oFrequency_Blocks.Data = FData
    oFrequency_Blocks.Freq = freq
    oFrequency_Blocks.Parameters = winSpecs

    return oFrequency_Blocks    