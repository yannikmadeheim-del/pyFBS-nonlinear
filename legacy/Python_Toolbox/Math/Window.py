# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

1. Only hanning window is implemented yet.
    
# =============================================================================

"""
from numpy import arange, asarray
from math import cos, pi

class winSpecs(object):
    
    Window_Type = None
    Window_Length = None
    Coherent_Gain = None
    Equivalent_Noise_Bandwidth = None
    Broadband_Correction_Factor = None
    Zero_Padding_Factor = None

def window(Number_of_Time_Steps, Window):
    
    win = generate_window(Window, Number_of_Time_Steps)
    winSpecs = generate_window_specs(win, Window)
    
    return win, winSpecs
    
    
def generate_window(Window, Number_of_Time_Steps):
    
    x = arange(0, Number_of_Time_Steps)
    t = x/Number_of_Time_Steps
    
    if Window == 'hann':
        win = [0.5 - 0.5 * (cos(2*pi*item)) for item in t]
              
    return win

def generate_window_specs(win, winType):
    
    win = asarray(win)
    
    N = len(win)
    CG = sum(win)/N #Gain
    ENBW = (N * win.dot(win)) / (sum(win) * sum(win)) #Equivalent noise bandwidth
    
    owinSpecs = winSpecs()
    owinSpecs.Window_Type = winType
    owinSpecs.Window_Length = N
    owinSpecs.Coherent_Gain = CG
    owinSpecs.Equivalent_Noise_Bandwidth = ENBW
    owinSpecs.Broadband_Correction_Factor = 1 / ENBW
    
    return owinSpecs
        