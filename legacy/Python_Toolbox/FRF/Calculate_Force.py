# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================


# =============================================================================

"""
from Python_Toolbox.Freq_Blocks.Freq_Blocks import Freq_Blocks
from numpy import zeros, reshape
from numpy.linalg import pinv

def calculate_force(Y, u): 
        
    if u.Data.ndim == 2:
        u.Data = reshape(u.Data, (len(u.Data),1,len(u.Data[0,:])))
    
    Data = zeros((Y.nRefChannels, len(u.Data[0,:,0]), Y.nFreq),complex)
    
    for i in range (0,Y.nFreq):
        Data[:,:,i] = pinv(Y.Data[:,:,i]).dot(u.Data[:,:,i])
        
    oFreq_Blocks = Freq_Blocks()
    oFreq_Blocks.Data = Data
    oFreq_Blocks.Channels = Y.Channels
    oFreq_Blocks.Freq = Y.Freqs
    oFreq_Blocks.nFreq = Y.nFreq
    oFreq_Blocks.Blocks = u.Blocks
    oFreq_Blocks.nBlocks = u.nBlocks
    
    return oFreq_Blocks