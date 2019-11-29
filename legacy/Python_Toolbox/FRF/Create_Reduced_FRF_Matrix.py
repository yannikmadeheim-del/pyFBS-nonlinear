# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 


# =============================================================================

"""
import numpy as np

from Python_Toolbox.FRF.FRF import FRF

def create_reduced_FRF_matrix(Y, IDMf, IDMu = None):
    
    if IDMu is not None:
        
        oFRF = FRF()
        
        Yqm = np.zeros((len(IDMu.T),len(IDMf.T),Y.nFreq),complex)
        
        for i in range (0,Y.nFreq):
            
            Yqm[:,:,i] = IDMu.T.dot(Y.Data[:,:,i]).dot(IDMf.T.transpose())
            
        oFRF.Data = Yqm
        oFRF.Freqs = Y.Freqs
        oFRF.nFreq = Y.nFreq
        oFRF.Channels = IDMu.Virtual_Channels
        oFRF.nChannels = IDMu.nVirtual_Channels
        oFRF.RefChannels = IDMf.Virtual_Channels
        oFRF.nRefChannels = IDMf.nVirtual_Channels
        
    else:
        
        oFRF = FRF()
        
        Yum = np.zeros((len(Y.Data[:,0,0]),len(IDMf.T),Y.nFreq),complex)
        
        for i in range (0,Y.nFreq):
            
            Yum[:,:,i] = Y.Data[:,:,i].dot(IDMf.T.transpose())
            
        oFRF.Data = Yum
        oFRF.Freqs = Y.Freqs
        oFRF.nFreq = Y.nFreq
        oFRF.Channels = Y.Channels
        oFRF.nChannels = Y.nChannels
        oFRF.RefChannels = IDMf.Virtual_Channels
        oFRF.nRefChannels = IDMf.nVirtual_Channels
    
    return oFRF
          