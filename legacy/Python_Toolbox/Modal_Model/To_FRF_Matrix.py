# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""

from numpy import zeros, outer

from Python_Toolbox.FRF.FRF import FRF
from Python_Toolbox.Modal_Model.Nearest_Neighbor import nearest_neighbor

def to_FRF_matrix(oModal_Model, channels, refChannels, FreqRange, FRFType):
    
    nearest_nodes_channels = nearest_neighbor(oModal_Model.Nodes, channels)
    nearest_dofs_channels = oModal_Model.ndofs*nearest_nodes_channels
    
    nearest_nodes_refchannels = nearest_neighbor(oModal_Model.Nodes, refChannels)
    nearest_dofs_refchannels = oModal_Model.ndofs*nearest_nodes_refchannels
    
    E_Channel = zeros((len(channels), len(oModal_Model.X)),float)
    E_RefChannel = zeros((len(refChannels), len(oModal_Model.X)),float)
    
    for i in range (0,len(channels)):
        E_Channel[i,nearest_dofs_channels[0,i]:nearest_dofs_channels[0,i]+3] = channels[i].Direction
        
    for i in range (0,len(refChannels)):
        E_RefChannel[i,nearest_dofs_refchannels[0,i]:nearest_dofs_refchannels[0,i]+3] = refChannels[i].Direction
    
    
    Xi = E_Channel.dot(oModal_Model.X)
    Xj = E_RefChannel.dot(oModal_Model.X)
    
    oFRF = FRF()
    oFRF.Freqs = FreqRange
    oFRF.nFreq = len(FreqRange)
    oFRF.Channels = channels
    oFRF.nChannels = len(channels)
    oFRF.RefChannels = refChannels
    oFRF.nRefChannels = len(refChannels)
    
    Data = zeros((oFRF.nChannels, oFRF.nRefChannels, len(FreqRange)), complex)
    
    for i in range (0,len(FreqRange)):   
        for j in range (0,oModal_Model.nModes):
     
            Numerator = outer(Xi[:,j], Xj[:,j])
            
            if j < 6:
                
                Denominator = -1*2*3.14*FreqRange[i] * 2*3.14*FreqRange[i]
                
            else:
            
                Denominator = complex(-2*3.14*FreqRange[i] * 2*3.14*FreqRange[i] + oModal_Model.omega[j] * oModal_Model.omega[j],  \
                                  2 * 2*3.14*FreqRange[i] * oModal_Model.omega[j] * oModal_Model.zeta[j])
            
            Data[:,:,i] = -1*2*3.14*FreqRange[i] * 2*3.14*FreqRange[i]*Numerator/Denominator + Data[:,:,i] #For converting displacements to accelerations omega is multiplied.
                 
    oFRF.Data = Data 
    
    return oFRF