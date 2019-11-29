# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================
This is an improved file which tries to perform coupling on the basis of channels and reference channels.
# =============================================================================
"""

from scipy.sparse import block_diag
import numpy as np

from Python_Toolbox.FRF.FRF import FRF
from Python_Toolbox.FRF.Create_Boolean_Matrix import create_boolean_matrix

def couple(subsA, subsB):
    
    oFRF = FRF()
    Block_Matrix_List = list()
    Channels = list()
    RefChannels = list()
    
    Channels = subsA.Channels + subsB.Channels
    RefChannels = subsA.RefChannels + subsB.RefChannels
              
    
    oFRF.Freqs = subsA.Freqs
    oFRF.nFreq = subsA.nFreq
    
    oFRF.Channels = Channels
    oFRF.RefChannels = RefChannels
    
    oFRF.nChannels = len(Channels)
    oFRF.nRefChannels = len(RefChannels)
       
    Y_uncoupled = np.zeros((oFRF.nChannels, oFRF.nRefChannels, oFRF.nFreq),complex)
    Y_coupled = np.zeros((oFRF.nChannels, oFRF.nRefChannels, oFRF.nFreq),complex)
       
    for i in range (0,oFRF.nFreq):
        for item in [subsA, subsB]:
            Block_Matrix_List.append(item.Data[:,:,i])
        Y_uncoupled[:,:,i] = block_diag(Block_Matrix_List).todense()
        Block_Matrix_List = list()
        
    
    #----------------------Boolean Matrix-------------------------------------
    
    B_c, B_e = create_boolean_matrix(oFRF, subsA, subsB)
        
    #-------------------------Coupling-----------------------------------------
    
    for i in range (0,oFRF.nFreq):
        
        # first term ist Y*transp(B)
        first_term = Y_uncoupled[:,:,i].dot(B_e.transpose())
        # second term is B*Y*transp(B)
        interface_flexibility_without_inverse = B_c.dot(Y_uncoupled[:,:,i]).dot(B_e.transpose())
        
        interface_flexibility = np.linalg.inv(interface_flexibility_without_inverse)
        
        del interface_flexibility_without_inverse
        
        Y_assembled = first_term.dot(interface_flexibility).dot(B_c).dot(Y_uncoupled[:,:,i])
        
        del first_term
        del interface_flexibility
        
        Y_coupled[:,:,i] = (Y_uncoupled[:,:,i] - Y_assembled)
        
    oFRF.Data = Y_coupled
 
    return oFRF
    
        