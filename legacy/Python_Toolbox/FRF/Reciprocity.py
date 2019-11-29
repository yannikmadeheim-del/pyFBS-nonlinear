# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================
"""

from numpy import conj, mean


def reciprocity(Y_Data, Freq = None):
    
    if Freq == None:
        Freq = Y_Data.nFreq
    
    u = Y_Data.Data
    u_filt = Y_Data.Data.transpose(1, 0, 2)
    
    conj_u = conj(u)
    conj_u_filt = conj(u_filt)
    
    if u[:,0].all() == 0: 
        u = u[:, 1:]
        conj_u = conj_u[:, 1:]
    if u_filt[:,0].all() == 0:
        u_filt = u_filt[:, 1:]
        conj_u_filt = conj_u_filt[:, 1:]
    
    Numerator = (u_filt + u)*(conj_u_filt + conj_u)
    Denominator = 2*(u_filt*conj_u_filt + u*conj_u)
    
    Specific_Consistency = abs(Numerator/Denominator)
        
    Specific_Consistency = mean(Specific_Consistency,axis=2)
    
    
    return Specific_Consistency