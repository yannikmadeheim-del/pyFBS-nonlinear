# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================
"""

from numpy.linalg import norm
from numpy import conj, arange, mean, sum
import matplotlib.pyplot as plt

from Python_Toolbox.Transformation_Matrix.Rigidify import rigidify


def consistency(Y_Data, IDM_Data, dofs_i, dofs_j, Freq = None, Dim = None, Type = None):
    
    if Freq.all() == None:
        Freq = Y_Data.nFreq
        
    Freq_Indices = arange(list(Freq).index(Freq[0]),list(Freq).index(Freq[-1])+1)
#    IDM_Data.P = rigidify(IDM_Data, dofs_i, dofs_j) #Incorrect function. Should be improved.
    
    if Type == "Overall":
        
        u = Y_Data.Data[dofs_i, :, Freq_Indices[0]:Freq_Indices[-1]+1]
        u = u[:, dofs_j, :]
        
        if Dim == "Sensor": 
            u = sum(u, axis = 1)
                
        if Dim == 'Hammer':       
            u = sum(u, axis = 0)
            
        u_filt = IDM_Data.P.dot(u)
        
        Filtered_Response = norm(u_filt,axis = 0)
        Actual_Response = norm(u,axis = 0)
        
        Overall_Consistency = Filtered_Response/Actual_Response*100
                
        plt.figure()
        plt.grid(which='both')
        plt.plot(Overall_Consistency)  
        plt.xlabel("Frequency (Hz)")
        ylabel = Type + ' ' + Dim + ' Consistency (%)'
        plt.ylabel(ylabel) 
        plt.ylim(0, 110)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        plt.fill_between(Freq,Overall_Consistency)
                
    elif Type == 'Specific':
        
        u = Y_Data.Data[dofs_i, :, Freq_Indices[0]:Freq_Indices[-1]+1]
        u = u[:, dofs_j, :]
        
        if Dim == 'Sensor':
            u = sum(u, axis = 1)
            dofs = dofs_i
            
        if Dim == 'Hammer':      
            u = sum(u, axis = 0)
            dofs = dofs_j
            
        u_filt = IDM_Data.P.dot(u)
        
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
        
        Specific_Consistency = Numerator/Denominator
            
        Specific_Consistency = mean(Specific_Consistency,axis=1)
        
        plt.figure()
        plt.grid(which='both')
        plt.bar(dofs, abs(Specific_Consistency)*100)

        plt.xlabel(Dim + " DoF")
        ylabel = Type + ' ' + Dim + ' Consistency (%)'
        plt.ylabel(ylabel) 
        plt.ylim(0, 100)
        plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
         
    
   
                              
     