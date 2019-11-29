# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 10:43:44 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts:


# =============================================================================

"""
import numpy as np
from scipy.linalg import block_diag
import pandas as pd

def IDM_f(Ref_Channels, oV_Ref_Channel_list, virtual_points, Ref_Impact_Group = None):
    
    vp_coordinates = oV_Ref_Channel_list[0].Position
    
    R_Matrix = np.zeros((6*virtual_points, 3*len(Ref_Channels)),float)
    index = list()
    iterator = 0
    
    Channels_Not_Transformed = None 
        
    Distance_Matrix = np.zeros((len(Ref_Channels),3),float)
      
    Groups = [d.Grouping for d in Ref_Channels]
    
    orientation_vector_list = [d.Direction for d in Ref_Channels]
             
    orientation_matrix = block_diag(*orientation_vector_list).transpose()
    
    position_1 = [d.Position[0] for d in Ref_Channels]
    position_2 = [d.Position[1] for d in Ref_Channels]
    position_3 = [d.Position[2] for d in Ref_Channels]
            
    for i in range (0,len(Ref_Channels)):
        Distance_Matrix[i,0] = position_1[i] - vp_coordinates[0]
        Distance_Matrix[i,1] = position_2[i] - vp_coordinates[1]
        Distance_Matrix[i,2] = position_3[i] - vp_coordinates[2]
        
        
    for dof in range (0,3):
        if dof + 1 == 1:    
            for i in range (0,len(Ref_Channels)):
                for k in range (0,virtual_points):
                    R_Matrix[0 + 6*k][3*i] = 1
                    R_Matrix[1 + 6*k][3*i] = 0
                    R_Matrix[2 + 6*k][3*i] = 0
                    R_Matrix[3 + 6*k][3*i] = 0
                    R_Matrix[4 + 6*k][3*i] = Distance_Matrix[i,2]
                    R_Matrix[5 + 6*k][3*i] = -1*Distance_Matrix[i,1]
        
        if dof + 1 == 2:
            for i in range (0,len(Ref_Channels)):
                for k in range (0,virtual_points):
                    R_Matrix[0 + 6*k][1 + 3*i] = 0
                    R_Matrix[1 + 6*k][1 + 3*i] = 1
                    R_Matrix[2 + 6*k][1 + 3*i] = 0
                    R_Matrix[3 + 6*k][1 + 3*i] = -1*Distance_Matrix[i,2]
                    R_Matrix[4 + 6*k][1 + 3*i] = 0
                    R_Matrix[5 + 6*k][1 + 3*i] = Distance_Matrix[i,0]
                    
        if dof + 1 == 3:
            for i in range (0,len(Ref_Channels)):
                for k in range (0,virtual_points):
                    R_Matrix[0 + 6*k][2 + 3*i] = 0
                    R_Matrix[1 + 6*k][2 + 3*i] = 0
                    R_Matrix[2 + 6*k][2 + 3*i] = 1
                    R_Matrix[3 + 6*k][2 + 3*i] = Distance_Matrix[i,1]
                    R_Matrix[4 + 6*k][2 + 3*i] = -1*Distance_Matrix[i,0]
                    R_Matrix[5 + 6*k][2 + 3*i] = 0
                    
    
    R_f = np.dot(R_Matrix, orientation_matrix)
    
    if Ref_Impact_Group is not None:
        
        Ref_Grouping = pd.unique([item.Grouping for item in Ref_Impact_Group])
   
        for i, j in enumerate(Groups):
            for item in Ref_Grouping:
                if j == item:
                    index.append(i)
                    
        for item in index:
            R_f[:,item] = 0
            
        Add = np.zeros((len(index), len(Ref_Channels)), float)
    
        for item in index:
            Add[iterator, item] = 1
            iterator = iterator + 1
        
        R_f = np.vstack([R_f, Add])
        
    R_f = R_f.transpose()
        
    return R_f       
    
    
    
    