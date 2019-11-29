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

def IDM_u(Channels, oV_Channel_list, virtual_points, Ref_Sensor_Group = None):
    
    ndofs = len([d.Node_Number for d in Channels if d.Node_Number == Channels[0].Node_Number])
    vp_coordinates = oV_Channel_list[0].Position
    
    orientation_matrix = np.zeros((3, 3),float)
    orientation_matrix_list = list()
    R_Matrix = np.zeros((len(Channels),6*virtual_points),float)
        
    Distance_Matrix = np.zeros((int(len(Channels)/ndofs),3),float)
    index = list()
    iterator = 0
    position_1 = list()
    position_2 = list()
    position_3 = list()
      
    for i in range (0,len(Channels),ndofs):
        position_1.append(Channels[i].Position[0])
        position_2.append(Channels[i].Position[1])
        position_3.append(Channels[i].Position[2])
    
    Groups = [d.Grouping for d in Channels]
    
    for i in range (0,len(Channels),ndofs):        
        for j in range (0,ndofs):  
            orientation_matrix[j,:] = Channels[i+j].Direction
            
        orientation_matrix_list.append(orientation_matrix)
        orientation_matrix = np.zeros((3, 3),float)    
        
    orientation_matrix = block_diag(*orientation_matrix_list)
            
    for i in range (0,int(len(Channels)/ndofs)):
        Distance_Matrix[i,0] = position_1[i] - vp_coordinates[0]
        Distance_Matrix[i,1] = position_2[i] - vp_coordinates[1]
        Distance_Matrix[i,2] = position_3[i] - vp_coordinates[2]
        
        
    for dof in range (0,ndofs):
        if dof + 1 == 1:    
            for i in range (0,int(len(Channels)/ndofs)):
                for k in range (0,virtual_points):
                    R_Matrix[3*i][0 + 6*k] = 1
                    R_Matrix[3*i][1 + 6*k] = 0
                    R_Matrix[3*i][2 + 6*k] = 0
                    R_Matrix[3*i][3 + 6*k] = 0
                    R_Matrix[3*i][4 + 6*k] = Distance_Matrix[i,2]
                    R_Matrix[3*i][5 + 6*k] = -1*Distance_Matrix[i,1]
        
        if dof + 1 == 2:
            for i in range (0,int(len(Channels)/ndofs)):
                for k in range (0,virtual_points):
                    R_Matrix[1 + 3*i][0 + 6*k] = 0
                    R_Matrix[1 + 3*i][1 + 6*k] = 1
                    R_Matrix[1 + 3*i][2 + 6*k] = 0
                    R_Matrix[1 + 3*i][3 + 6*k] = -1*Distance_Matrix[i,2]
                    R_Matrix[1 + 3*i][4 + 6*k] = 0
                    R_Matrix[1 + 3*i][5 + 6*k] = Distance_Matrix[i,0]
                    
        if dof + 1 == 3:
            for i in range (0,int(len(Channels)/ndofs)):
                for k in range (0,virtual_points):
                    R_Matrix[2 + 3*i][0 + 6*k] = 0
                    R_Matrix[2 + 3*i][1 + 6*k] = 0
                    R_Matrix[2 + 3*i][2 + 6*k] = 1
                    R_Matrix[2 + 3*i][3 + 6*k] = Distance_Matrix[i,1]
                    R_Matrix[2 + 3*i][4 + 6*k] = -1*Distance_Matrix[i,0]
                    R_Matrix[2 + 3*i][5 + 6*k] = 0
                    
    R_u = np.dot(orientation_matrix,R_Matrix) 
    
    if Ref_Sensor_Group is not None:
        
        Ref_Grouping = pd.unique([item.Grouping for item in Ref_Sensor_Group])

        for i, j in enumerate(Groups):
            for item in Ref_Grouping:
                if j == item:
                    index.append(i)
                    
        for item in index:
            R_u[item,:] = 0
            
        Add = np.zeros((len(Channels), len(index)), float)
    
        for item in index:
            Add[item, iterator] = 1
            iterator = iterator + 1
        
        R_u = np.hstack([R_u, Add])
        
    return R_u