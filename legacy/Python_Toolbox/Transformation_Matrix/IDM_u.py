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
from pandas import unique
from collections import Counter

def IDM_u(Channels, oV_Channel_list, virtual_points, Ref_Sensor_Group = None):
    
    index = list()
    position_1 = list()
    position_2 = list()
    position_3 = list()
    iterator = 0
    vp = 0
    
    Groups = [item.Grouping for item in Channels]
    
    if Ref_Sensor_Group is not None:     
        Channels_Without_Ref_Groups = [item for item in Channels if item not in Ref_Sensor_Group]
    else:
        Channels_Without_Ref_Groups = Channels
        
    Groups_Without_Ref_Groups = [item.Grouping for item in Channels_Without_Ref_Groups]
    Number_of_Groups = Counter(Groups_Without_Ref_Groups)
    
    R_Matrix = np.zeros((len(Groups_Without_Ref_Groups),6*virtual_points),float)
    
    #DOFs per node. 
    ndofs = len([d.Node_Number for d in Channels if d.Node_Number == Channels[0].Node_Number])
    
    Distance_Matrix = np.zeros((int(len(Groups_Without_Ref_Groups)/ndofs),3),float)
    
    vp_coordinates_dict = {item.Grouping:item.Position for item in oV_Channel_list if oV_Channel_list.index(item) == 0 or \
                      item.Position != oV_Channel_list[oV_Channel_list.index(item)-1].Position}  
    
    #2D array created. The first column contains the grouping of channel and the second column contains the x,y and z coordinates. 
    for i in range (0,len(Groups_Without_Ref_Groups), ndofs):
        position_1.append({Channels[i].Grouping:Channels[i].Position[0]})
        position_2.append({Channels[i].Grouping:Channels[i].Position[1]})
        position_3.append({Channels[i].Grouping:Channels[i].Position[2]})
    
    #Variable key contains grouping number.
    #Distance matrix has rows equal to the number of sensors and columns equal to 3. Each column contains the distance (x,y,z) b/w 
    #the virtual point and sensor. In order to ensure that the sensor and the virtual point have the same group, the variable key
    #is used. 
    
    for item in position_1:
        for key in item.keys():         
            Distance_Matrix[iterator, 0] =  item[key] - vp_coordinates_dict[key][0]
        iterator = iterator + 1
            
    iterator = 0
    for item in position_2:
        for key in item.keys():         
            Distance_Matrix[iterator, 1] =  item[key] - vp_coordinates_dict[key][1]
        iterator = iterator + 1
        
    iterator = 0
    for item in position_3:
        for key in item.keys():         
            Distance_Matrix[iterator, 2] =  item[key] - vp_coordinates_dict[key][2]
        iterator = iterator + 1
        
    iterator = 0
    
    orientation_matrix = np.zeros((3, 3),float)
    orientation_matrix_list = list()
    
    for i in range (0,len(Channels_Without_Ref_Groups) - ndofs + 1,ndofs):        
        for j in range (0,ndofs):  
            orientation_matrix[j,:] = Channels_Without_Ref_Groups[i+j].Direction
            
        orientation_matrix_list.append(orientation_matrix)
        orientation_matrix = np.zeros((3, 3),float)    
        
    orientation_matrix = block_diag(*orientation_matrix_list)
                                    
    for dof in range (0,ndofs):
        iterator = 0
        vp = 0
        if dof + 1 == 1:    
            for keys in Number_of_Groups:
                for i in range (0,int(Number_of_Groups[keys]/ndofs)):
                    R_Matrix[3*i + 3*iterator][0 + 6*vp] = 1
                    R_Matrix[3*i + 3*iterator][1 + 6*vp] = 0
                    R_Matrix[3*i + 3*iterator][2 + 6*vp] = 0
                    R_Matrix[3*i + 3*iterator][3 + 6*vp] = 0
                    R_Matrix[3*i + 3*iterator][4 + 6*vp] = Distance_Matrix[i+iterator,2]
                    R_Matrix[3*i + 3*iterator][5 + 6*vp] = -1*Distance_Matrix[i+iterator,1]
                iterator = iterator + int(Number_of_Groups[keys]/ndofs)
                vp = vp + 1
        
        if dof + 1 == 2:
            for keys in Number_of_Groups:     
                for i in range (0,int(Number_of_Groups[keys]/ndofs)):
                    R_Matrix[1 + 3*i + 3*iterator][0 + 6*vp] = 0
                    R_Matrix[1 + 3*i + 3*iterator][1 + 6*vp] = 1
                    R_Matrix[1 + 3*i + 3*iterator][2 + 6*vp] = 0
                    R_Matrix[1 + 3*i + 3*iterator][3 + 6*vp] = -1*Distance_Matrix[i+iterator,2]
                    R_Matrix[1 + 3*i + 3*iterator][4 + 6*vp] = 0
                    R_Matrix[1 + 3*i + 3*iterator][5 + 6*vp] = Distance_Matrix[i+iterator,0]
                iterator = iterator + int(Number_of_Groups[keys]/ndofs)
                vp = vp + 1
        
        if dof + 1 == 3:
            for keys in Number_of_Groups:     
                for i in range (0,int(Number_of_Groups[keys]/ndofs)):
                    R_Matrix[2 + 3*i + 3*iterator][0 + 6*vp] = 0
                    R_Matrix[2 + 3*i + 3*iterator][1 + 6*vp] = 0
                    R_Matrix[2 + 3*i + 3*iterator][2 + 6*vp] = 1
                    R_Matrix[2 + 3*i + 3*iterator][3 + 6*vp] = Distance_Matrix[i+iterator,1]
                    R_Matrix[2 + 3*i + 3*iterator][4 + 6*vp] = -1*Distance_Matrix[i+iterator,0]
                    R_Matrix[2 + 3*i + 3*iterator][5 + 6*vp] = 0
                iterator = iterator + int(Number_of_Groups[keys]/ndofs)
                vp = vp + 1
                    
    R_u = np.dot(orientation_matrix,R_Matrix)
    R_conc = R_u
    iterator = len(R_u[0,:])
    
    if Ref_Sensor_Group is not None:
        
        Ref_Grouping = unique([item.Grouping for item in Ref_Sensor_Group])

        for i, j in enumerate(Groups):
            for item in Ref_Grouping:
                if j == item:
                    index.append(i)
                    
        R_u = np.pad(R_u, ((0, len(index)),(0, len(index))), 'constant', constant_values=(0)) #Padding width = ((Top, Bottom),(Left, Right))
            
        for item in index:
            R_u[item, iterator] = 1
            iterator = iterator + 1
        
    return R_u, R_conc