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
from collections import Counter
from pandas import unique
     

def IDM_f(Ref_Channels, oV_Ref_Channel_list, virtual_points, Ref_Impact_Group = None):
    
    index = list()
    position_1 = list()
    position_2 = list()
    position_3 = list()
    iterator = 0
    vp = 0
    
    Groups = [item.Grouping for item in Ref_Channels]
    
    if Ref_Impact_Group is not None:      
        Ref_Channels_Without_Ref_Groups = [item for item in Ref_Channels if item not in Ref_Impact_Group]
    else:
        Ref_Channels_Without_Ref_Groups = Ref_Channels
        
    Groups_Without_Ref_Groups = [item.Grouping for item in Ref_Channels_Without_Ref_Groups]
    Number_of_Groups = Counter(Groups_Without_Ref_Groups)
    
    R_Matrix = np.zeros((6*virtual_points, 3*len(Groups_Without_Ref_Groups)),float) #Hard coded
    Distance_Matrix = np.zeros((len(Groups_Without_Ref_Groups),3),float)
    
#    vp_coordinates_dict = {item.Grouping:item.Position for item in oV_Ref_Channel_list if oV_Ref_Channel_list.index(item) == 0 or \
#                      item.Position != oV_Ref_Channel_list[oV_Ref_Channel_list.index(item)-1].Position}
    
    vp_coordinates_dict = {item.Grouping:item.Position for item in oV_Ref_Channel_list if oV_Ref_Channel_list.index(item) == 0 or \
                      item.Grouping != oV_Ref_Channel_list[oV_Ref_Channel_list.index(item)-1].Grouping}   
    
    for d in Ref_Channels_Without_Ref_Groups:
        position_1.append({d.Grouping:d.Position[0]})
        position_2.append({d.Grouping:d.Position[1]})
        position_3.append({d.Grouping:d.Position[2]})
        
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
        
    orientation_vector_list = [d.Direction for d in Ref_Channels_Without_Ref_Groups]         
    orientation_matrix = block_diag(*orientation_vector_list).transpose()
        
    for dof in range (0,3): #Hard coded
        iterator = 0
        vp = 0
        if dof + 1 == 1:    
            for keys in Number_of_Groups:
                for i in range (0,Number_of_Groups[keys]):  
                    R_Matrix[0 + 6*vp][3*i + 3*iterator] = 1 
                    R_Matrix[1 + 6*vp][3*i + 3*iterator] = 0
                    R_Matrix[2 + 6*vp][3*i + 3*iterator] = 0
                    R_Matrix[3 + 6*vp][3*i + 3*iterator] = 0
                    R_Matrix[4 + 6*vp][3*i + 3*iterator] = Distance_Matrix[i + iterator,2]
                    R_Matrix[5 + 6*vp][3*i + 3*iterator] = -1*Distance_Matrix[i + iterator,1]
                iterator = iterator + Number_of_Groups[keys]
                vp = vp + 1
        
        if dof + 1 == 2:
            for keys in Number_of_Groups:     
                for i in range (0,Number_of_Groups[keys]):
                    R_Matrix[0 + 6*vp][1 + 3*i + 3*iterator] = 0
                    R_Matrix[1 + 6*vp][1 + 3*i + 3*iterator] = 1
                    R_Matrix[2 + 6*vp][1 + 3*i + 3*iterator] = 0
                    R_Matrix[3 + 6*vp][1 + 3*i + 3*iterator] = -1*Distance_Matrix[i + iterator,2]
                    R_Matrix[4 + 6*vp][1 + 3*i + 3*iterator] = 0
                    R_Matrix[5 + 6*vp][1 + 3*i + 3*iterator] = Distance_Matrix[i + iterator,0]
                iterator = iterator + Number_of_Groups[keys]
                vp = vp + 1
        
        if dof + 1 == 3:
            for keys in Number_of_Groups:     
                for i in range (0,Number_of_Groups[keys]):
                    R_Matrix[0 + 6*vp][2 + 3*i + 3*iterator] = 0
                    R_Matrix[1 + 6*vp][2 + 3*i + 3*iterator] = 0
                    R_Matrix[2 + 6*vp][2 + 3*i + 3*iterator] = 1
                    R_Matrix[3 + 6*vp][2 + 3*i + 3*iterator] = Distance_Matrix[i + iterator,1]
                    R_Matrix[4 + 6*vp][2 + 3*i + 3*iterator] = -1*Distance_Matrix[i + iterator,0]
                    R_Matrix[5 + 6*vp][2 + 3*i + 3*iterator] = 0
                iterator = iterator + Number_of_Groups[keys]
                vp = vp + 1
                

    R_f = np.dot(R_Matrix, orientation_matrix)
    R_conc = R_f.transpose()
    iterator = len(R_f)
    
    if Ref_Impact_Group is not None:
        
        Ref_Grouping = unique([item.Grouping for item in Ref_Impact_Group])
   
        for i, j in enumerate(Groups):
            for item in Ref_Grouping:
                if j == item:
                    index.append(i)
                    
        R_f = np.pad(R_f, ((0, len(index)),(len(index),0)), 'constant', constant_values=(0)) #Padding width = ((Top, Bottom),(Left, Right))
    
        for item in index:
            R_f[iterator, item] = 1
            iterator = iterator + 1
        
    R_f = R_f.transpose()
        
    return R_f, R_conc
    
    
    
    