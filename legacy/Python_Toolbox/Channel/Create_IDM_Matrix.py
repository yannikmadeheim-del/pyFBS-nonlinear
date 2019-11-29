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

from Python_Toolbox.Unit.Unit import Unit
from Python_Toolbox.Quantity.Quantity import Quantity
from Python_Toolbox.Channel.Channel import Channel
from Python_Toolbox.IDM.IDM import IDM
from Python_Toolbox.Transformation_Matrix.Transformation_Matrix import Transformation_Matrix

from Python_Toolbox.Transformation_Matrix.IDM_u import IDM_u 
from Python_Toolbox.Transformation_Matrix.IDM_f import IDM_f

from Python_Toolbox.Transformation_Matrix.Create_Transformation_U_Matrix import create_transformation_u_matrix
from Python_Toolbox.Transformation_Matrix.Create_Transformation_F_Matrix import create_transformation_f_matrix

def create_IDM_matrix(Data, vp_prop, Ref_Group = None):
    
    Directions = {1:"+X", 2:"+Y", 3:"+Z", -1:"-X", -2:"-Y", -3:"-Z"}
    
    Channels = Data
        
    oTransformation_Matrix = Transformation_Matrix()
    oIDM = IDM()
    
    oV_Channel_list = list()
    
    for i in range (0,len(vp_prop)):
        
        oV_Unit = Unit()
        oV_Quantity = Quantity()
        oV_Channel = Channel()
        
        oV_Unit.Name = vp_prop["Unit"][i]
        oV_Quantity.Name = vp_prop["Quantity"][i]
        
        oV_Channel.Direction = [vp_prop["Direction_1"][i], vp_prop["Direction_2"][i], vp_prop["Direction_3"][i]]
        oV_Channel.Direction_Number = np.nonzero(oV_Channel.Direction)[0][0] + 1
        oV_Channel.Direction_Label = Directions[oV_Channel.Direction_Number]
        
        oV_Channel.Position = [vp_prop["Position_1"][i], vp_prop["Position_2"][i], vp_prop["Position_3"][i]]
        oV_Channel.Name = vp_prop["Name"][i]
        oV_Channel.Node_Number = vp_prop["NodeNumber"][i]
        oV_Channel.Grouping = vp_prop["Grouping"][i]
                
        oV_Quantity.Unit = oV_Unit
        oV_Channel.Quantity = oV_Quantity
        oV_Channel.Unit = oV_Unit
        
        oV_Channel_list.append(oV_Channel)
        
    virtual_points = len(set([d.Node_Number for d in oV_Channel_list]))
    
    if Channels[0].Unit.Name == 'N':
        R, R_conc = IDM_f(Channels, oV_Channel_list, virtual_points, Ref_Group)            
        W = np.identity(np.size(R, 0))
        W_conc = np.identity(np.size(R_conc, 0))
        T = create_transformation_f_matrix(R,W)
        T_conc = create_transformation_f_matrix(R_conc,W_conc)
    else:
        R, R_conc = IDM_u(Channels, oV_Channel_list, virtual_points, Ref_Group)
        W = np.identity(np.size(R, 0))
        W_conc = np.identity(np.size(R_conc, 0))
        T = create_transformation_u_matrix(R,W)
        T_conc = create_transformation_f_matrix(R_conc,W_conc)
        
    if Ref_Group is not None:
        oV_Channel_list = oV_Channel_list + Ref_Group
    
    oTransformation_Matrix.R = R
         
    oIDM.W = W       
    oIDM.R = R
    oIDM.T = T
    oIDM.P = R_conc.dot(T_conc)
    
    oIDM.nChannels = len(Channels)
    oIDM.nVirtual_Channels = len(oV_Channel_list)
    
    oIDM.Channels = Channels
    oIDM.Virtual_Channels = oV_Channel_list
    
    return oIDM
    
    