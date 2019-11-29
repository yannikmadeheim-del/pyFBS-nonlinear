# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 
    1. FRF Series object.
    2. Sensor object
    3. Impact object
    
Returns the uncoupled FRF matrix object.

# =============================================================================

"""

from Python_Toolbox.FRF.FRF import FRF
from Python_Toolbox.FRF.Create_Matrix_Data import create_matrix_data
from Python_Toolbox.Channel.Channel import Channel
from Python_Toolbox.Unit.Unit import Unit

from Python_Toolbox.FRF.Assign_Grouping_Number import assign_grouping_number

def from_series_to_matrix(sub_structure, sensors, impacts):
    
    sub_structure = assign_grouping_number(sub_structure, sensors, impacts)
     
    oFRF = FRF()
    oChannel_list = list()
    oRef_Channel_list = list()
    Channel_Direction = list()
    
    
    oFRF.Freqs = sub_structure[0].DataSets.X_Channels.Data
    oFRF.nFreq = len(oFRF.Freqs)
         
    oFRF.Data = create_matrix_data(sub_structure, oFRF.nFreq)
    
    oFRF.Measurement_Info = sub_structure[0].Measurement_Info.Date
    
    Sensor_Names = sorted(set([d.DataSets.Y_Channels.Channel_Info.Name for d in sub_structure]))
    
    Impact_Names = sorted(set([d.DataSets.Y_Channels.Ref_Channel_Info.Name for d in sub_structure]))
    
    Channels = [d.DataSets.Y_Channels.Channel_Info for d in sub_structure if d.DataSets.Y_Channels.Channel_Info.Name \
                in Sensor_Names and d.DataSets.Y_Channels.Ref_Channel_Info.Name == Impact_Names[0]]
    
    Ref_Channels = [d.DataSets.Y_Channels.Ref_Channel_Info for d in sub_structure if d.DataSets.Y_Channels.Ref_Channel_Info.Name \
                in Impact_Names and d.DataSets.Y_Channels.Channel_Info.Name == Sensor_Names[0]]
       
    for i in range (0,len(Channels)):
        
        oUnit = Unit()
        oChannel = Channel()
        
        oUnit.Name = Channels[i].Unit
        oChannel.Name = Channels[i].Name
        oChannel.Grouping = Channels[i].Grouping
        oChannel.Node = Channels[i].Node
        oChannel.Node_Number = Channels[i].Node_Number
        oChannel.Quantity = Channels[i].Quantity
        oChannel.Unit = oUnit
        oChannel.Direction_Label = Channels[i].Direction_Label
        oChannel.Direction_Number = Channels[i].Direction_Number
        oChannel_list.append(oChannel)
        
    for i in range (0,len(Ref_Channels)):
        
        oUnit = Unit()
        oRef_Channel = Channel()
        
        oUnit.Name = Ref_Channels[i].Unit
        oRef_Channel.Name = Ref_Channels[i].Name
        oRef_Channel.Grouping = Ref_Channels[i].Grouping
        oRef_Channel.Node = Ref_Channels[i].Node
        oRef_Channel.Node_Number = Ref_Channels[i].Node_Number
        oRef_Channel.Quantity = Ref_Channels[i].Quantity
        oRef_Channel.Unit = oUnit
        oRef_Channel.Direction_Label = Ref_Channels[i].Direction_Label
        oRef_Channel.Direction_Number = Ref_Channels[i].Direction_Number
        oRef_Channel_list.append(oRef_Channel)
        
            
    Channel_Position = [s.Position for s in sensors for c in Channels if s.Node_Number == c.Node_Number]
    
    for s in sensors:
        counter = 0
        for c in Channels:
            if s.Node_Number == c.Node_Number:
                Channel_Direction.append(s.Orientation.Matrix[:,counter])
                counter = counter + 1
    
    for i in range (0,len(Channels)):
        oChannel_list[i].Node.Position = Channel_Position[i]
        oChannel_list[i].Position = Channel_Position[i]
        oChannel_list[i].Direction = Channel_Direction[i]
      
    oFRF.Channels = oChannel_list    
    oFRF.nChannels = len(Channels)
    
    
    Ref_Channel_Direction = [i.Direction for i in impacts for r in Ref_Channels if i.Node_Number == r.Node_Number]
    Ref_Channel_Position = [i.Position for i in impacts for r in Ref_Channels if i.Node_Number == r.Node_Number]
    
    for i in range (0,len(Ref_Channel_Direction)):
        
        oRef_Channel_list[i].Direction = Ref_Channel_Direction[i]
        oRef_Channel_list[i].Node.Position = Ref_Channel_Position[i]
        oRef_Channel_list[i].Position = Ref_Channel_Position[i]

    
    oFRF.RefChannels = oRef_Channel_list
    oFRF.nRefChannels = len(Ref_Channels)

    
    return oFRF