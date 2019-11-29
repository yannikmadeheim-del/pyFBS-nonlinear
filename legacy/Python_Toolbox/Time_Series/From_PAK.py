# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""
from math import ceil

from Python_Toolbox.FRF.Assign_Grouping_Number import assign_grouping_number
from Python_Toolbox.FRF.Create_Matrix_Data import create_matrix_data

from Python_Toolbox.Time_Series.Time_Series import Time_Series
from Python_Toolbox.Channel.Channel import Channel

def from_PAK(input_data, sensors, impacts):
    
    oTime_Series = Time_Series()
    
    #%% Assign miscellaneous items
    
    oTime_Series.dt = input_data[0].DataSets.X_Channels.Data[1] - input_data[0].DataSets.X_Channels.Data[0]
    oTime_Series.T = input_data[0].DataSets.X_Channels.Data[-1]
    oTime_Series.fs = ceil(1/oTime_Series.dt)
    oTime_Series.Time = input_data[0].DataSets.X_Channels.Data
    oTime_Series.nTime = len(oTime_Series.Time)
    input_data = assign_grouping_number(input_data, sensors, impacts)
    
    #%% Creating data matrix
    
    oTime_Series.Data = create_matrix_data(input_data, len(input_data[0].DataSets.X_Channels.Data))
    
    #%% Channels and RefChannels
    
    oChannel_list = list()  
    oRef_Channel_list = list()
    Channel_Direction = list()
    
    Sensor_Names = sorted(set([d.DataSets.Y_Channels.Channel_Info.Name for d in input_data]))
    
    Impact_Names = sorted(set([d.DataSets.Y_Channels.Ref_Channel_Info.Name for d in input_data]))
    
    Channels = [d.DataSets.Y_Channels.Channel_Info for d in input_data if d.DataSets.Y_Channels.Channel_Info.Name \
                in Sensor_Names and d.DataSets.Y_Channels.Ref_Channel_Info.Name == Impact_Names[0]]
    
    Ref_Channels = [d.DataSets.Y_Channels.Ref_Channel_Info for d in input_data if d.DataSets.Y_Channels.Ref_Channel_Info.Name \
                in Impact_Names and d.DataSets.Y_Channels.Channel_Info.Name == Sensor_Names[0]]
    
    for i in range (0,len(Channels)):
        
        oChannel = Channel()
        oChannel.Name = Channels[i].Name
        oChannel.Grouping = Channels[i].Grouping
        oChannel.Node = Channels[i].Node
        oChannel.Node_Number = Channels[i].Node_Number
        oChannel.Quantity = Channels[i].Quantity
        oChannel.Unit = Channels[i].Unit
        oChannel.Direction_Label = Channels[i].Direction_Label
        oChannel.Direction_Number = Channels[i].Direction_Number
        oChannel_list.append(oChannel)
    
        
    for i in range (0,len(Ref_Channels)):
        
        oRef_Channel = Channel()
        oRef_Channel.Name = Ref_Channels[i].Name
        oRef_Channel.Grouping = Ref_Channels[i].Grouping
        oRef_Channel.Node = Ref_Channels[i].Node
        oRef_Channel.Node_Number = Ref_Channels[i].Node_Number
        oRef_Channel.Quantity = Ref_Channels[i].Quantity
        oRef_Channel.Unit = Ref_Channels[i].Unit
        oRef_Channel.Direction_Label = Ref_Channels[i].Direction_Label
        oRef_Channel.Direction_Number = Ref_Channels[i].Direction_Number
        oRef_Channel_list.append(oRef_Channel)
        
    Channel_Position = [s.Position for s in sensors for c in Channels if s.Node_Number == c.Node_Number]
    
    for s in sensors:
        counter = 0
        for c in Channels:
            if s.Node_Number == c.Node_Number:
                Channel_Direction.append(s.Orientation.Matrix[counter,:])
                counter = counter + 1
    
    for i in range (0,len(Channels)):
        oChannel_list[i].Node.Position = Channel_Position[i]
        oChannel_list[i].Position = Channel_Position[i]
        oChannel_list[i].Direction = Channel_Direction[i]
      
    oTime_Series.Channels = oChannel_list    
    oTime_Series.nChannels = len(Channels)
    
    
    Ref_Channel_Direction = [i.Direction for i in impacts for r in Ref_Channels if i.Node_Number == r.Node_Number]
    Ref_Channel_Position = [i.Position for i in impacts for r in Ref_Channels if i.Node_Number == r.Node_Number]
    
    for i in range (0,len(Ref_Channel_Direction)):
        
        oRef_Channel_list[i].Direction = Ref_Channel_Direction[i]
        oRef_Channel_list[i].Node.Position = Ref_Channel_Position[i]
        oRef_Channel_list[i].Position = Ref_Channel_Position[i]

    
    oTime_Series.RefChannels = oRef_Channel_list
    oTime_Series.nRefChannels = len(Ref_Channels)
    
    return oTime_Series