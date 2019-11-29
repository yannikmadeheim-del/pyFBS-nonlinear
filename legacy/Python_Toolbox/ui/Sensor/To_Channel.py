# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:16:34 2018

@author: Umer Sherdil Paracha
"""

from Python_Toolbox.Channel.Channel import Channel

def to_channel(input_data):
    
    oChannel_list = list()
    Directions = {0:"+X", 1:"+Y", 2:"+Z"}
    
    if "Sensor" in input_data[0].Name or "sensor" in input_data[0].Name:    
             
        for i in range (0,len(input_data)): 
            for j in range (0,3): #Hard coded 
                
                oChannel = Channel()
                oChannel.Name = input_data[i].Name
                oChannel.Grouping = input_data[i].Grouping
                oChannel.Description = input_data[i].Description
                oChannel.Node_Number = input_data[i].Node_Number
                oChannel.Position = input_data[i].Position
                oChannel.Direction_Number = j + 1
                oChannel.Direction_Label = Directions[j]
                oChannel.Unit = input_data[i].Unit
                oChannel.Quantity = input_data[i].Quantity
                oChannel.Direction = input_data[i].Orientation.Matrix[:,j]
                
                oChannel_list.append(oChannel)
                
    elif "Impact" in input_data[0].Name or "impact" in input_data[0].Name: 
            
        for i in range (0,len(input_data)):
            
            oChannel = Channel()
            oChannel.Name = input_data[i].Name
            oChannel.Grouping = input_data[i].Grouping
            oChannel.Description = input_data[i].Description
            oChannel.Node_Number = input_data[i].Node_Number
            oChannel.Position = input_data[i].Position
            oChannel.Direction = input_data[i].Direction
            oChannel.Unit = input_data[i].Unit
            oChannel.Quantity = input_data[i].Quantity
            
            oChannel_list.append(oChannel)
            
    return oChannel_list