# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:16:34 2018

@author: Umer Sherdil Paracha
"""

import pandas as pd
import numpy as np

from Python_Toolbox.ui.Sensor.Sensor import Sensor
from Python_Toolbox.Quantity.Quantity import Quantity
from Python_Toolbox.Unit.Unit import Unit
from Python_Toolbox.Orientation.Orientation import Orientation
from Python_Toolbox.ui.Sensor.Assert_Correctness import assert_correctness

def fromXls(file_name, sheet_name_sensor, sheet_name_channel):
    
    xls = pd.ExcelFile(file_name)
    Data_Sensor = pd.read_excel(xls, sheet_name_sensor)  
    Data_Channel = pd.read_excel(xls, sheet_name_channel) 
    
    Channel_Matrix_List, Angle_List = assert_correctness(Data_Sensor, Data_Channel)
    
    Sensor_List = list()
    
    for i in range(0,len(Data_Sensor)):
        
        
        oSensor = Sensor()
        
        oQuantity = Quantity()
        oUnit = Unit()
        oOrientation = Orientation()
        
        oUnit.Name = Data_Sensor['Unit'][i]
        
        if "Orientation_4" in Data_Sensor:
                
            Local_X = np.asarray([Data_Sensor["Orientation_1"][i], Data_Sensor["Orientation_2"][i], Data_Sensor["Orientation_3"][i]])
            Local_Y = np.asarray([Data_Sensor["Orientation_4"][i], Data_Sensor["Orientation_5"][i], Data_Sensor["Orientation_6"][i]])
            Local_Z = np.asarray([Data_Sensor["Orientation_7"][i], Data_Sensor["Orientation_8"][i], Data_Sensor["Orientation_9"][i]])
            oOrientation.Matrix = np.asarray([Local_X,Local_Y,Local_Z])
                       
        else:
            
            if Angle_List:
                oOrientation.Angles = Angle_List[i]
                
            oOrientation.Matrix = Channel_Matrix_List[i]
        
        oQuantity.Name = Data_Sensor['Quantity'][i]
        oQuantity.Unit = oUnit
        
        oSensor.Type = Data_Sensor['Type'][i]
        oSensor.Size = Data_Sensor['Size'][i]
        oSensor.Node_Number = Data_Sensor['NodeNumber'][i]
        oSensor.Grouping = Data_Sensor['Grouping'][i]

        oSensor.Position = [Data_Sensor['Position_1'][i], Data_Sensor['Position_2'][i], Data_Sensor['Position_3'][i]]
        oSensor.Name = Data_Sensor['Name'][i]
        oSensor.Description = Data_Sensor['Description'][i]
        
        
        oSensor.Orientation = oOrientation
        oSensor.Quantity = oQuantity
        oSensor.Unit = oUnit
        
        Sensor_List.append(oSensor)
        
    return Sensor_List
        

