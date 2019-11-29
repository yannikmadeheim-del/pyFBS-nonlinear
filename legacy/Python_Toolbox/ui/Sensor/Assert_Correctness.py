# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:16:34 2018

@author: Umer Sherdil Paracha
"""
from numpy import around, zeros
from scipy.spatial.transform import Rotation
from collections import Counter

def assert_correctness(sensor_data, channel_data):
    
    Matrix_List = list()
    Channel_Direction_List = list()
    Channel_List = list()
    Angle_List = list()
    
    nodes = Counter(channel_data["NodeNumber"])
    total_row_idx = channel_data.index.values
    Matrix = zeros((3,3),float)

    for item in total_row_idx:
        Channel_Direction_List.append([channel_data["Direction_1"][item], channel_data["Direction_2"][item], channel_data["Direction_3"][item]])
        
    for keys in nodes:
        for i in range (0,nodes[keys]):   
            Matrix[:,i] = Channel_Direction_List[0]
            Channel_Direction_List.pop(0)
        Channel_List.append(Matrix)
        Matrix = zeros((3,3),float)
              
        Angle_List = list()
    
    if not (sensor_data["Orientation_1"].isnull().values.all() or sensor_data["Orientation_2"].isnull().values.all() \
    or sensor_data["Orientation_3"].isnull().values.all()):
        
        for i in range(0,len(sensor_data)):
    
            oRot = Rotation.from_euler("xyz", [sensor_data["Orientation_1"][i], sensor_data["Orientation_2"][i], sensor_data["Orientation_3"][i]],degrees=True)
            Rot_Matrix = around(oRot.as_dcm())
            Matrix_List.append(Rot_Matrix)
            
        for i in range (0,len(sensor_data)):  
            
            if (Matrix_List[i] != Channel_List[i]).all():
                
                print("Angles of Sensor: ", i+1, " are not correct. Please check!")
                
                oRot = Rotation.from_dcm(Channel_List[i])
                Angles = oRot.as_euler('xyz', degrees=True)
                
                print("The Euler angles (xyz) format should be: ", Angles)
                
            else:
                
               Angles =  [sensor_data["Orientation_1"][i], sensor_data["Orientation_2"][i], sensor_data["Orientation_3"][i]]
           
            Angle_List.append(Angles)
            
    return Channel_List, Angle_List