# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:57:11 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

self.Position_xyz_list = List of sensor position in space stored as comma separated values.

# =============================================================================

"""


class Sensor(object):
    def __init__(self):
        
        self.Type = None
        self.Size = None
        self.Mass = None
        self.Node_Number = None
        self.Grouping = None
        self.Quantity = None
        self.Unit = None
        self.Position = None
        self.Orientation = None
        self.Name = None
        self.Description = None
        self.Alignment = None
        
        self.Position_xyz_list = list() #Old