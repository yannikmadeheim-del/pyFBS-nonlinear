# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:41:48 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

Accepts: 
    1. A [1x3] list where each entry represents the angle of sensor/impact (in radians) w.r.t the x, y and z axis respectively.
    
Returns a [3x3] orientation matrix. 

Info link: "https://en.wikipedia.org/wiki/Rotation_matrix"

# =============================================================================

"""
from numpy import asarray, eye
from math import cos, sin

def rotation_matrices(orientation_vector):
    
    Index_Nonzero_entries = [i for i, e in enumerate(orientation_vector) if e != 0]
    
    if 0 in Index_Nonzero_entries:
        
        Rot_x = asarray([[1,0,0],[0,cos(orientation_vector[0]),-sin(orientation_vector[0])], [0, sin(orientation_vector[0]), cos(orientation_vector[0])]])
        
    if 0 not in Index_Nonzero_entries:
        
        Rot_x = eye(3,3)
     
    if 1 in Index_Nonzero_entries:
        
        Rot_y = asarray([[cos(orientation_vector[1]), 0, sin(orientation_vector[1])],[0,1,0],[-sin(orientation_vector[1]), 0, cos(orientation_vector[1])]])
        
    if 1 not in Index_Nonzero_entries:
        
        Rot_y = eye(3,3)
        
    if 2 in Index_Nonzero_entries:
        
        Rot_z = asarray([[cos(orientation_vector[2]), -sin(orientation_vector[2]), 0],[sin(orientation_vector[2]), cos(orientation_vector[2]), 0],[0,0,1]])
    
    if 2 not in Index_Nonzero_entries:
        Rot_z = eye(3,3)
        
    
    Rot = Rot_x.dot(Rot_y).dot(Rot_z)
    
    
    return Rot