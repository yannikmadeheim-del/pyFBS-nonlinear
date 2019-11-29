# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:16:20 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
Accepts:
    1. Nodal x coordinates read in function "data_reader".
    2. Nodal y coordinates read in function "data_reader".
    3. Nodal z coordinates read in function "data_reader".

Returns:
    1. xyz_list in the form of (x,y,z). 
    
When is the function called:
    This function is called for systems where the input is not read from ansys full file. 
    
# =============================================================================
"""

def xyz_list(x,y,z):
    xyz_list = list()
    xyz_list_temp = list()
    for i in range (len(x)):
        xyz_list_temp.append(str("%.6f" %x[i]))
        xyz_list_temp.append(str("%.6f" %y[i]))
        xyz_list_temp.append(str("%.6f" %z[i]))
        xyz_list.append(",".join(xyz_list_temp))
        xyz_list_temp = list()
        
    return xyz_list