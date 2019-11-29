# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""
from scipy.spatial import KDTree
from numpy import asarray

def nearest_neighbor(oNodes, data):
    
    node_positions = asarray([item.Position for item in oNodes])
    data_points = asarray([item.Position for item in data])
    
    tree = KDTree(node_positions)
    A, Nodes = tree.query([data_points])
    
    return Nodes