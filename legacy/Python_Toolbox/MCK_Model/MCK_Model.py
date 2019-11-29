# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""
from Python_Toolbox.Node.Node import Node

class MCK_Model(object):
    def __init__(self, k = None, m = None, nodes_and_coordinates = None):
        
        self.M = m
        self.C = None
        self.K = k
        self.Nodes = Node()
        
        if nodes_and_coordinates is not None:
            
            node_list = list() 
            
            for key in nodes_and_coordinates:  
                oNode = Node()
                oNode.NodeNumber = key
                oNode.Position = nodes_and_coordinates[key]
                node_list.append(oNode)
                
            self.Nodes = node_list
                
        else:
            self.Nodes = Node()
        
        self.DOFMapping = None
        self.DoFs = None
        self.nDoFs = None
        self.nNodes = None
        self.Name = None
        self.Description = None
        self.Parameters = None