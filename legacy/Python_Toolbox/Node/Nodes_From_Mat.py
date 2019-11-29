# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================

# =============================================================================
"""

from Python_Toolbox.Node.Node import Node

def nodes_from_mat(nodes):
    
    oNode_List = list()
    
    for i in range (0,len(nodes)):
        oNode = Node()
        oNode.NodeNumber = int(nodes[i])
        oNode.Position = [nodes[i,1], nodes[i,2], nodes[i,3]]
        oNode_List.append(oNode)
        
    return oNode_List