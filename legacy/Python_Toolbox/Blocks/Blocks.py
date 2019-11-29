# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""



class Blocks(object):
    
    def __init__(self, t1 = None, l = None, Name = None):
        
        self.t1 = t1
        self.l = l
        self.t2 = l + t1
        self.tc = (2*t1 + l)/2
        self.z = t1
        self.Name = Name
        