# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

Definitions:

Class variables:
    1. Name: Name of Physical Quantity.
    2. Unit: Reference to class Unit.
    
# =============================================================================

"""

class Quantity(object):
    def __init__(self):
        
        self.Name = None
        self.Unit = None