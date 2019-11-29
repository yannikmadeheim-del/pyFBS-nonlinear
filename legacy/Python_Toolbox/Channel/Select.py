# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================



# =============================================================================

"""


def select(Data, Grouping = None, Name = None, Quantity = None):
    
    if Grouping is not None:
        Selected_Data = [d for d in Data for item in Grouping if d.Grouping == item]
    elif Name is not None:
        Selected_Data = [d for d in Data for item in Name if d.Name == item]
    elif Quantity is not None:
        Selected_Data = [d for d in Data if d.Quantity == Quantity]

    return Selected_Data

def find(Data, Grouping = None, Name = None):
    
    if Grouping is not None:
        Name_List = [d.Grouping for d in Data]
        Selected_Data = [index for index, value in enumerate(Name_List) for item in Grouping if value == item]
        
    elif Name is not None:
        Name_List = [d.Name for d in Data]
        Selected_Data = Name_List.index(Name)

    return Selected_Data