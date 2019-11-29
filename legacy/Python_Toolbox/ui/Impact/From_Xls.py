# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:16:34 2018

@author: Umer Sherdil Paracha
"""
import pandas as pd

from Python_Toolbox.ui.Impact.Impact import Impact
from Python_Toolbox.Quantity.Quantity import Quantity
from Python_Toolbox.Unit.Unit import Unit


def fromXls(file_name,sheet_name):
    
    xls = pd.ExcelFile(file_name)
    Data = pd.read_excel(xls, sheet_name)
    
    Impact_List = list()
    
    for i in range(0,len(Data)):
        
        oImpact = Impact()
        
        oQuantity = Quantity()
        oUnit = Unit()
        
        oUnit.Name = Data['Unit'][i]
        
        oQuantity.Name = Data['Quantity'][i]
        oQuantity.Unit = oUnit
        
        oImpact.Node_Number = Data['NodeNumber'][i]
        oImpact.Grouping = Data['Grouping'][i]

        oImpact.Position = [Data['Position_1'][i], Data['Position_2'][i], Data['Position_3'][i]]
        oImpact.Direction = [Data['Direction_1'][i], Data['Direction_2'][i], Data['Direction_3'][i]]
        oImpact.Name = Data['Name'][i]
        oImpact.Description = Data['Description'][i]
        
        oImpact.Quantity = oQuantity
        oImpact.Unit = oUnit
        
        Impact_List.append(oImpact)
        
    return Impact_List

