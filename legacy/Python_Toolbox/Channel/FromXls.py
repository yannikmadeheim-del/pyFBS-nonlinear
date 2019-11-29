# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""
import pandas as pd

"""
# =============================================================================
# Reads data from the excel file.
# =============================================================================
"""
def fromXls(excel_file,excel_sheet):
    
    data = pd.read_excel(excel_file,excel_sheet)
    
    return data 
    
    
        