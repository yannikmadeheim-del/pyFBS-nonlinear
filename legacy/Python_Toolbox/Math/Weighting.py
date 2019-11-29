# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

    
# =============================================================================

"""

from math import sqrt
from numpy import log10, zeros

def weighting(Data, weighting_type):
    
    if weighting_type == 'A':
        
        iterator = 0
        A = zeros(len(Data.Freq), float)
        
        for freq in Data.Freq:
            Ra = 1.007145835*(pow(12200, 2) * pow(freq, 4)) / \
                 ((pow(freq, 2) + pow(20.6, 2)) * (pow(freq, 2) + pow(12200, 2)) * sqrt(pow(freq,2) + pow(107.7,2)) * sqrt(pow(freq,2) + pow(737.9,2))) 
            A[iterator] = 20*(log10(Ra)) + 2
            iterator += 1
            
        return A
    
     