# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:55:29 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

Accepts:
    1. Object of class Coupling.
    2. Response structure, node and dof. 
    3. Excitation structure, node and dof

Plots the coupled_FRF matrix at the specified inputs.

# =============================================================================
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_coupled_FRF(self, response_structure, response_node, response_dof, excitation_structure, subcase, excitation_dof = None):
    values = list()
    response_node_with_dof_with_structure = str(response_node) + "," + str(response_dof) + "," + str(response_structure)
    if excitation_dof is not None:    
        excitation_node_with_dof_with_structure = str(subcase) + "," + str(excitation_dof) + "," + str(excitation_structure)
    else:
        excitation_node_with_dof_with_structure = str(subcase) + "," + str(excitation_structure)
    for freq in self.frequencies:    
        Temp = self.y_coupled[freq]
        values.append(abs(Temp.loc[response_node_with_dof_with_structure, excitation_node_with_dof_with_structure]))
        
    values = np.asarray(values)  
#    plt.plot(self.frequencies, 20*np.log10(values/max(values)))
    plt.plot(self.frequencies, values)
    print(min(values))