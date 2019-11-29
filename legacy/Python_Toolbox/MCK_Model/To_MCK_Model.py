# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 15:28:02 2018

@author: Umer Sherdil Paracha
"""

"""
# =============================================================================

# =============================================================================

"""

from Python_Toolbox.MCK_Model.MCK_Model import MCK_Model

def to_mck_model(dof_reference_table, k, m, BC_nodes_and_dofs, ndof_per_node):
    oMCK_Model = MCK_Model(k, m)
        