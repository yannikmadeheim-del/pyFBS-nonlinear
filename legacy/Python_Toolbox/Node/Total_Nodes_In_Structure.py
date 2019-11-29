# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:46:51 2018

@author: Umer Sherdil Paracha
"""
"""
# =============================================================================
If Input = List, then the data was read from an excel file or a full file. Hence the nodes will be in the form of a list.
If Type = Array, then the FRF data was read directly from .mat file and total nodes in system have to be judged based on total DOFs in the given FRF matrix and number of dofs per node.
# =============================================================================
"""

import numpy as np

def total_nodes_in_structure(self, Input = None):
    if (isinstance(Input,list)):
        self.nodes = Input
    elif (isinstance(Input,np.ndarray)):
        ndof_per_node = self.oDOF.ndof_per_node
        self.nodes = np.arange(1,len(Input[:,0])/ndof_per_node + 1)
                         
    