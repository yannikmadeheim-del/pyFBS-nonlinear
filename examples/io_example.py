

%load_ext autoreload
%autoreload 2

from pyfbs.io import Impacts,Sensors
from pyfbs.util import read_uff_file

from pyfbs.frf import FRF

import os
import numpy as np

# get path to data

path_list = os.path.abspath('__file__').split(os.sep)
script_directory = path_list[0:len(path_list)-1]
example_xlsx = "/".join(script_directory) + "/" + "data/test_geometry.xlsx"

#%%
import pandas as pd

xls = pd.ExcelFile(example_xlsx)
Data = pd.read_excel(xls, "Sensors")

#%%
ASensors = Sensors(example_xlsx,"Sensors", "Channels")

AImpacts = Impacts(example_xlsx,"Impacts")
#%%

example_frf = "/".join(script_directory) + "/" + "data/rawFRF.uf"
rawData = read_uff_file(example_frf)

example_frf = "/".join(script_directory) + "/" + "data/rawCoh.uf"
rawCoh = read_uff_file(example_frf)

#%%
Y = FRF()
Y.from_series_to_matrix(rawData,ASensors,AImpacts,_coh = rawCoh)
#%%
"""
.. _ref_load_shaft_result:
Shaft Modal Analysis
~~~~~~~~~~~~~~~~~~~~
Visualize a full cyclic model
"""
import pyansys

###############################################################################
# load a sector modal analysis file
rotor = pyansys.download_sector_modal()
print(rotor)


###############################################################################
# plot the rotor
rotor.plot(smooth_shading=True)


###############################################################################
# plot a sector of the rotor
rotor.mas_grid.plot(color='w', smooth_shading=True)


###############################################################################
# plot a nodal solution
rotor.plot_nodal_solution(0)


###############################################################################
# animate a mode
rotor.animate_nodal_solution(1, interactive=True)
# set interactive to True to enable continuous plotting