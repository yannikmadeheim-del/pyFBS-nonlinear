%load_ext autoreload
%autoreload 2
import numpy as np

from pyfbs.io import Impacts,Channels,Sensors
from pyfbs.vpt import VPT
import os


path_list = os.path.abspath('__file__').split(os.sep)
script_directory = path_list[0:len(path_list)-1]

example_xlsx = "/".join(script_directory) + "/" + "data/test_geometry.xlsx"


#%%

from pyfbs.io import Sensors,Impacts
from pyfbs.util import read_uff_file
from pyfbs.frf import FRF
import pandas as pd
import numpy as np

ASensors = Sensors(example_xlsx,"Sensors", "Channels")
AImpacts = Impacts(example_xlsx,"Impacts")


example_frf = "/".join(script_directory) + "/" + "data/rawFRF.uf"
rawData = read_uff_file(example_frf)

example_frf = "/".join(script_directory) + "/" + "data/rawCoh.uf"
rawCoh = read_uff_file(example_frf)


Y = FRF()
Y.from_series_to_matrix(rawData,ASensors,AImpacts,_coh = rawCoh)


#%%
gg = VPT(example_xlsx)
gg.apply_VPT(Y)

