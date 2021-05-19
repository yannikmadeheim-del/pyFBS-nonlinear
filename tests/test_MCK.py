
import pytest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os

# Tests, if files are downloaded and imports file names from conftest.py files
def test_data_load(file_names_B, file_names_xlsx):
    full_file, rst_file, stl_file = file_names_B
    xlsx = file_names_xlsx
    pyFBS.download_lab_testbench()
    assert os.path.isfile(full_file) 
    assert os.path.isfile(rst_file) 
    assert os.path.isfile(stl_file) 
    assert os.path.isfile(xlsx) 

# Checks all combination of defined parameters
@pytest.mark.parametrize("allow_pickle", [False, True])
@pytest.mark.parametrize("recalculate", [False, True])
def test_model_initialization(file_names_B, allow_pickle, recalculate):
    full_file, rst_file = file_names_B[:2]
    MK = pyFBS.MK_model(rst_file, full_file, no_modes = 10, allow_pickle = allow_pickle, recalculate = recalculate)
    pass

# Checks only prescribed combinations of defined parameters
@pytest.mark.parametrize(
    "limit_modes, modal_damping, frf_type, _all", [
        (None, None, "accelerance", False),
        (5, 10**-3, "mobility", True),
        (None, None, "receptance", False),
        ])
def test_FRF_synt(files_data_frames, MK_model, limit_modes, modal_damping, frf_type, _all):
    df_acc, df_chn, df_imp = files_data_frames
    MK = MK_model
    MK.FRF_synth(df_chn, df_imp, f_start = 0, f_end = 2, f_resolution = 1, limit_modes=limit_modes, modal_damping=modal_damping, frf_type=frf_type, _all=_all)

    assert isinstance(MK.FRF, np.ndarray)


