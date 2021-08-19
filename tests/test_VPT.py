import pytest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os


def test_VPT():
    # download testbench
    pyFBS.download_lab_testbench()

    # static names 
    exp_B = r"./lab_testbench/Measurements/Y_B.p"
    xlsx_pos = r"./lab_testbench/Measurements/ammeasurements.xlsx"

    # load channels
    df_imp = pd.read_excel(xlsx_pos, sheet_name='Impacts_B')
    df_chn = pd.read_excel(xlsx_pos, sheet_name='Channels_B')
    df_vp = pd.read_excel(xlsx_pos, sheet_name='VP_Channels')
    df_vpref = pd.read_excel(xlsx_pos, sheet_name='VP_RefChannels')

    # create VPT
    vpt_A = pyFBS.VPT(df_chn,df_imp,df_vp,df_vpref)

    # check IDMs
    assert(np.any(vpt_A.Ru))
    assert(np.any(vpt_A.Rf))

    # check transformation matrices
    assert(np.any(vpt_A.Tu))
    assert(np.any(vpt_A.Tf))

    # check filtering matrices
    assert(np.any(vpt_A.Fu))
    assert(np.any(vpt_A.Ff))

    
