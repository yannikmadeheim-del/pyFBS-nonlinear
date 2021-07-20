import pytest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os


def test_SVT():
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

    # load experimental data
    freq, _Y_B_exp = np.load(exp_B, allow_pickle = True)
    Y_B_exp = np.transpose(_Y_B_exp, (2, 0, 1))

    # create SVT
    k = 3
    svt_B = pyFBS.SVT(df_chn,df_imp,freq,Y_B_exp,[1,10],k)

    # check IDMs
    assert(np.any(svt_B.Ru))
    assert(np.any(svt_B.Rf))

    # check transformation matrices
    assert(np.any(svt_B.Tu))
    assert(np.any(svt_B.Tf))

    # check filtering matrices
    assert(np.any(svt_B.Fu))
    assert(np.any(svt_B.Ff))


    
