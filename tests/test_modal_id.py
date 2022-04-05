import pytest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os


def test_modal_id_initalization():
    # download testbench
    pyFBS.download_lab_testbench()

    # static names 
    exp_file = r"./lab_testbench/Measurements/Y_A.p"

    freq, Y_AB_exp = np.load(exp_file, allow_pickle = True)
    Y_AB_exp = np.transpose(Y_AB_exp, (2, 0, 1))

    _id = pyFBS.modal_id(freq, Y_AB_exp)

    assert isinstance(_id.FRF, np.ndarray)
    assert isinstance(_id.freq, np.ndarray)


def test_modal_id_pLSCF(modal_id_):
    modal_id_.pLSCF(max_order=20)

    assert isinstance(modal_id_.stab_plot, np.ndarray)
    assert isinstance(modal_id_.poles, list)
    assert isinstance(modal_id_.mpf, list)
