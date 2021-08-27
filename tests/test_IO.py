import pytest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os


# Tests the download of the automotive testbench
def test_automotive_testbench_download(automotive_testbench_file_names):

    pyFBS.download_automotive_testbench()
    
    all_files = automotive_testbench_file_names

    for file in all_files:
        assert os.path.isfile(file) 

# Tests the download of the lab testbench
def test_lab_testbench_download(lab_testbench_file_names):

    pyFBS.download_lab_testbench()
    
    all_files = lab_testbench_file_names

    for file in all_files:
        assert os.path.isfile(file) 

#TODO: add test for PAK function