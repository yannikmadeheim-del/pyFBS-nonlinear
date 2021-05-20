import pytest
import pandas as pd
import pyFBS

@pytest.fixture()
def file_names_A():
    full_file = r"./lab_testbench/FEM/A.full"
    rst_file = r"./lab_testbench/FEM/A.rst"
    stl_file = r"./lab_testbench/STL/A.stl"
    return full_file, rst_file, stl_file

@pytest.fixture()
def file_names_B():
    full_file = r"./lab_testbench/FEM/B.full"
    rst_file = r"./lab_testbench/FEM/B.rst"
    stl_file = r"./lab_testbench/STL/B.stl"
    return full_file, rst_file, stl_file

@pytest.fixture()
def file_names_AB():
    full_file = r"./lab_testbench/FEM/AB.full"
    rst_file = r"./lab_testbench/FEM/AB.rst"
    stl_file = r"./lab_testbench/STL/AB.stl"
    return full_file, rst_file, stl_file

@pytest.fixture()
def file_names_xlsx():
    xlsx = r"./lab_testbench/Measurements/AM_Measurements.xlsx"
    return xlsx

@pytest.fixture()
def files_data_frames(file_names_xlsx):
    xlsx = file_names_xlsx
    df_acc = pd.read_excel(xlsx, sheet_name='Sensors_B')
    df_chn = pd.read_excel(xlsx, sheet_name='Channels_B')
    df_imp = pd.read_excel(xlsx, sheet_name='Impacts_B')
    return df_acc, df_chn, df_imp

@pytest.fixture()
def MK_model(file_names_B):
    full_file, rst_file, xlsx = file_names_B
    MK = pyFBS.MK_model(rst_file, full_file, no_modes = 10, allow_pickle = True, recalculate = False)
    return MK