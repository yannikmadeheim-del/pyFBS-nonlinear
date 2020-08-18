import unittest
import sys

import pyFBS
import numpy as np

import pandas as pd
import os.path
from os import path

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')


class TestPyfbs(unittest.TestCase):
    """
    Tests for `pyFBS` package.
    """

    def test_3Ddisplay_static(self):
        """
        Test static part of the 3D display. Evaluate only if the code runs without error.
        """

        # load 3D display
        view3D = pyFBS.view3D(off_screen = True)

        pyFBS.download_lab_testbench()


        # STL file
        stl_dir = r"./lab_testbench/STL/B.stl"
        view3D.add_stl(stl_dir, opacity=1)

        pos_xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"

        # sensors
        df_acc = pd.read_excel(pos_xlsx, sheet_name='Sensors_AB')
        view3D.show_acc(df_acc)
        view3D.label_acc(df_acc)

        # channels
        df = pd.read_excel(pos_xlsx, sheet_name='Channels_AB')
        view3D.show_chn(df)
        view3D.label_chn(df)

        # impacts
        df = pd.read_excel(pos_xlsx, sheet_name='Impacts_AB')
        view3D.show_imp(df)
        view3D.label_imp(df)

        # VPs
        df = pd.read_excel(pos_xlsx, sheet_name='VP_Channels')
        view3D.show_vp(df)
        view3D.label_vp(df)

        # clear everything
        view3D.plot.clear()

    def test_3Ddisplay_interactive(self):
        """
        Test interactive part of the 3D display. Evaluate only if the code runs without error.
        """
        pyFBS.download_lab_testbench()

        #view3D = pyFBS.view3D(off_screen = True) # cant use off_screen plotting, sphere widhets are not available
        view3D = pyFBS.view3D()

        # STL file
        stl = r"./lab_testbench/STL/A.stl"
        mesh = view3D.add_stl(stl, name="ts", color="#83afd2")

        # load the required DataFrames
        pos_xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"
        df_sensors = pd.read_excel(pos_xlsx, sheet_name='Sensors_A')
        df_impacts = pd.read_excel(pos_xlsx, sheet_name='Impacts_A')
        df_vp = pd.read_excel(pos_xlsx, sheet_name='VP_Channels')

        # add interactive accs
        view3D.add_acc_dynamic(mesh, predefined=df_sensors)
        df_acc_updated = view3D.get_acc_data()

        # generate channels from accs and vice versa
        df_chn_updated = pyFBS.utility.generate_channels_from_sensors(df_acc_updated)
        df_acc_from_chn = pyFBS.utility.generate_sensors_from_channels(df_chn_updated)

        # add interactive impacts
        view3D.add_imp_dynamic(mesh, predefined=df_impacts)
        df_imp_updated = view3D.get_imp_data()

        # virtual points
        view3D.add_vp_dynamic(mesh, predefined=df_vp)
        df_vp_updated = view3D.get_vp_data()

    def test_MCK(self):
        """
        Test of MK_model. Evaluate only if the code runs without error.
        """

        full_file = r"./lab_testbench/FEM/B.full"
        rst_file = r"./lab_testbench/FEM/B.rst"

        xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"

        df_acc = pd.read_excel(xlsx, sheet_name='Sensors_B')
        df_chn = pd.read_excel(xlsx, sheet_name='Channels_B')
        df_imp = pd.read_excel(xlsx, sheet_name='Impacts_B')

        MK = pyFBS.MK_model(rst_file, full_file, no_modes = 10, allow_pickle = False, recalculate = False)

        df_chn_up = MK.update_locations_df(df_chn)
        df_imp_up = MK.update_locations_df(df_imp)

        MK.FRF_synth(df_channel = df_chn, df_impact = df_imp,
             f_start = 0, f_end = 50, f_resolution = 1,
             limit_modes = 5, modal_damping = 0.003,
             frf_type = "accelerance")

        MK.add_noise(n1 = 2e-1, n2 = 2e-1, n3 = 5e-2 ,n4 = 5e-2)

    def test_VPT(self):
        """
        Test Virtual Point Transformation. Evaluate only if the code runs without error.
        """
        pyFBS.download_lab_testbench()

        pos_xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"


        df_imp = pd.read_excel(pos_xlsx, sheet_name='Impacts_B')
        df_chn = pd.read_excel(pos_xlsx, sheet_name='Channels_B')

        df_vp = pd.read_excel(pos_xlsx, sheet_name='VP_Channels')
        df_vpref = pd.read_excel(pos_xlsx, sheet_name='VP_RefChannels')

        vpt = pyFBS.VPT(df_chn, df_imp, df_vp, df_vpref)

        vpt.apply_VPT(np.asarray([1]), np.random.random((1,21,21)))
        vpt.consistency([1], [1])

    def test_SEMM(self):
        """
        Test System Equivalent Model Mixing. Evaluate only if the code runs without error.
        """

        pyFBS.download_lab_testbench()

        xlsx = r"./lab_testbench/Measurements/AM_measurements.xlsx"

        full_file = r"./lab_testbench/FEM/AB.full"
        rst_file = r"./lab_testbench/FEM/AB.rst"

        exp_file = r"./lab_testbench/Measurements/Y_AB.p"

        df_chn = pd.read_excel(xlsx, sheet_name='Channels_AB')
        df_imp = pd.read_excel(xlsx, sheet_name='Impacts_AB')

        MK = pyFBS.MK_model(rst_file, full_file, no_modes = 100, recalculate = False)

        freq, Y_exp = np.load(exp_file, allow_pickle = True)
        Y_exp = np.transpose(Y_exp, (2, 0, 1))

        MK.FRF_synth(df_chn,df_imp,
             f_start=0,
             f_end=2002.5,
             f_resolution=25,
             modal_damping = 0.003,
             frf_type = "accelerance")

        Y_AB_SEMM = pyFBS.SEMM(MK.FRF, Y_exp[::10, 0:15, 5:20],
                       df_chn_num = df_chn,
                       df_imp_num = df_imp,
                       df_chn_exp = df_chn[0:15],
                       df_imp_exp = df_imp[5:20],
                       SEMM_type='fully-extend-svd', red_comp=10, red_eq=10)

        Y_AB_SEMM = pyFBS.SEMM(MK.FRF, Y_exp[::10, 0:15, 5:20],
                       df_chn_num = df_chn,
                       df_imp_num = df_imp,
                       df_chn_exp = df_chn[0:15],
                       df_imp_exp = df_imp[5:20],
                       SEMM_type='fully-extend')

if __name__ == '__main__':
    unittest.main()