import unittest
import sys

import pyFBS
import numpy as np
from test_data import *

import pandas as pd
import os.path
from os import path

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')


class TestPyfbs(unittest.TestCase):
    """
    Tests for `pyFBS` package.
    """


    def test_example_data(self):
        """
        Check if the relative directions to example datasets are correct.
        """

        for key_one in pyFBS.example_lab_testbench:
            for key_two in pyFBS.example_lab_testbench[key_one]:
                assert path.exists(pyFBS.example_lab_testbench[key_one][key_two])

        for key_one in pyFBS.example_auto_testbench:
            for key_two in pyFBS.example_auto_testbench[key_one]:
                assert path.exists(pyFBS.example_auto_testbench[key_one][key_two])

    def test_3Ddisplay_static(self):
        """
        Test static part of the 3D display. Evaluate only if the code runs without error.
        """

        # load 3D display
        view3D = pyFBS.view3D(off_screen = True)

        # STL file
        stl_dir = pyFBS.example_lab_testbench["STL"]["B"]
        view3D.add_stl(stl_dir, opacity=1)

        pos_xlsx = pyFBS.example_lab_testbench["meas"]["xlsx"]

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
        #view3D = pyFBS.view3D(off_screen = True) # cant use off_screen plotting, sphere widhets are not available
        view3D = pyFBS.view3D()

        # STL file
        stl = pyFBS.example_lab_testbench["STL"]["A"]
        mesh = view3D.add_stl(stl, name="ts", color="#83afd2")

        # load the required DataFrames
        pos_xlsx = pyFBS.example_lab_testbench["meas"]["xlsx"]
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

    def test_VPT(self):
        """
        Test Virtual Point Transformation. Evaluate only if the code runs without error.
        """

        xlsx_pos = pyFBS.example_lab_testbench["meas"]["xlsx"]

        df_imp = pd.read_excel(xlsx_pos, sheet_name='Impacts_B')
        df_chn = pd.read_excel(xlsx_pos, sheet_name='Channels_B')

        df_vp = pd.read_excel(xlsx_pos, sheet_name='VP_Channels')
        df_vpref = pd.read_excel(xlsx_pos, sheet_name='VP_RefChannels')

        vpt = pyFBS.VPT(df_chn, df_imp, df_vp, df_vpref)

        vpt.apply_VPT(np.asarray([1]), np.random.random((1,21,21)))
        vpt.consistency([1], [1])

    def test_SEMM(self):
        """
        Test System Equivalent Model Mixing. Evaluate only if the code runs without error.
        """

        path_val = "./test_data/SEMM_val_data.npz"

        val = np.load(path_val)

        Y_Num, Y_Exp, Y_SEMM_node5, Y_SEMM_full, Y_SEMM_full_red = val["Y_Num"], val["Y_Exp"], val["Y_SEMM_node5"], val["Y_SEMM_full"], val["Y_SEMM_full_red"]

        Y_Num_node5 = np.asarray(Y_Num[:, (np.asarray([8, 9])), :])
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num_node5, Y_Exp, overlay, DoF, loc_Y_num,  SEMM_type="fully-extend")[:, 0, 0], Y_SEMM_node5))
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num, Y_Exp, overlay, DoF,  SEMM_type="fully-extend")[:, 0, 0], Y_SEMM_full))
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num, Y_Exp, overlay, DoF, SEMM_type="fully-extended-svd", red_comp=0, red_eq=20)[:, 0, 0], Y_SEMM_full_red))


if __name__ == '__main__':
    unittest.main()