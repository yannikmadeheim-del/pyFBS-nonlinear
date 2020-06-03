#!/usr/bin/env python

"""Tests for `pyfbs` package."""


import unittest
from click.testing import CliRunner
import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyFBS
# from pyFBS import cli

from test_data import overlay, DoF, loc_Y_num

class TestPyfbs(unittest.TestCase):
    """Tests for `pyfbs` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'pyfbs.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output

    def test_SEMM(self):

        path_val = "./test_data/SEMM_val_data.npz"

        val = np.load(path_val)

        Y_Num, Y_Exp, Y_SEMM_node5, Y_SEMM_full, Y_SEMM_full_red = val["Y_Num"], val["Y_Exp"], val["Y_SEMM_node5"], val["Y_SEMM_full"], val["Y_SEMM_full_red"]

        Y_Num_node5 = np.asarray(Y_Num[:, (np.asarray([8, 9])), :])
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num_node5, Y_Exp, overlay, DoF, loc_Y_num,  SEMM_type="fully-extend")[:, 0, 0], Y_SEMM_node5))
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num, Y_Exp, overlay, DoF,  SEMM_type="fully-extend")[:, 0, 0], Y_SEMM_full))
        self.assertIsNone(np.testing.assert_array_equal(pyFBS.SEMM(Y_Num, Y_Exp, overlay, DoF, SEMM_type="fully-extended-svd", red_comp=0, red_eq=20)[:, 0, 0], Y_SEMM_full_red))

if __name__ == '__main__':
    unittest.main()