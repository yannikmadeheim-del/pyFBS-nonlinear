# import everything
from .IO import *
from .utility import *
from .VPT import *
from .display import *
from .SEMM import *
from .MCK import *


# relative directories for example datasets
from pathlib import Path
import os 

# automotive testbench
example_auto_testbench = {}



# geometry files - stl
ts = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "ts.stl"
transmission_mount = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "transmission_mount.stl"
shaker_only = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "shaker_only.stl"
roll_mount = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "roll_mount.stl"
receiver = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "receiver.stl"
engine_mount = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "STL" + os.sep + "engine_mount.stl"

example_auto_testbench["STL"] = {"ts": ts,"transmission_mount": transmission_mount,"shaker_only": shaker_only,"roll_mount": roll_mount,"receiver": receiver,"engine_mount": engine_mount}

# measurements
xlsx_A = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "A.xlsx"
xlsx_AB_ref = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "AB_ref.xlsx"
xlsx_B_ref = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "B_ref.xlsx"
xlsx_BTS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "BTS.xlsx"
xlsx_TS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "TS.xlsx"

Y_A = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "A.p"
Y_AB_ref = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "AB_ref.p"
Y_B_ref = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "B_ref.p"
Y_BTS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "BTS.p"
Y_TS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "TS.p"


xlsx_ODS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "ODS.xlsx"
Y_ODS = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "ODS.p"

xlsx_modal = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "modal.xlsx"
Y_m_1 = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "frame_rubbermounts_sourceplate.p"
Y_m_2 = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "Measurements" + os.sep + "frame_rubbermounts.p"



example_auto_testbench["meas"] = {"xlsx_A":xlsx_A,"xlsx_AB_ref":xlsx_AB_ref,"xlsx_B_ref":xlsx_B_ref,"xlsx_BTS":xlsx_BTS,"xlsx_TS":xlsx_TS,"xlsx_ODS":xlsx_ODS,"xlsx_modal":xlsx_modal,"Y_m_1":Y_m_1,"Y_m_2":Y_m_2,"Y_A":Y_A,"Y_AB_ref":Y_AB_ref,"Y_B_ref":Y_B_ref,"Y_BTS":Y_BTS,"Y_TS":Y_TS,"Y_ODS":Y_ODS}

TM_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "TM.rst"
RM_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "RM.rst"
EM_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "EM.rst"

TM_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "TM.full"
RM_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "RM.full"
EM_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "automotive_testbench" + os.sep + "FEM" + os.sep + "EM.full"

example_auto_testbench["FEM"] = {"TM_rst": TM_rst, "RM_rst": RM_rst, "EM_rst": EM_rst, "TM_full": TM_full, "RM_full": RM_full, "EM_full": EM_full}

# laboratory testbench
example_lab_testbench = {}


# geometry files - stl
A = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "STL" + os.sep + "A.stl"
B = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "STL" + os.sep + "B.stl"
AB = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "STL" + os.sep + "AB.stl"

example_lab_testbench["STL"] = {"A": A,"B": B,"AB": AB}


# measurements
xlsx = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "AM_Measurements.xlsx"
Y_A = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "Y_A.p"
Y_B = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "Y_B.p"
Y_AB = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "Y_AB.p"

xlsx_coupling = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "coupling_example.xlsx"
xlsx_decoupling = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "Measurements" + os.sep + "decoupling_example.xlsx"

example_lab_testbench["meas"] = {"xlsx": xlsx,"xlsx_coupling": xlsx_coupling,"xlsx_decoupling": xlsx_decoupling, "Y_A": Y_A,"Y_B": Y_B,"Y_AB": Y_AB}

# FEM 
A_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "A" + os.sep + "file.rst"
A_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "A" + os.sep + "file.full"

B_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "B" + os.sep + "file.rst"
B_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "B" + os.sep + "file.full"

AB_rst = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "AB" + os.sep + "file.rst"
AB_full = str(Path(__file__).parents[1] )+os.sep+"data" +os.sep + "lab_testbench" + os.sep + "FEM" + os.sep + "AB" + os.sep + "file.full"

example_lab_testbench["FEM"] = {"A_rst": A_rst, "A_full": A_full,"B_rst": B_rst, "B_full": B_full,"AB_rst": AB_rst, "AB_full": AB_full}
