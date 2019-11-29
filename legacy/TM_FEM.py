from numpy import arange
import pickle

from Python_Toolbox.util.Ansys_Data_Reader import read_ansys_full_file

from Python_Toolbox.MCK_Model.MCK_Model import MCK_Model
from Python_Toolbox.MCK_Model.To_Modal_Model import to_modal_model

from Python_Toolbox.Modal_Model.Add_Damping import add_damping

from Python_Toolbox.ui.Sensor.From_Xls import fromXls as fromXls_Sensor
from Python_Toolbox.ui.Impact.From_Xls import fromXls as fromXls_Impact
from Python_Toolbox.Channel.FromXls import fromXls as fromXls_VPChannels
from Python_Toolbox.ui.Sensor.To_Channel import to_channel

from Python_Toolbox.Modal_Model.To_FRF_Matrix import to_FRF_matrix

from Python_Toolbox.FRF.Plot_FRF import plot_FRF as plot
from Python_Toolbox.Channel.Select import find
from Python_Toolbox.Channel.Create_IDM_Matrix import create_IDM_matrix

from Python_Toolbox.FRF.Consistency import consistency

from Python_Toolbox.FRF.Create_Reduced_FRF_Matrix import create_reduced_FRF_matrix


#%% Output File

Output = 'Yqm_TM'

#%% MCK and Modal Model

full_file = './Hyundai_TPA/Datasets/ansys/TM/TM.full'
result_file = './Hyundai_TPA/Datasets/ansys/TM/TM.rst'

k, m, dof_reference_table, ndof_per_node, nodes_and_coordinates = read_ansys_full_file(full_file, result_file)

mck_A = MCK_Model(k, m, nodes_and_coordinates)
mck_A.DOFMapping = dof_reference_table

nModes = 100
mModel_A_d = to_modal_model(mck_A, nModes)
mModel_A_d.ndofs = ndof_per_node

zeta = 0.003
add_damping(mModel_A_d, 'modal', zeta)

#%% FRF Measurement - DoF Selection

sensors = fromXls_Sensor("./Hyundai_TPA/meta/TM_fem.xlsx", 'Sensors', 'Channels')
channels = to_channel(sensors)

impacts = fromXls_Impact("./Hyundai_TPA/meta/TM_fem.xlsx", 'Impacts')
refChannels = to_channel(impacts)

virtual_channels = fromXls_VPChannels("./Hyundai_TPA/meta/TM_fem.xlsx", 'VP RefChannels')
#
##%% FRF Synthesis

FreqRange = arange(1,20000)
FRFType = 2 # 0 = Receptance, 1 = mobility, 2 = accelerance

#%% Y_FEM

Y_FEM = to_FRF_matrix(mModel_A_d, channels, refChannels, FreqRange, FRFType)

#%% Plot a FRF function

Freq = arange(0,2000)
ch_i = 1
ch_j = 2

plot(Y_FEM, ch_i, ch_j, Freq, PlotType = 'log')
plot(Y_FEM, ch_i, ch_j, Freq, PlotType = 'phase')

#%% Virtual Point admittance

qA = fromXls_VPChannels("./Hyundai_TPA/meta/TM_fem.xlsx","VP Channels")

IDMu_A = create_IDM_matrix(Y_FEM.Channels, qA)

dofs_i = find(Y_FEM.Channels, Grouping = [1])
dofs_j = find(Y_FEM.RefChannels, Grouping = [1])

#consistency(Y_FEM, IDMu_A, dofs_i, dofs_j, arange(1,2000), Dim = 'Sensor', Type = 'Overall')
consistency(Y_FEM, IDMu_A, dofs_i, dofs_j, arange(0,2000), Dim = 'Sensor', Type = 'Specific')

mA = fromXls_VPChannels("./Hyundai_TPA/meta/TM_fem.xlsx","VP RefChannels")

IDMf_A = create_IDM_matrix(Y_FEM.RefChannels, mA)

consistency(Y_FEM, IDMf_A, dofs_i, dofs_j, arange(0,2000), Dim = 'Hammer', Type = 'Specific')

Yqm_FEM = create_reduced_FRF_matrix(Y_FEM, IDMf_A, IDMu_A)

plot(Yqm_FEM, 2, 2, Freq, PlotType = 'log')
plot(Yqm_FEM, 2, 2, Freq, PlotType = 'phase')

#%% Writing results

with open("./Hyundai_TPA/export/FEM_Mounts/" + Output, 'wb') as output:
    pickle.dump(Yqm_FEM, output, pickle.HIGHEST_PROTOCOL)
