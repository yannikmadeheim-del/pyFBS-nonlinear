%load_ext autoreload
%autoreload 2

#%%

import matplotlib.pyplot as plt
from numpy import arange, log10
import pickle

from Python_Toolbox.util.Read_UFF_File import read_uff_file

from Python_Toolbox.ui.Sensor.From_Xls import fromXls as fromXls_Sensor
from Python_Toolbox.ui.Impact.From_Xls import fromXls as fromXls_Impact

from Python_Toolbox.FRF.From_Series_to_Matrix import from_series_to_matrix

from Python_Toolbox.FRF.Plot_FRF import plot_FRF as plot

from Python_Toolbox.Channel.FromXls import fromXls as fromXls_VPChannels
from Python_Toolbox.Channel.Select import select
from Python_Toolbox.Channel.Create_IDM_Matrix import create_IDM_matrix

from Python_Toolbox.Channel.Select import find
from Python_Toolbox.FRF.Consistency import consistency

from Python_Toolbox.FRF.Create_Reduced_FRF_Matrix import create_reduced_FRF_matrix


#%% Settings


fmax = 1500

#%%Inputs - Soft Structure

Raw_FRF = "Hyundai-TPA_ComponentBased_FRF_ManualHammer_SUB_SOFTRECEIVERandTS_191605.analysis_17_1.uf"
Raw_Coherence = "Hyundai-TPA_ComponentBased_FRF_ManualHammer_SUB_SOFTRECEIVERandTS_191605.analysis_2_1.uf"
Excel_File = "ShakerOnPlate_RECEIVER.xlsx"

#Outputs

FRF = "YAB_qm_ShakerOnPlate_Receiver"
#%% SoftStruc - Load, build and fill FRF matrices

YAB_PAK_rawData = read_uff_file('./Hyundai_TPA/Datasets/pak/Component_Based/' + Raw_FRF)

sensors = fromXls_Sensor("./Hyundai_TPA/meta/" + Excel_File,"Sensors", "Channels")
impacts = fromXls_Impact("./Hyundai_TPA/meta/" + Excel_File,"Impacts")

YAB_uf = from_series_to_matrix(YAB_PAK_rawData, sensors, impacts)

#%% Coherence

Coherence_PAK_rawData = read_uff_file("./Hyundai_TPA/Datasets/pak/Component_Based/" + Raw_Coherence)

sensors = fromXls_Sensor("./Hyundai_TPA/meta/" + Excel_File,"Sensors", "Channels")
impacts = fromXls_Impact("./Hyundai_TPA/meta/" + Excel_File,"Impacts")

Coherence_uf = from_series_to_matrix(Coherence_PAK_rawData, sensors, impacts)

YAB_uf.Coherence = Coherence_uf.Data

#%% Plotting FRFs

ch_i = 32
ch_j = 8

plt.figure()
plt.subplot(2,1,1)
plt.plot(20*log10(YAB_uf.Data[ch_i, ch_j, arange(0,fmax+1)]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (dB)")
plt.grid(which = "both")

plt.figure()
plt.subplot(2,1,2)
plt.plot(YAB_uf.Coherence[ch_i, ch_j, arange(0,fmax+1)])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence (-)")
plt.grid(which = "both")

plt.show()

# %% Experimental Modelling

qA = fromXls_VPChannels("./Hyundai_TPA/meta/" + Excel_File,"VP Channels")
mA = fromXls_VPChannels("./Hyundai_TPA/meta/" + Excel_File,"VP RefChannels")

fA = YAB_uf.RefChannels
IDMf_A = create_IDM_matrix(fA, mA)
 
# Implement rigidify to select only one grouping. Right now consistency checks can only be made for all DOFs.

dofs_i = find(YAB_uf.Channels, Grouping = [1,2,3])
dofs_j = find(YAB_uf.RefChannels, Grouping = [1,2,3])

consistency(YAB_uf, IDMf_A, dofs_i, dofs_j, Freq = arange(0,fmax+1), Dim = 'Hammer', Type = 'Specific')


print(dofs_i)
print(dofs_j)



uA = YAB_uf.Channels
uB_extra = select(uA, Grouping = [10])

IDMu_A = create_IDM_matrix(uA, qA, uB_extra)
#consistency(YAB_uf, IDMu_A, dofs_i, dofs_j, Freq = arange(0,fmax+1), Dim = 'Sensor', Type = 'Overall')
plt.show()


#%%
YAB_qm = create_reduced_FRF_matrix(YAB_uf, IDMf_A, IDMu_A)




dof_i = 0
dof_j = 0

plt.figure()
plot(YAB_qm, dof_i, dof_j)
plt.xlim(0,2000)
plt.show()


#%% Writing results

with open("./Hyundai_TPA/export/FRF/" + FRF, 'wb') as output:
    pickle.dump(YAB_qm, output, pickle.HIGHEST_PROTOCOL)