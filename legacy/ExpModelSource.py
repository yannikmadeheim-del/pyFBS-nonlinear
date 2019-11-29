import matplotlib.pyplot as plt
from numpy import arange, zeros, log10
from numpy.linalg import cond, svd
import pickle

from Python_Toolbox.util.Read_UFF_File import read_uff_file

from Python_Toolbox.ui.Sensor.From_Xls import fromXls as fromXls_Sensor
from Python_Toolbox.ui.Impact.From_Xls import fromXls as fromXls_Impact

from Python_Toolbox.FRF.From_Series_to_Matrix import from_series_to_matrix

from Python_Toolbox.Channel.FromXls import fromXls as fromXls_VPChannels
from Python_Toolbox.Channel.Create_IDM_Matrix import create_IDM_matrix

from Python_Toolbox.Channel.Select import find
from Python_Toolbox.FRF.Consistency import consistency

from Python_Toolbox.FRF.Create_Reduced_FRF_Matrix import create_reduced_FRF_matrix

#%% Settings

fmax = 1500

#%%Inputs - Soft Structure

Raw_FRF = "Hyundai-TPA_ComponentBased_FRF_Shaker_on_plate_ManualHammer_SUB_SOURCE_191505.analysis_19_1.uf"
Raw_Coherence = "Hyundai-TPA_ComponentBased_FRF_Shaker_on_plate_ManualHammer_SUB_SOURCE_191505.analysis_3_1.uf"
Excel_File = "ShakerOnPlate_SOURCE.xlsx"

#Outputs

FRF = "YAB_qm_ShakerOnPlate_Source"
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

ch_i = 24
ch_j = 8

plt.figure()
plt.subplot(2,1,1)
plt.plot(20*log10(abs(YAB_uf.Data[ch_i, ch_j, arange(0,fmax+1)])))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (dB)")
plt.grid(which = "both")

plt.figure()
plt.subplot(2,1,2)
plt.plot(YAB_uf.Coherence[ch_i, ch_j, arange(0,fmax+1)])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Coherence (-)")
plt.grid(which = "both")


# %% Experimental Modelling

qA = fromXls_VPChannels("./Hyundai_TPA/meta/" + Excel_File,"VP Channels")
mA = fromXls_VPChannels("./Hyundai_TPA/meta/" + Excel_File,"VP RefChannels")

fA = YAB_uf.RefChannels
IDMf_A = create_IDM_matrix(fA, mA)

dofs_i = find(YAB_uf.Channels, Grouping = [1,2,3])
dofs_j = find(YAB_uf.RefChannels, Grouping = [1,2,3])

consistency(YAB_uf, IDMf_A, dofs_i, dofs_j, Freq = arange(0,fmax+1), Dim = 'Hammer', Type = 'Specific')

uA = YAB_uf.Channels
IDMu_A = create_IDM_matrix(uA, qA)

with open("./Hyundai_TPA/fem/" + "IDMu_A", 'wb') as output:
    pickle.dump(IDMu_A, output, pickle.HIGHEST_PROTOCOL)

YAB_qm = create_reduced_FRF_matrix(YAB_uf, IDMf_A, IDMu_A)

# %% Check CMIF and conditioning

CN = zeros(YAB_qm.nFreq, float)
CMIF = zeros((YAB_qm.nRefChannels, YAB_qm.nFreq), float)
U = zeros((YAB_qm.nChannels, YAB_qm.nRefChannels, YAB_qm.nFreq), complex)

for i in range (0,YAB_qm.nFreq):
    CN[i] = cond(YAB_qm.Data[:,:,i])
    U[:,:,i], CMIF[:,i], _ = svd(YAB_qm.Data[:,:,i])
    
plt.figure()
plt.subplot(2,1,1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("CMIF")
plt.grid(which = "both")

for i in range (0,len(CMIF[:,0])):  
    plt.semilogy(arange(0,fmax+1), CMIF[i,0:fmax+1])
    
plt.subplot(2,1,2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Condition Number")
plt.grid(which = "both")
plt.semilogy(arange(0,fmax+1), CN[0:fmax+1])

plt.figure()
plt.plot(20*log10(U[6,7:12,arange(0,fmax+1)]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("SVD - U Component")
plt.grid(which = "both")

#%% Writing results

with open("./Hyundai_TPA/export/FRF/" + FRF, 'wb') as output:
    pickle.dump(YAB_qm, output, pickle.HIGHEST_PROTOCOL)