import pickle
from numpy import zeros, eye, vstack, hstack, arange, log10
from numpy.linalg import svd, cond, inv
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from seaborn import heatmap
from math import degrees
from cmath import phase

from Python_Toolbox.FRF.FRF import FRF
from Python_Toolbox.FRF.Reciprocity import reciprocity
from Python_Toolbox.Math.Symmetrize import symmetrize
from Python_Toolbox.Math.Smooth import smooth

from Python_Toolbox.FRF.Plot_FRF import plot_FRF as plot

from Python_Toolbox.ui.Sensor.From_Xls import fromXls as fromXls_Sensor
from Python_Toolbox.ui.Impact.From_Xls import fromXls as fromXls_Impact
from Python_Toolbox.ui.Sensor.To_Channel import to_channel

from Python_Toolbox.Channel.FromXls import fromXls as fromXls_VPChannels

from Python_Toolbox.util.Read_UFF_File import read_uff_file
from Python_Toolbox.Freq_Blocks.Freq_Blocks import Freq_Blocks


#%% Reading Files

with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_EM", 'rb') as input:
        Y_EM = pickle.load(input)
        
with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_RM", 'rb') as input:
        Y_RM = pickle.load(input)
        
with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_TM", 'rb') as input:
        Y_TM = pickle.load(input)
        
with open('./Hyundai_TPA/export/FRF/' + "YAB_qm_ShakerOnPlate_Receiver", 'rb') as input:
        Yqm_receiver = pickle.load(input)
        
with open('./Hyundai_TPA/export/FRF/' + "YAB_qm_ShakerOnPlate_Source", 'rb') as input:
        Yqm_source = pickle.load(input)
        
#%% Build block diagonal matrix of the mounts
        
Y_mounts = zeros((18,18,2001), complex)

for i in range (0,2001):
    
    Y_mounts[:,:,i] = block_diag(Y_TM.Data[:,:,i], Y_RM.Data[:,:,i], Y_EM.Data[:,:,i])

# uncorrelated virtual points for the receiver FRF matrix
    
Yqm_receiver.Data[0:6,6:18,:] = 0
Yqm_receiver.Data[6:12,0:6,:] = 0
Yqm_receiver.Data[6:12,12:18,:] = 0
Yqm_receiver.Data[12:18,0:12,:] = 0

# Boolean matrices

B_c_1 = hstack([eye(18), -eye(18), zeros((18,6),int),    zeros((18,18),int)])
B_c_2 = hstack([zeros((18,18),int), eye(18), zeros((18,6),int), -eye(18)])

B_c = vstack([B_c_1, B_c_2])

B_e_1 = hstack([eye(18), -eye(18), zeros((18,18),int)])
B_e_2 = hstack([zeros((18,18),int), -eye(18), eye(18)])

B_e = vstack([B_e_1, B_e_2])

# uncoupled system

YAB_un = zeros((60,54,2001),complex)

for k in range (0,2001):
    
    YAB_un[0:18,0:18,k] = Yqm_source.Data[:,:,k]
    YAB_un[18:42,18:36,k] = Yqm_receiver.Data[:,:,k]
    YAB_un[42:60,36:54,k] = -1.*Y_mounts[:,:,k]
    
# interface flexibility matrix
    
Y_int = zeros((len(B_c[:,0]),len(B_e), 2001),complex)
CN = zeros((2001),float)
CMIF = zeros((len(Y_int[:,0]),2001),float)
    
for k in range (0,2001):
    
    Y_int[:,:,k] = B_c.dot(YAB_un[:,:,k]).dot(B_e.transpose())
    CN[k] = cond(Y_int[:,:,k])
    _, CMIF[:, k], _ = svd(Y_int[:,:,k])

plt.figure()
plt.subplot(2,1,1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("CMIF")
plt.grid(which = "both")

for i in range (0,len(CMIF[:,0])):  
    plt.semilogy(arange(0,2001), CMIF[i,0:2001])
    
plt.subplot(2,1,2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Condition Number")
plt.grid(which = "both")
plt.semilogy(arange(0,2001), CN[0:2001])

# LMFBS

YAB_DS_qm = zeros((60,54,2001),complex)

for k in range (0,2001):
    YAB_DS_qm[:,:,k] = YAB_un[:,:,k] - YAB_un[:,:,k].dot(B_e.transpose()).dot(inv(B_c.dot(YAB_un[:,:,k]).dot(B_e.transpose()))).dot(B_c).dot(YAB_un[:,:,k])
    
# remove redundant DoFs
    
YAB_DS_qm = YAB_DS_qm[18:42,18:36,:]

# create object matrix

YAB_DS_qm_FRFMatrix = FRF();
YAB_DS_qm_FRFMatrix.Data = YAB_DS_qm;
YAB_DS_qm_FRFMatrix.Freq = arange(0,1501)
    
# Save DS Matrix
    
with open("./Hyundai_TPA/export/Validation/YAB_DS_qm_FRFMatrix", 'wb') as output:
    pickle.dump(YAB_DS_qm_FRFMatrix, output, pickle.HIGHEST_PROTOCOL)
    
#%% Symmetrization virtual point admittance
    
# Load predicted response u_tpa 'Fsoft - Ystiff'
    
with open('./Hyundai_TPA/export/Validation/' + "YAB_DS_qm_FRFMatrix", 'rb') as input:
    YAB_DS_qm_FRFMatrix = pickle.load(input)
    
# Reduce YAB_DS_qm only to the VP DoFs

YAB_DS_qm_FRFMatrix.Data = YAB_DS_qm_FRFMatrix.Data[0:18,:,:]

# Give channel info to YAB_DS_qm

with open('./Hyundai_TPA/export/FRF/' + "YAB_qm_ShakerOnPlate_Source", 'rb') as input:
    Yqm_source = pickle.load(input)

# Give channel info to YAB_DS_qm

channels = Yqm_source.Channels
refchannels = Yqm_source.RefChannels
YAB_DS_qm_FRFMatrix.Channels = channels
YAB_DS_qm_FRFMatrix.RefChannels = refchannels

recipA = reciprocity(YAB_DS_qm_FRFMatrix)
Yqm_As = symmetrize(YAB_DS_qm_FRFMatrix)
recipAs = reciprocity(Yqm_As)

heatmap(recipA)
plt.title("Mean Reciprocity")
plt.xlabel("Virtual Forces")
plt.ylabel("Virtual Displacements")

#%% Driving Point Passivity

ch_i = 5
ch_j = ch_i

plot(YAB_DS_qm_FRFMatrix, ch_i, ch_j, arange(0,1501), PlotType = "log")
plot(YAB_DS_qm_FRFMatrix, ch_i, ch_j, arange(0,1501), PlotType = "phase")

#%% 2 FRFs reciprocity

ch_i = 2
ch_j = 5

plt.figure()
plt.subplot(2,1,1)
plt.semilogy(abs(YAB_DS_qm_FRFMatrix.Data[ch_i, ch_j, 0:1501]))
plt.semilogy(abs(YAB_DS_qm_FRFMatrix.Data[ch_j, ch_i, 0:1501]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Admittance in " + YAB_DS_qm_FRFMatrix.Channels[0].Unit.Name)      
plt.grid(which = "both")

plt.subplot(2,1,2)
Data = YAB_DS_qm_FRFMatrix.Data[ch_i, ch_j, 0:1501]
Data = [degrees(phase(d)) for d in Data]
plt.plot(Data)
Data = YAB_DS_qm_FRFMatrix.Data[ch_j, ch_i, 0:1501]
Data = [degrees(phase(d)) for d in Data]
plt.plot(Data)
plt.ylim(-180, +180)
plt.yticks([-180, -90, 0, 90, 180])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase Angle in Degrees")
plt.grid()

#%% Experimental Modelling

# Locate Excel file

sensors = fromXls_Sensor("./Hyundai_TPA/meta/ShakerOnPlate_SoftStruc_StiffRubber_ManualHammer.xlsx", 'Sensors', 'Channels')
impacts = fromXls_Impact("./Hyundai_TPA/meta/ShakerOnPlate_SoftStruc_StiffRubber_ManualHammer.xlsx", 'Impacts')

channels = to_channel(sensors)
refChannels = to_channel(impacts)

virtual_channels = fromXls_VPChannels("./Hyundai_TPA/meta/ShakerOnPlate_SoftStruc_StiffRubber_ManualHammer.xlsx", 'VP RefChannels')

#%% Validation DS and InSitu

# Load predicted response u_tpa 'Fsoft - Ystiff'

with open("./Hyundai_TPA/export/Validation/YAB_DS_qm_FRFMatrix", 'rb') as input:
    YAB_DS_qm_FRFMatrix = pickle.load(input)

# Load operational response u_op_f stiff
    
with open("./Hyundai_TPA/export/Validation/Hyundai-TPA_FRF_Shaker_on_plate_ManualHammer_SOFT_190905.analysis_2_1.FRF.pkl", 'rb') as input:
    YAB_INSITU_qm_FRFMatrix = pickle.load(input)
    
# Channel i of DS 24 is the same channel as INSITU 33
# Response_DS: TM [0:5], RM [6:11], EM [12:17], Target1 [18:20], Target2 [21:23]
# NOTE: Do not compare channel [0:17]
    
ch_i_DS = 18
ch_i_INSITU = ch_i_DS + 9

# Excitation: TM [0:5], RM [6:11], EM [12:17]
ch_j_DS = 0
ch_j_INSITU = ch_j_DS

sm = 10
fmax = 1500

plt.figure()
plt.semilogy(YAB_DS_qm_FRFMatrix.Freq[0:fmax+1],smooth(abs(YAB_DS_qm_FRFMatrix.Data[ch_i_DS, ch_j_DS, 0:fmax+1])))
plt.semilogy(YAB_INSITU_qm_FRFMatrix.Freqs[0:fmax+1],abs(YAB_INSITU_qm_FRFMatrix.Data[ch_i_INSITU, ch_j_INSITU, 0:fmax+1]))

plt.title("Validation - Channel " + str(ch_i_INSITU))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Accelerance (m/s^2/N)")
plt.legend(["DS", "INSITU"])

#%% Component Based TPA

# FFT of operational impact in assembly

PAK_Raw = read_uff_file("./Hyundai_TPA/Datasets/pak/Component_Based/Hyundai-TPA_ComponentBased_Operational_Impacts_Shaker_on_plate_ManualHammer_SOFT.analysis_5_1.uf")

Data = zeros((7,9,6401),complex)
counter = 0

for j in range (0,9):
    for i in range (0,7):
        Data[i,j,:] = PAK_Raw[i+counter].DataSets.Y_Channels.Data
    counter = counter + 7

u_f = Freq_Blocks()
u_f.Data = Data[:,:,0:1501]
u_f.Freq = arange(0,1501)

plt.figure()
plt.plot(20*log10(abs(u_f.Data[6,8,0:1501])))
plt.title("APS Pulse")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Force (dB)")

plt.figure()
plt.plot(20*log10(abs(u_f.Data[2,8,0:1501])))
plt.title("FFT Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (dB)")

# FFT of operational impact through component-based TPA

# loading operational FFT responses on the source component

del PAK_Raw

PAK_Raw = read_uff_file("./Hyundai_TPA/Datasets/pak/Component_Based/Hyundai-TPA_ComponentBased_Operational_Impacts_Shaker_on_plate_ManualHammer_SUB_SOURCE.analysis_7_1.uf")

Data2 = zeros((28,9,12801),complex)
counter = 0

for j in range (0,9):
    for i in range (0,28):
        Data2[i,j,:] = PAK_Raw[i+counter].DataSets.Y_Channels.Data
    counter = counter + 28
    
u_f2 = Freq_Blocks()
u_f2.Data = Data2[:,:,0:1501]
u_f2.Freq = arange(0,1501)

plt.figure()
plt.plot(20*log10(abs(u_f2.Data[27,8,0:1501])))
plt.title("APS Pulse")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Force (dB)")

plt.figure()
plt.plot(20*log10(abs(u_f2.Data[2,8,0:1501])))
plt.title("FFT Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (dB)")
#
with open("./Hyundai_TPA/fem/IDMu_A", 'rb') as input:
    IDMu = pickle.load(input)
    
# transforming the operational responses from measured to virtual domain
    
q_f2 = zeros((len(IDMu.T), len(u_f2.Data[0,:,:]), 1501), complex)
    
for i in range (0,1501):
    q_f2[:,:,i] = IDMu.T.dot(u_f2.Data[:27, :, i])
    
# loading virtual FRFs of numerical mounts

with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_EM", 'rb') as input:
        Y_EM = pickle.load(input)
        
with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_RM", 'rb') as input:
        Y_RM = pickle.load(input)
        
with open('./Hyundai_TPA/export/FEM_Mounts/' + "Yqm_TM", 'rb') as input:
        Y_TM = pickle.load(input)
        
Y_mounts = zeros((18,18,2001), complex)

for i in range (0,2001):
    
    Y_mounts[:,:,i] = block_diag(Y_TM.Data[:,:,i], Y_RM.Data[:,:,i], Y_EM.Data[:,:,i])
    
with open('./Hyundai_TPA/export/FRF/' + "YAB_qm_ShakerOnPlate_Receiver", 'rb') as input:
        Yqm_receiver = pickle.load(input)
        
with open('./Hyundai_TPA/export/FRF/' + "YAB_qm_ShakerOnPlate_Source", 'rb') as input:
        Yqm_source = pickle.load(input)
        
# uncorrelated virtual points for the receiver FRF matrix
    
Yqm_receiver.Data[0:6,6:18,:] = 0
Yqm_receiver.Data[6:12,0:6,:] = 0
Yqm_receiver.Data[6:12,12:18,:] = 0
Yqm_receiver.Data[12:18,0:12,:] = 0

# Manual decoupling for receiver component

B_c = hstack([eye(18), zeros((18,6),int), -eye(18)])
B_e = hstack([eye(18), -eye(18)])

#uncoupled system

YAB_un = zeros((42,36,2001),complex)

for k in range (0,2001):
    YAB_un[0:24,0:18,k] = Yqm_receiver.Data[:,:,k]
    YAB_un[24:42,18:36,k] = -1*Y_mounts[:,:,k]
    
# Interface flexibility matrix
    
Y_int = zeros((len(B_c[:,0]),len(B_e), 2001),complex)
CN = zeros((2001),float)
CMIF = zeros((len(Y_int[:,0]),2001),float)
    
for k in range (0,2001):
    
    Y_int[:,:,k] = B_c.dot(YAB_un[:,:,k]).dot(B_e.transpose())
    CN[k] = cond(Y_int[:,:,k])
    _, CMIF[:, k], _ = svd(Y_int[:,:,k])
    
plt.figure()
plt.subplot(2,1,1)
plt.xlabel("Frequency (Hz)")
plt.ylabel("CMIF")
plt.grid(which = "both")

for i in range (0,len(CMIF[:,0])):  
    plt.semilogy(arange(0,2001), CMIF[i,0:2001])
    
plt.subplot(2,1,2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Condition Number")
plt.grid(which = "both")
plt.semilogy(arange(0,2001), CN[0:2001])

# LMFBS for obtaining receiver structure without mounts

YAB_DS_receiveronly_qm_temp = zeros((42,36,2001),complex)

for k in range (0,2001):
    X = inv(B_c.dot(YAB_un[:,:,k]).dot(B_e.transpose())).dot(B_c.dot(YAB_un[:,:,k]))
    YAB_DS_receiveronly_qm_temp[:,:,k] = YAB_un[:,:,k] - YAB_un[:,:,k].dot(B_e.transpose()).dot(X)
    
# remove redundant DoFs
    
YAB_DS_receiveronly_qm = YAB_DS_receiveronly_qm_temp[0:24,0:18,:] 

# TPA component-based procedure

u_f2_TPA = zeros((24,9,1501),complex)

for k in range (0,1501):
    u_f2_TPA[:,:,k] = YAB_DS_receiveronly_qm[:,:,k].dot(inv(Yqm_source.Data[:,:,k] + YAB_DS_receiveronly_qm[0:18,:,k]).dot(q_f2[:,:,k]))
    
u_tpa = u_f2_TPA

u_op_f = u_f 

#%%  Plot Component-based TPA

# channels of prediction
# Response_DS: TM [1:6], RM [7:12], EM [13:18], Target1 [19:21], Target2 [22:24]
# NOTE: Do not compare channel [1:18]
ch_i_tpa = 18
# OP_Impact X [1:3], Y [4:6], Z [7:9]
ch_j_tpa = 0

# channels of validation
# Here u_f is a 7x9x1501 matrix
#--> [2 targetsensors *3 + 1 impact, 3 impacts in each dir. * 3 averages, freq]
# Target1 [1:3], Target2 [4:6]
ch_i_op = 0
# OP_Impact X [1:3], Y [4:6], Z [7:9]
ch_j_op = ch_j_tpa


plt.figure()
plt.semilogy(abs(u_tpa[ch_i_tpa, ch_j_tpa, 1:1501]))
plt.semilogy(abs(u_op_f.Data[ch_i_op, ch_j_op, 0:1501]), linestyle = "dashed")
plt.title("Valdiation: Component Based TPA - TargetSensor 1: +X / Impact: +X")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend(["Prediction: u^{TPA}","Validation: u^{OP}_{3}"])
plt.grid()