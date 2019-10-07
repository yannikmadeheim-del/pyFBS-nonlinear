import pandas as pd
import numpy as np
from pyfbs.io import Impacts, Channels
from collections import Counter
from scipy.linalg import block_diag
from scipy.linalg import block_diag,svd,norm
from tqdm import tqdm

class VPTfreq(object):
    """

    """
    def __init__(self,excel_file,Y, ch="Channels", refch="Impacts", vp_ch = "VP Channels" ,vp_refch = "VP RefChannels"):

        self.Virtual_RefChannels = None
        self.Virtual_Channels = None

        self.Channels = None
        self.RefChannels = None

        #
        self.Y = Y

        #Edit names
        self.Virtual_RefChannels = Impacts(excel_file, vp_refch).Data
        self.Virtual_Channels = Impacts(excel_file, vp_ch).Data

        self.Channels = Impacts(excel_file, ch).Data
        self.RefChannels = Impacts(excel_file, refch).Data

        #Define the
        self.define_IDM_U()
        self.define_IDM_F()

        #Could be defined like that think about it
        #class IDM_U: pass
        #self.IDM_U = IDM_U

    def define_IDM_U(self):
        """
        Function to define the IDM U!
        """
        ov_u, _vps, mask_u = self.find_overlap_ch(self.Channels, self.Virtual_Channels)
        R_all = []
        # iterates through all unique virtual points (through grouping)
        _Warray = []
        for i in range(len(ov_u)):

            # gets the unique VP position
            _posVP = np.asarray(self.Virtual_Channels[_vps[1][i]].Position)

            # gets the current positions
            ov_c = ov_u[i]
            r = np.zeros((len(ov_c[0]), 6))

            for i, ch in enumerate(ov_c[0]):
                _pos = np.asarray(self.Channels[ch].Position)
                _dir = np.asarray([self.Channels[ch].Direction])
                _group = self.Channels[ch].Grouping
                _type = self.Channels[ch].Quantity.Name
                r[i, :] = _dir @ self.R_matrix_U(_posVP - _pos,type =_type)

                _Warray.append(self.W_rotational(_pos, _dir, type = _type))


            R_all.append(r)

        Ru = block_diag(*R_all)
        Ru = block_diag(Ru,np.eye(len(np.where(mask_u != 0)[0])))
        self.Ru = Ru
        #print(Ru.shape)

        Wu = block_diag(*_Warray)
        Wu = block_diag(Wu,np.eye(len(np.where(mask_u != 0)[0])))
        self.Wu = Wu

        # Wu_f is a 4D numpy array for each input set you use where -j refers to the freq input W[i,:,:,j]
        # n_imp... number of impacts
        # n_out... number of outputs
        # n_freq.. number of freqs

        n_out = self.Y.Data.shape[0]
        n_imp = self.Y.Data.shape[1]
        n_freq= self.Y.Data.shape[2]

        Wu_f = np.zeros((n_imp, n_out , n_freq))
        Tu_f = np.zeros((n_imp, self.Ru.shape[1] , n_out , n_freq))
        #Fu_f = np.zeros((n_imp, n_out , n_out , 1000))

        import math

        def sigmoid(x, slope=0.01, f=800):
            return 1 / (1 + np.exp(-slope * (x - f)))

        for _f in tqdm(range(n_freq)):
            for _i in range(n_imp):

                _tW = block_diag(*np.abs(self.Y.Coherence[:,_i,_f]))**2
                #print(_tW.shape)
                #for _d,diag_val in enumerate(np.diag(_tW)):
                #    if _d in [9,10,11,21,22,23,33,34,35]:
                #        _tW[_d,_d] = 0#sigmoid(self.Y.Freqs[_f])

                W_sum = Wu + _tW
                Wu_f[_i,:,_f] = np.diag(W_sum)

                Tu = np.linalg.pinv(Ru.T @ W_sum @ Ru) @ Ru.T @ W_sum
                #Fu = Ru @ Tu
                Tu_f[_i,:,:,_f] = Tu
                #Fu_f[_i, :, :, _f] = Fu

        self.Tu = Tu
        self.Wu_f = Wu_f

        self.Tu_f = Tu_f
        #elf.Fu_f = Fu_f

    def define_IDM_F(self):
        """
        Function to define the IDM F!
        """
        ov_f, _vps, mask_f = self.find_overlap_ch(self.RefChannels, self.Virtual_RefChannels)
        R_all = []
        # iterates through all unique virtual points (through grouping)
        for i in range(len(ov_f)):
            # gets the unique VP position
            _posVP = np.asarray(self.Virtual_Channels[_vps[1][i]].Position)

            # gets the current positions
            ov_c = ov_f[i]
            r = np.zeros((len(ov_c[0]), 6))

            for i, ch in enumerate(ov_c[0]):
                _pos = np.asarray(self.RefChannels[ch].Position)
                _dir = np.asarray([self.RefChannels[ch].Direction])
                _group = self.RefChannels[ch].Grouping

                r[i, :] = (self.R_matrix_F(_posVP - _pos) @ (_dir.T)).reshape(-1)


            R_all.append(r)

        Rf = block_diag(*R_all, np.eye(len(np.where(mask_f != 0)[0])))

        Wf = np.eye(np.max(Rf.shape))
        Tf = Wf @ Rf @ np.linalg.pinv(Rf.T @ Wf @ Rf)
        Ff = Rf @ Tf.T

        self.Rf = Rf
        self.Wf = Wf
        self.Tf = Tf
        self.Ff = Ff



    def R_matrix_U(self,pos,type = "Acceleration"):
        """
        Return the R matrix according to the position.
        """
        rx, ry, rz = pos[0], pos[1], pos[2]

        if type == "Acceleration":
            _R = np.asarray([[1, 0, 0, 0, rz, -ry],
                               [0, 1, 0, -rz, 0, rx],
                               [0, 0, 1, ry, -rx, 0]])

        elif type == "Angular Acceleration":
            _R = np.asarray([[0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

        return _R

    def W_rotational(self,pos,dir,type = "Angular Acceleration"):
        rx, ry, rz = pos[0], pos[1], pos[2]

        _W = 1

        if type == "Angular Acceleration":
            c = np.where(np.asarray(dir) != 0)[1][0]
            if c == 0:
                _W = np.sqrt(rz ** 2 + ry ** 2) ** 2
            elif c == 1:
                _W = np.sqrt(rz ** 2 + rx ** 2) ** 2
            elif c == 2:
                _W = np.sqrt(ry ** 2 + rx ** 2) ** 2
            #_W = 0
        return _W


    def R_matrix_F(self,pos):
        """
        Return the R matrix according to the position.
        """
        rx, ry, rz = pos[0], pos[1], pos[2]

        _R = np.asarray([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, -rz, ry],
                           [rz, 0, -rx],
                           [ry, -rx, 0]])

        return _R

    def find_overlap_ch(self,channelsA, channelsB):
        """
        Finds an overlap of grouping through two
        :param channelsA:
        :param channelsB:
        :return:
        """

        # Get the grouping
        _group_ch = np.asarray([d.Grouping for d in channelsA])

        # Get the
        _group_chVP = np.asarray([d.Grouping for d in channelsB])

        _overlap = []
        for a in np.unique(_group_chVP):
            _overlap.append(np.where(_group_ch == a))

        mask = np.isin(_group_ch, np.unique(_group_chVP), invert=True).astype(int)

        return _overlap, np.unique(_group_chVP, return_index=True), mask


    def apply_VPT(self,freq_lim = 2000):
        """
        Applies the VPT
        TODO: check about if the channels match
        :param Y_object:
        :return:
        """
        Y_object = self.Y
        _freq_lim =  Y_object.Data.shape[2]
        #_Y_vpt = np.zeros((self.Tu.shape[0], self.Tf.shape[1], _freq_lim), dtype=complex)

        _Y_vpt = np.zeros((self.Tu.shape[0], self.Tf.shape[1], _freq_lim), dtype=complex)


        n_out = self.Y.Data.shape[0]
        n_imp = self.Y.Data.shape[1]
        n_freq = self.Y.Data.shape[2]


        for _f in tqdm(range(_freq_lim)):
            _tempY = np.zeros((self.Ru.shape[1],n_imp),dtype = complex)
            for _i in range(n_imp):                #Tu_f[_i,:,:,_f] = Tu

                _tempY[:,_i] = self.Tu_f[_i,:,:,_f] @ Y_object.Data[:, _i, _f]

            _Y_vpt[:,:,_f] = _tempY @ self.Tf

        self.vptData = _Y_vpt
        self.vptFreqs = self.Y.Freqs




        #for i in range(_freq_lim):
        #    _Y_vpt[:, :, i] = self.Tu @ Y_object.Data[:, :, i] @ self.Tf

        #self.vptData = _Y_vpt
        #self.vptFreqs = Y_object.Freqs[:_freq_lim]

        #self.Data = Y_object.Data



if __name__ == '__main__':
    print("Test: Cat!")

