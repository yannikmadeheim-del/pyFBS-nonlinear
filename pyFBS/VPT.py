import pandas as pd
import numpy as np
from pyFBS.IO import Impacts, Channels
from collections import Counter
from scipy.linalg import block_diag
from scipy.linalg import block_diag,svd,norm

class VPT(object):
    """
    Description
    In current form the code assumes two things:
        1. the grouping is the same for VP_RefCh and VP_Ch and each VP has 6DoF
        2. the additional DoF come after VP input/output channels and are not listed in VP channels

    TODO: Remove the assumptions! Add variable DoF of VPT and variable order of input
    TODO: Frequency dependent weighting matrix
    Attributes:

    """
    def __init__(self,excel_file, ch="Channels", refch="Impacts", vp_ch = "VP Channels" ,vp_refch = "VP RefChannels"):

        self.Virtual_RefChannels = None
        self.Virtual_Channels = None

        self.Channels = None
        self.RefChannels = None

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
        #print(mask_u)

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

        Ru = block_diag(*R_all, np.eye(len(np.where(mask_u != 0)[0])))

        Wu = block_diag(*_Warray)
        Wu = block_diag(Wu,np.eye(len(np.where(mask_u != 0)[0])))
        Tu = np.linalg.pinv(Ru.T @ Wu @ Ru) @ Ru.T @ Wu
        Fu = Ru @ Tu

        self.Ru = Ru
        self.Wu = Wu
        self.Tu = Tu
        self.Fu = Fu

    def define_IDM_F(self):
        """
        Function to define the IDM F!
        """
        ov_f, _vps, mask_f = self.find_overlap_ch(self.RefChannels, self.Virtual_RefChannels)
        #print(mask_f)
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

        #print(pos,type,_W)

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
        #print(_group_ch)
        # Get the
        _group_chVP = np.asarray([d.Grouping for d in channelsB])
        #print(_group_chVP)

        _overlap = []
        for a in np.unique(_group_chVP):
            _overlap.append(np.where(_group_ch == a))

        mask = np.isin(_group_ch, np.unique(_group_chVP), invert=True).astype(int)

        return _overlap, np.unique(_group_chVP, return_index=True), mask


    def apply_VPT(self,Y_object,freq_lim = 2000):
        """
        Applies the VPT
        TODO: check about if the channels match
        :param Y_object:
        :return:
        """
        _freq_lim =  Y_object.Data.shape[2]
        _Y_vpt = np.zeros((self.Tu.shape[0], self.Tf.shape[1], _freq_lim), dtype=complex)
        for i in range(_freq_lim):
            _Y_vpt[:, :, i] = self.Tu @ Y_object.Data[:, :, i] @ self.Tf

        self.vptData = _Y_vpt
        self.vptFreqs = Y_object.Freqs[:_freq_lim]

        self.Data = Y_object.Data



    def consistency(self,grouping,ref_grouping):

        def find_group(gr,gr_list):
            """
            returns locations of grouping
            :param gr:
            :param gr_list:
            :return:
            """
            _overlap = []
            for a in np.unique(gr):
                _overlap.append(np.where(gr_list == a))
            return np.asarray(_overlap).reshape(-1)

        _ch_all = np.asarray([d.Grouping for d in self.Channels])
        _chVP_all = np.asarray([d.Grouping for d in self.Virtual_Channels])

        _Rch_all = np.asarray([d.Grouping for d in self.RefChannels])
        _RchVP_all = np.asarray([d.Grouping for d in self.Virtual_RefChannels])

        ind_ch = find_group(grouping,_ch_all)
        #ind_chVP = find_group(grouping,_chVP_all)

        ind_Rch = find_group(ref_grouping, _Rch_all)
        #ind_RchVP = find_group(ref_grouping, _RchVP_all)




        # Sensor consistency
        sub_Y = self.Data[ind_ch, :, :][:, ind_Rch, :]
        sub_Fu = self.Fu[ind_ch,:][:,ind_ch]

        u_f = np.zeros((sub_Y.shape[0],1,sub_Y.shape[2]),dtype = complex)
        u = np.zeros((sub_Y.shape[0],1,sub_Y.shape[2]),dtype = complex)

        for i in range(sub_Y.shape[2]):
            u_f[:,:,i] = sub_Fu @ sub_Y[:,:,i] @ np.ones((sub_Y.shape[1],1))
            u[:, :, i] = sub_Y[:,:,i] @ np.ones((sub_Y.shape[1],1))

        self.u_f = u_f
        self.u = u

        overall_sensor = []
        for i in range(sub_Y.shape[2]):
            overall_sensor.append(norm(u_f[:,:,i])/norm(u[:,:,i]))

        self.overall_sensor = np.asarray(overall_sensor)

        c_u = np.conj(u)
        c_u_f = np.conj(u_f)

        Numerator = (u_f + u) * (c_u_f + c_u)
        Denominator = 2 * (u_f * c_u_f + u * c_u)

        self.specific_sensor = np.asarray(Numerator/Denominator,dtype = float)


        # Impact consistency
        sub_Y = self.Data[ind_ch, :, :][:, ind_Rch, :]
        sub_Ff = self.Ff[ind_Rch,:][:,ind_Rch]

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            y_f[:,:,i] = (np.ones((sub_Y.shape[0],1)).T@sub_Y[:,:,i] @ sub_Ff).T
            y  [:,:,i] = (np.ones((sub_Y.shape[0],1)).T@sub_Y[:,:,i]).T

        self.y_f = y_f
        self.y = y

        overall_impact = []
        for i in range(sub_Y.shape[2]):
            overall_impact.append(norm(y_f[:, :, i]) / norm(y[:, :, i]))

        self.overall_impact = np.asarray(overall_impact)

        c_y = np.conj(y)
        c_y_f = np.conj(y_f)

        Numerator = (y_f + y) * (c_y_f + c_y)
        Denominator = 2 * (y_f * c_y_f + y * c_y)

        self.specific_impact = np.asarray(Numerator / Denominator, dtype=float)


        """
        #print(_gr_ch,_gr_chVP)

        # Measurement quality indicators

        # Sensor consistency

        # Impact consistency
        """




if __name__ == '__main__':
    print("Test: Cat!")
    
