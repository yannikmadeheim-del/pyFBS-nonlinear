import pandas as pd
import numpy as np
from scipy.linalg import block_diag, svd, norm
from pyFBS.utility import coh_frf

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

    def __init__(self, excel_file, ch="Channels", refch="Impacts", vp_ch="VP Channels", vp_refch="VP RefChannels",Wu = None, Wf = None):

        self.Virtual_RefChannels = None
        self.Virtual_Channels = None

        self.Channels = None
        self.RefChannels = None

        # Edit names
        self.Virtual_RefChannels = pd.read_excel(excel_file, sheet_name=vp_refch)
        self.Virtual_Channels = pd.read_excel(excel_file, sheet_name=vp_ch)

        self.Channels = pd.read_excel(excel_file, sheet_name=ch)
        self.RefChannels = pd.read_excel(excel_file, sheet_name=refch)

        # Define the
        self.define_IDM_U()
        self.define_IDM_F()

        self.Wu_p = Wu
        self.Wf_p = Wf


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
            _posVP = np.asarray(self.Virtual_Channels.iloc[i][["Position_1","Position_2","Position_3"]].to_numpy())

            # gets the current positions
            ov_c = ov_u[i]
            r = np.zeros((len(ov_c[0]), 6))

            for j, ch in enumerate(ov_c[0]):
                _pos = np.asarray(self.Channels.iloc[ch][["Position_1","Position_2","Position_3"]].to_numpy()).astype(float)
                _dir = np.asarray(self.Channels.iloc[ch][["Direction_1","Direction_2","Direction_3"]].to_numpy()).astype(float)
                #print(ch)
                _group = self.Channels.iloc[ch]["Grouping"]
                _type = self.Channels.iloc[ch]["Quantity"]


                r[j, :] = _dir @ self.R_matrix_U(_posVP - _pos, type=_type)

                _Warray.append(self.W_rotational(_pos, _dir, type=_type))

            R_all.append(r)

        Ru = block_diag(*R_all, np.eye(len(np.where(mask_u != 0)[0])))

        R_n = np.zeros_like(Ru)

        R_all = np.asarray(R_all)[0, :, :]
        gg = 0
        trig = True
        for i, k in enumerate(mask_u):
            if k == 1:
                R_n[i, gg] = 1
                gg += k
            else:
                if trig:
                    R_n[i:i + R_all.shape[0], gg:gg + R_all.shape[1]] = R_all

                    gg += R_all.shape[1]
                    trig = False

        Ru = R_n

        if self.Wu_p == None:
            Wu = block_diag(*_Warray)
            Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))
        else:
            Wu = block_diag(*self.Wu_p)
            Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))


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
        # print(mask_f)
        R_all = []
        # iterates through all unique virtual points (through grouping)
        for i in range(len(ov_f)):
            # gets the unique VP position
            _posVP = np.asarray(self.Virtual_RefChannels.iloc[i][["Position_1","Position_2","Position_3"]].to_numpy())

            # gets the current positions
            ov_c = ov_f[i]
            r = np.zeros((len(ov_c[0]), 6))

            for j, ch in enumerate(ov_c[0]):
                _pos = np.asarray(self.RefChannels.iloc[ch][["Position_1", "Position_2", "Position_3"]].to_numpy()).astype(float)
                _dir = np.asarray(self.RefChannels.iloc[ch][["Direction_1", "Direction_2", "Direction_3"]].to_numpy()).astype(float)

                _group = self.RefChannels.iloc[ch]["Grouping"]
                _type = self.RefChannels.iloc[ch]["Quantity"]


                r[j, :] = (self.R_matrix_F(_posVP - _pos) @ (_dir.T)).reshape(-1)

            R_all.append(r)

        # print(np.where(mask_f != 0))
        # R_n = np.eye(len(np.where(mask_f != 0)[0]))
        # print(R_n)

        Rf = block_diag(*R_all, np.eye(len(np.where(mask_f != 0)[0])))

        R_n = np.zeros_like(Rf)

        R_all = np.asarray(R_all)[0, :, :]
        gg = 0
        trig = True
        for i, k in enumerate(mask_f):
            if k == 1:
                R_n[i, gg] = 1
                gg += k
            else:
                if trig:
                    R_n[i:i + R_all.shape[0], gg:gg + R_all.shape[1]] = R_all

                    gg += R_all.shape[1]
                    trig = False
        Rf = R_n

        if self.Wf_p == None:
            Wf = np.eye(np.max(Rf.shape))
        else:
            Wf = self.Wf_p

        Tf = Wf @ Rf @ np.linalg.pinv(Rf.T @ Wf @ Rf)
        Ff = Rf @ Tf.T

        self.Rf = Rf
        self.Wf = Wf
        self.Tf = Tf
        self.Ff = Ff

    def R_matrix_U(self, pos, type="Acceleration"):
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

    def W_rotational(self, pos, dir, type="Angular Acceleration"):
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

        # print(pos,type,_W)

        return _W

    def R_matrix_F(self, pos):
        """
        Return the R matrix according to the position.
        """
        rx, ry, rz = pos[0], pos[1], pos[2]

        _R = np.asarray([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, -rz, ry],
                         [rz, 0, -rx],
                         [-ry, rx, 0]])

        return _R

    def find_overlap_ch(self, channelsA, channelsB):
        """
        Finds an overlap of grouping through two
        :param channelsA:
        :param channelsB:
        :return:
        """

        # Get the grouping
        _group_ch = channelsA.Grouping.to_numpy()
        # print(_group_ch)
        # Get the
        _group_chVP = channelsB.Grouping.to_numpy()
        # print(_group_chVP)

        _overlap = []
        for a in np.unique(_group_chVP):
            _overlap.append(np.where(_group_ch == a))

        mask = np.isin(_group_ch, np.unique(_group_chVP), invert=True).astype(int)

        return _overlap, np.unique(_group_chVP, return_index=True), mask

    def apply_VPT(self, freq,FRF):
        """
        Applies the VPT
        :param Y_object:
        :return:
        """
        _freq_lim = FRF.shape[2]
        _Y_vpt = np.zeros((self.Tu.shape[0], self.Tf.shape[1], _freq_lim), dtype=complex)
        for i in range(_freq_lim):
            _Y_vpt[:, :, i] = self.Tu @ FRF[:, :, i] @ self.Tf

        self.vptData = _Y_vpt
        self.vptFreqs = freq
        self.FRF = FRF

    def find_group(self,gr, gr_list):
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

    def consistency(self, grouping, ref_grouping):

        _ch_all = self.Channels.Grouping.to_numpy()
        _chVP_all = self.Virtual_Channels.Grouping.to_numpy()

        _Rch_all = self.RefChannels.Grouping.to_numpy()
        _RchVP_all = self.Virtual_RefChannels.Grouping.to_numpy()

        ind_ch = self.find_group(grouping, _ch_all)
        ind_Rch = self.find_group(ref_grouping, _Rch_all)

        # Sensor consistency
        sub_Y = self.FRF[ind_ch, :, :][:, ind_Rch, :]
        sub_Fu = self.Fu[ind_ch, :][:, ind_ch]

        u_f = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)
        u = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            u_f[:, :, i] = sub_Fu @ sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))
            u[:, :, i] = sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))

        self.u_f = u_f[:,0,:]
        self.u = u[:,0,:]

        self.overall_sensor = norm(self.u_f,axis = 0) / norm(self.u,axis = 0)

        specific_sensor = []
        for i in range(self.u.shape[0]):
            specific_sensor.append(coh_frf(self.u_f[i, :], self.u[i, :]))

        self.specific_sensor = np.asarray(specific_sensor)


        # Impact consistency
        sub_Y = self.FRF[ind_ch, :, :][:, ind_Rch, :]
        sub_Ff = self.Ff[ind_Rch, :][:, ind_Rch]

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            y_f[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i] @ sub_Ff).T
            y[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i]).T

        self.y_f = y_f[:,0,:]
        self.y = y[:,0,:]


        self.overall_impact = norm(self.y_f,axis = 0) / norm(self.y,axis = 0)


        specific_impact = []
        for i in range(self.y.shape[0]):
            specific_impact.append(coh_frf(self.y_f[i,:],self.y[i,:]))

        self.specific_impact = np.asarray(specific_impact)

    """
    Frequency-dependend weighting matrix
    
    Wu = block_diag(*_Warray)
    Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))
    self.Wu = Wu

    # Wu_f is a 4D numpy array for each input set you use where -j refers to the freq input W[i,:,:,j]
    # n_imp... number of impacts
    # n_out... number of outputs
    # n_freq.. number of freqs

    n_out = self.Y.Data.shape[0]
    n_imp = self.Y.Data.shape[1]
    n_freq = self.Y.Data.shape[2]

    Wu_f = np.zeros((n_imp, n_out, n_freq))
    Tu_f = np.zeros((n_imp, self.Ru.shape[1], n_out, n_freq))
    # Fu_f = np.zeros((n_imp, n_out , n_out , 1000))

    for _f in tqdm(range(n_freq)):
        for _i in range(n_imp):
            _tW = block_diag(*np.abs(self.Y.Coherence[:, _i, _f])) ** 2
            # print(_tW.shape)
            # for _d,diag_val in enumerate(np.diag(_tW)):
            #    if _d in [9,10,11,21,22,23,33,34,35]:
            #        _tW[_d,_d] = 0#sigmoid(self.Y.Freqs[_f])

            W_sum = Wu + _tW
            Wu_f[_i, :, _f] = np.diag(W_sum)

            Tu = np.linalg.pinv(Ru.T @ W_sum @ Ru) @ Ru.T @ W_sum
            # Fu = Ru @ Tu
            Tu_f[_i, :, :, _f] = Tu
            # Fu_f[_i, :, :, _f] = Fu

    self.Tu = Tu
    self.Wu_f = Wu_f

    self.Tu_f = Tu_f
    self.Fu_f = Fu_f
    """

if __name__ == '__main__':
    print("Test: Cat!")

