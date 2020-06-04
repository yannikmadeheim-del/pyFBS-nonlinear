import numpy as np
from scipy.linalg import block_diag, svd, norm
from pyFBS.utility import coh_frf

class VPT(object):
    """
    Virtual point transformation - Description

    :param ch: Channels used in the transformation (outputs - sensors).
    :type ch: pandas.DataFrame
    :param refch: Reference channels used in the transformation (input - impacts).
    :type refch: pandas.DataFrame
    :param vp_ch: Virtual point channels used in the transformation (output - VP).
    :type vp_ch: pandas.DataFrame
    :param vp_refch: Virtual point reference channels used in the transformation (input - VP).
    :type vp_refch: pandas.DataFrame
    :param Wu: Displacement weighting matrix.
    :type Wu: numpy.array
    :param Wf: Displacement weighting matrix.
    :type Wf: numpy.array
    """
    def __init__(self, ch, refch, vp_ch, vp_refch ,Wu = None, Wf = None):
        # Load the physical input-output DoFs
        self.Channels = ch
        self.RefChannels = refch

        # Load virtual input-output DoFs
        self.Virtual_Channels = vp_ch
        self.Virtual_RefChannels = vp_refch

        # Load Weighting matrices if None, no weighting is applied in the transformation
        self.Wu_p = Wu
        self.Wf_p = Wf

        # Define the IDM_U and IDM_F matrix
        self.define_IDM_U()
        self.define_IDM_F()


    def define_IDM_U(self):
        """
        Calculates the Ru, Tu, Fu matrices based on the supplied position and orientation of Channels and Virtual Channels. In the current implementation only Rigid IDMs are supported.
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
                _group = self.Channels.iloc[ch]["Grouping"]
                _type = self.Channels.iloc[ch]["Quantity"]
                r[j, :] = _dir @ self.R_matrix_U(_posVP - _pos, type=_type)
                _Warray.append(self.W_rotational(_pos, _dir, type=_type))
            R_all.append(r)

        Ru = block_diag(*R_all, np.eye(len(np.where(mask_u != 0)[0])))

        R_n = np.zeros_like(Ru)

        # position the transformation matrix based on location it the .xlsx file
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

        # definition of weighting matrix
        if self.Wu_p == None:
            Wu = block_diag(*_Warray)
            Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))
        else:
            Wu = block_diag(*self.Wu_p)
            Wu = block_diag(Wu, np.eye(len(np.where(mask_u != 0)[0])))


        # calculate the Tu, Fu matrices
        Tu = np.linalg.pinv(Ru.T @ Wu @ Ru) @ Ru.T @ Wu
        Fu = Ru @ Tu

        self.Ru = Ru
        self.Wu = Wu
        self.Tu = Tu
        self.Fu = Fu

    def define_IDM_F(self):
        """
        Calculates the Rf, Tf, Ff matrices based on the supplied position and orientation of Channels and Virtual Channels. In the current implementation only Rigid IDMs are supported.
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

        # position the transformation matrix based on location it the .xlsx file
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

        # definition of weighting matrix
        if self.Wf_p == None:
            Wf = np.eye(np.max(Rf.shape))
        else:
            Wf = self.Wf_p

        # calculate the Tf, Ff matrices
        Tf = Wf @ Rf @ np.linalg.pinv(Rf.T @ Wf @ Rf)
        Ff = Rf @ Tf.T

        self.Rf = Rf
        self.Wf = Wf
        self.Tf = Tf
        self.Ff = Ff

    @staticmethod
    def R_matrix_U(pos, type="Acceleration"):
        """
        Calculate Ru matrix based on the channel position/orientation and sensor type.

        :param pos: Position of the channel.
        :type pos: numpy.array
        :param type: Type of the channel (i.e. Acceleration or Angular Acceleration).
        :returns: numpy.array, Ru
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

    @staticmethod
    def W_rotational(pos, dir, type="Angular Acceleration"):
        """
        Defines the weighting matrix based on the

        :param pos: Position of the channel.
        :type pos: numpy.array
        :param type: Type of the channel (i.e. Acceleration or Angular Acceleration)
        :type type: str
        """
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

        return _W

    @staticmethod
    def R_matrix_F(pos):
        """
        Calculate Rf matrix based on the reference channel position/orientation.

        :param pos: Position of the reference channel relative to virtual point.
        :type pos: numpy.array
        :returns: numpy.array, Rf
        """
        rx, ry, rz = pos[0], pos[1], pos[2]

        _R = np.asarray([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1],
                         [0, -rz, ry],
                         [rz, 0, -rx],
                         [-ry, rx, 0]])

        return _R

    @staticmethod
    def find_overlap_ch(channelsA, channelsB):
        """
        Finds an overlap of grouping number between two channel datasets.

        :param channelsA:
        :type channelsA: pandas.DataFrame
        :param channelsB:
        :type channelsB: pandas.DataFrame
        :return GG: numpy.array, Overlap mask
        """

        # Get the grouping numbers from DataFrames
        _group_ch = channelsA.Grouping.to_numpy()
        _group_chVP = channelsB.Grouping.to_numpy()

        # Find overlap between the two channel datasets
        _overlap = []
        for a in np.unique(_group_chVP):
            _overlap.append(np.where(_group_ch == a))

        mask = np.isin(_group_ch, np.unique(_group_chVP), invert=True).astype(int)

        return _overlap, np.unique(_group_chVP, return_index=True), mask

    @staticmethod
    def find_group(gr, gr_list):
        """
        Get a grouping overlap between two sets.

        :param gr: blah
        :type gr: list
        :param gr_list: blah
        :type gr_list: list
        :return: numpy.array
        """
        _overlap = []
        for a in np.unique(gr):
            _overlap.append(np.where(gr_list == a))
        return np.asarray(_overlap).reshape(-1)


    def apply_VPT(self, freq,FRF):
        """
        Applies the Virtual Point Transformation on the supplied FRF matrix.

        :param freq: Frequency vector.
        :type freq: numpy.array
        :param FRF: A matrix of Frequency Response Functions FRFs.
        :type: numpy.array
        """
        #_freq_lim = FRF.shape[2]
        #_Y_vpt = np.zeros((self.Tu.shape[0], self.Tf.shape[1], _freq_lim), dtype=complex)
        #for i in range(_freq_lim):
        #    _Y_vpt[:, :, i] = self.Tu @ FRF[:, :, i] @ self.Tf

        _Y_vpt = self.Tu @ FRF @ self.Tf

        self.vptData = _Y_vpt
        self.freq = freq
        self.FRF = FRF

    def consistency(self, grouping, ref_grouping):
        """
        Calculates the VP consistency indicators based on the supplied grouping numbers.

        :param grouping: Grouping number of the VP.
        :type grouping: float
        :param ref_grouping: Grouping number of the reference VP.
        :type ref_grouping: float
        """

        # get all groupings from the vpt
        _ch_all = self.Channels.Grouping.to_numpy()
        _chVP_all = self.Virtual_Channels.Grouping.to_numpy()

        _Rch_all = self.RefChannels.Grouping.to_numpy()
        _RchVP_all = self.Virtual_RefChannels.Grouping.to_numpy()

        # extract the grouping mask
        ind_ch = self.find_group(grouping, _ch_all)
        ind_Rch = self.find_group(ref_grouping, _Rch_all)

        # Calculate sensor consistency
        sub_Y = np.transpose(self.FRF,(1,2,0))[ind_ch, :, :][:, ind_Rch, :]
        sub_Fu = self.Fu[ind_ch, :][:, ind_ch]

        u_f = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)
        u = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            u_f[:, :, i] = sub_Fu @ sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))
            # initial response
            u[:, :, i] = sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))

        self.u_f = u_f[:,0,:]
        self.u = u[:,0,:]

        # Calculate overall sensor consistency indicator
        self.overall_sensor = norm(self.u_f,axis = 0) / norm(self.u,axis = 0)


        # Calculate specific sensor consistency indicator
        specific_sensor = []
        for i in range(self.u.shape[0]):
            specific_sensor.append(coh_frf(self.u_f[i, :], self.u[i, :]))

        self.specific_sensor = np.asarray(specific_sensor)


        # Calculate impact consistency
        sub_Y = np.transpose(self.FRF,(1,2,0))[:, ind_Rch, :]
        sub_Ff = self.Ff[ind_Rch, :][:, ind_Rch]

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            y_f[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i] @ sub_Ff).T
            # initial response
            y[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i]).T

        self.y_f = y_f[:,0,:]
        self.y = y[:,0,:]

        # Calculate overall impact consistency indicator
        self.overall_impact = norm(self.y_f,axis = 0) / norm(self.y,axis = 0)

        # Calculate specific impact consistency indicator
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

