import numpy as np
from scipy.linalg import norm
from ..utility import coh_frf, cmif
from .vpt import VPT


class SVT(object):
    """
    Singular Vector Transformation - enables transformation of measured
    responses to a singular DoFs. The left and right singular vectors are used
    as a reduction basis. Care should be taken that the assumption of
    collocation is taken into account.

    :param ch: A DataFrame containing information on channels (i.e. outputs)
    :type ch: pd.DataFrame
    :param refch: A DataFrame containing information on reference channels (i.e. inputs)
    :type refch: pd.DataFrame
    :param freq: Frequency vector
    :type freq: array(float)
    :param frf: A matrix of Frequency Response Functions FRFs [f,out,in].
    :type frf: array(float)
    :param grouping_no: Grouping number where SVT is defined.
    :type grouping_no: int
    :param no_svs: Number of singular DoFs.
    :type no_svs: int
    :param Wu: Displacement weigting matrix
    :type Wu: array(float), optional
    :param Wf: Force weighting matrix
    :type Wf: array(float), optional
    """

    def __init__(
        self, ch, refch, freq, frf, grouping_no, no_svs, wu=None, wf=None
    ):
        # Load the physical input-output DoFs
        self.channels = ch
        self.ref_channels = refch
        self.group_no = grouping_no
        self.no_svs = no_svs

        # Load the FRF matrix
        self.freq = freq
        self.frf = frf

        # Define the IDM_U and IDM_F matrix
        ind_chn = VPT.find_group(
            self.group_no, self.channels.Grouping.to_numpy()
        )
        ind_imp = VPT.find_group(
            self.group_no, self.ref_channels.Grouping.to_numpy()
        )

        # Define a FRF subset
        sub_frf = self.frf[:, ind_chn, :][:, :, ind_imp]

        # Calculate left and right singular vectors with
        self.u, self.s, self.v = cmif(sub_frf, return_svector=True)

        # Without weighting matrixes
        # self.Tu = np.transpose(np.conj(U[:, :, :self.no_svs]), (0, 2, 1))
        # self.Tf = V[:, :, :self.no_svs]

        # Define reduction matrixes
        Ru = self.u[:, :, : self.no_svs]
        if wu == None:
            wu = np.identity(len(ind_chn))

        Rf = self.v[:, :, : self.no_svs]
        if wf == None:
            wf = np.identity(len(ind_imp))

        self.ru = Ru
        self.rf = Rf

        # SV transormation matrixes
        self.tu = np.linalg.pinv(self.h(Ru) @ wu @ Ru) @ self.h(Ru) @ wu
        self.tf = wf @ Rf @ np.linalg.pinv(self.h(Rf) @ wf @ Rf)

        # Back projection
        self.fu = Ru @ self.tu
        self.ff = Rf @ self.h(self.tf)

    def h(self, frf):
        """
        Performs a Hermition transpose on an FRF matrix.

        :param frf: A FRF matrix.
        :type frf: array(float)
        :return: Hermitian transpose of an FRF matrix
        """
        return np.conj(np.transpose(frf, (0, 2, 1)))

    def apply_svt(self, ch, refch, freq, frf):
        """
        Applies the singular vector transformation on an FRF matrix

        :param ch: A DataFrame containing information on channels
            (i.e. outputs)
        :type ch: pd.DataFrame
        :param refch: A DataFrame containing information on reference channels
            (i.e. inputs)
        :type refch: pd.DataFrame
        :param freq: Frequency vector
        :type freq: array(float)
        :param frf: A matrix of Frequency Response Functions FRFs [f,out,in].
        :type frf: array(float)
        """

        # Find an overlap between
        ind_chn = VPT.find_group(self.group_no, ch.Grouping.to_numpy())
        ind_imp = VPT.find_group(self.group_no, refch.Grouping.to_numpy())

        # Find non-overlapping section
        overlap_chn = np.setxor1d(ind_chn, range(ch.shape[0]))
        overlap_imp = np.setxor1d(ind_imp, range(refch.shape[0]))

        # Define global transformation matrixes
        TU_g = np.zeros(
            (len(freq), self.no_svs + len(overlap_chn), ch.shape[0]),
            dtype=complex,
        )
        TF_g = np.zeros(
            (len(freq), refch.shape[0], self.no_svs + len(overlap_imp)),
            dtype=complex,
        )

        # Insert data in global matrixes
        for i, _ind in enumerate(ind_chn):
            TU_g[:, : self.no_svs, _ind] = self.tu[:, :, i]

        for i, _ind in enumerate(ind_imp):
            TF_g[:, _ind, : self.no_svs] = self.tf[:, i, :]

        # not-transformed channels append
        i = np.identity(len(overlap_chn))
        I_chn = np.transpose(np.dstack([i] * len(freq)), (2, 0, 1))

        i = np.identity(len(overlap_imp))
        I_imp = np.transpose(np.dstack([i] * len(freq)), (2, 0, 1))

        for i, _ind in enumerate(overlap_chn):
            TU_g[:, self.no_svs :, _ind] = I_chn[:, :, i]

        for i, _ind in enumerate(overlap_imp):
            TF_g[:, _ind, self.no_svs :] = I_imp[:, :, i]

        # Apply the SVT on FRF matrix
        tran_frf = TU_g @ frf @ TF_g

        return TU_g, TF_g, tran_frf

    def consistency(self, grouping, frf):
        """
        Evaluates SVT consistency indicators based on the supplied grouping number.

        :param grouping: Grouping number
        :type grouping: int
        :param frf: A matrix of Frequency Response Functions FRFs [f,out,in].
        :type frf: array(float)
        """

        # get all groupings from the vpt
        _ch_all = self.channels.Grouping.to_numpy()
        _Rch_all = self.ref_channels.Grouping.to_numpy()

        # extract the grouping mask
        ind_ch = VPT.find_group(grouping, _ch_all)
        ind_Rch = VPT.find_group(grouping, _Rch_all)

        sub_Y = np.transpose(frf, (1, 2, 0))[ind_ch, :, :][:, ind_Rch, :]
        sub_Fu = self.fu

        u_f = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)
        u = np.zeros((sub_Y.shape[0], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            u_f[:, :, i] = (
                sub_Fu[i, :, :] @ sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))
            )
            # initial response
            u[:, :, i] = sub_Y[:, :, i] @ np.ones((sub_Y.shape[1], 1))

        self.u_f = u_f[:, 0, :]
        self.u = u[:, 0, :]

        # Calculate overall sensor consistency indicator
        self.overall_sensor = norm(self.u_f, axis=0) / norm(self.u, axis=0)

        # Calculate specific sensor consistency indicator
        specific_sensor = []
        for i in range(self.u.shape[0]):
            specific_sensor.append(coh_frf(self.u_f[i, :], self.u[i, :]))

        self.specific_sensor = np.asarray(specific_sensor)

        # Calculate impact consistency
        sub_Y = np.transpose(frf, (1, 2, 0))[:, ind_Rch, :]
        sub_Ff = self.ff

        y_f = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)
        y = np.zeros((sub_Y.shape[1], 1, sub_Y.shape[2]), dtype=complex)

        for i in range(sub_Y.shape[2]):
            # filtered response
            y_f[:, :, i] = (
                np.ones((sub_Y.shape[0], 1)).T
                @ sub_Y[:, :, i]
                @ sub_Ff[i, :, :]
            ).T
            # initial response
            y[:, :, i] = (np.ones((sub_Y.shape[0], 1)).T @ sub_Y[:, :, i]).T

        self.y_f = y_f[:, 0, :]
        self.y = y[:, 0, :]

        # Calculate overall impact consistency indicator
        self.overall_impact = norm(self.y_f, axis=0) / norm(self.y, axis=0)

        # Calculate specific impact consistency indicator
        specific_impact = []
        for i in range(self.y.shape[0]):
            specific_impact.append(coh_frf(self.y_f[i, :], self.y[i, :]))

        self.specific_impact = np.asarray(specific_impact)
