import numpy as np
import copy
import matplotlib.pyplot as plt
from pyFBS.utility import *



def find_locations_in_data_frames(df_1, df_2):
    """Find matching locations of data frames ``df_1`` and ``df_2``.

    :param df_1: Data frame 1
    :type df_1: pandas.DataFrame
    :param df_2: Data frame 2
    :type df_2: pandas.DataFrame
    :return: Vector of matching locations of both data frames.
    :rtype: array(float)
    """
    df_1_val = df_1[["Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3"]].values
    df_2_val = df_2[["Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3"]].values

    # to prevent numerical errors
    df_1_val = np.round(df_1_val, 6) 
    df_2_val = np.round(df_2_val, 6)

    return np.array(np.all((df_1_val[:, None, :] == df_2_val[None, :, :]), axis=-1).nonzero()).T


def SEMM(Y_num, Y_exp, df_chn_num, df_imp_num, df_chn_exp, df_imp_exp, SEMM_type='fully-extend', red_comp=0, red_eq=0):
    """
    This function performs SEMM. It couples numerical (``Y_num``) and experimental (``Y_exp``) model to hybrid model. 

    :param Y_num: Numerical response matrix
    :type Y_num: array(float)
    :param Y_exp: Experimental response matrix
    :type Y_exp: array(float)
    :param df_chn_num: Locations and directions of response in the ``Y_num``
    :type df_chn_num: pandas.DataFrame
    :param df_imp_num: Locations and directions of excitation in the ``Y_num``
    :type df_imp_num: pandas.DataFrame
    :param df_chn_exp: Locations and directions of response in the ``Y_exp``
    :type df_chn_exp: pandas.DataFrame
    :param df_imp_exp: Locations and directions of excitation in the ``Y_exp``
    :type df_imp_exp: pandas.DataFrame
    :param SEMM_type: Defined which type of SEMM will be performed - basic ("basic") or fully extended ("fully-extend") or fully extended with SVD truncation on compatibility or equilibrium ("fully-extend-svd")
    :type SEMM_type: str("basic" or "fully-extend" or "fully-extend-svd")
    :param red_comp: Defines how many maximum singular values will not be taken into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken into account in ensuring equilibrium conditions
    :type red_eq: int
    :return: Hybrid model based on numerical and experimental data
    :rtype: array(float)

    The form of the FRFs in the numerical matrix must match the ``df_chn_num`` and ``df_imp_num`` parameters. 
    The ``df_chn_num`` parameter represents the rows (responses) of the numeric matrix, and the ``df_imp_num`` parameter represents the columns (excitaions) that are presented in the numerical model.
    The same guidelines must also be followed for the experimental model, the corresponding response locationsare defined in the parameter ``df_chn_exp`` and the excitation locations in the parameter ``df_imp_exp``.

    The location and direction of an individual response point in the experimental model must coincide exactly with one location and direction of the response and in the numerical model. 
    The same must be true also for the location and direction of excitation.
    """

    # Validation of input data
    Y_num = np.asarray(Y_num)
    if len(Y_num.shape) != 3:
        raise Exception('Wrong shape of input numerical receptance matrix.')

    if len(Y_exp.shape) != 3:
        raise Exception('Input experimental matrx must be 3D matrix.')

    if df_chn_exp.shape[0] != Y_exp.shape[1]:
        raise Exception('The input channel data frame must contain those DoFs that are represented in the experimental model.')

    if df_imp_exp.shape[0] != Y_exp.shape[2]:
        raise Exception('The input impact data frame must contain those DoFs that are represented in the experimental model.')

    if df_chn_num.shape[0] != Y_num.shape[1]:
        raise Exception('The input channel data frame must contain those DoFs that are represented in the numerical model.')

    if df_imp_num.shape[0] != Y_num.shape[2]:
        raise Exception('The input impact data frame must contain those DoFs that are represented in the numerical model.')

    # Initialization data
    Y_par = np.asarray(np.copy(Y_num))
    Y_exp = np.asarray(Y_exp)
    Y_exp = np.copy(Y_exp).reshape(Y_exp.shape[0], Y_exp.shape[1]*Y_exp.shape[2])

    # Data preparation for building parent, remowed and overlay model
    # Reviewing all experimental obtained DoFs
    maching_locations_chn = find_locations_in_data_frames(df_chn_num, df_chn_exp)
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception('Not all locations in the channel data frame have their exact locations in the numeric channel data frame.')

    maching_locations_imp = find_locations_in_data_frames(df_imp_num, df_imp_exp)
    if maching_locations_imp.shape[0] != df_imp_exp.shape[0]:
        raise Exception('Not all locations in the impact data frame have their exact locations in the numeric impact data frame.')

    all_resp_nodes_DoF = np.repeat(maching_locations_chn[:, 0], len(maching_locations_imp[:, 0]))
    all_exc_nodes_DoF = np.array(list(maching_locations_imp[:, 0])*len(maching_locations_chn[:, 0]))

    # Define unique locations of performed excitations and responses
    uniq_resp_nodes_DoF = np.unique(all_resp_nodes_DoF)
    uniq_exc_nodes_DoF = np.unique(all_exc_nodes_DoF)

    # Define unique locations of performed excitations and responses started counting from 0
    all_resp_nodes_DoF_0 = [list(uniq_resp_nodes_DoF).index(i) for i in all_resp_nodes_DoF]
    all_exc_nodes_DoF_0 = [list(uniq_exc_nodes_DoF).index(i) for i in all_exc_nodes_DoF]

    # Construction of parent model
    # moved collumns
    _all_exc_nodes_DoF = Y_par[:, :, (uniq_exc_nodes_DoF)]
    Y_par = np.delete(Y_par, uniq_exc_nodes_DoF, axis=2)
    Y_par = np.concatenate((Y_par, _all_exc_nodes_DoF), axis=2)

    # moved rows
    _all_resp_nodes_DoF = Y_par[:, (uniq_resp_nodes_DoF), :]
    Y_par = np.delete(Y_par, uniq_resp_nodes_DoF, axis=1)
    Y_par = np.concatenate((Y_par, _all_resp_nodes_DoF), axis=1)

    # Construction of removed model
    Y_rem = _all_exc_nodes_DoF[:, (uniq_resp_nodes_DoF), :]

    # Construction of overlay model
    Y_ov = np.zeros((Y_par.shape[0], len(uniq_resp_nodes_DoF), len(uniq_exc_nodes_DoF)), dtype=complex)
    for i in range(Y_exp.shape[1]):
        Y_ov[:, all_resp_nodes_DoF_0[i], all_exc_nodes_DoF_0[i]] = Y_exp[:Y_par.shape[0], i]

    if SEMM_type == "basic":
        # Single-line method SEMM - basic form - eq(21)
        try:
            Y_SEMM = Y_par-Y_par[:, :, -len(uniq_exc_nodes_DoF):]@np.linalg.inv(Y_rem)@(
                Y_rem-Y_ov)@np.linalg.inv(Y_rem)@Y_par[:, -len(uniq_resp_nodes_DoF):, :]
        except np.linalg.LinAlgError:
            Y_SEMM = Y_par-Y_par[:, :, -len(uniq_exc_nodes_DoF):]@np.linalg.pinv(Y_rem)@(
                Y_rem-Y_ov)@np.linalg.pinv(Y_rem)@Y_par[:, -len(uniq_resp_nodes_DoF):, :]

    elif SEMM_type == "fully-extend":
        # Single-line method SEMM - fully-extend form - eq(31)
        Y_SEMM = Y_par-Y_par@np.linalg.pinv(Y_par[:, -len(uniq_resp_nodes_DoF):, :])@(
            Y_rem-Y_ov)@np.linalg.pinv(Y_par[:, :, -len(uniq_exc_nodes_DoF):])@Y_par

    elif SEMM_type == "fully-extend-svd":
        Y_SEMM = Y_par - Y_par @ np.linalg.pinv(TSVD(Y_par[:, -len(uniq_resp_nodes_DoF):, :], reduction=red_comp))  @ (
            Y_rem - Y_ov) @ np.linalg.pinv(TSVD(Y_par[:, :, -len(uniq_exc_nodes_DoF):], reduction=red_eq)) @ Y_par

    # rearranging SEMM model to input numerical form od DOFs
    # moved collumns
    _all_exc_nodes_DoF = Y_SEMM[:, :, -len(uniq_exc_nodes_DoF):]
    Y_SEMM = Y_SEMM[:, :, :-len(uniq_exc_nodes_DoF)]

    for index, i in enumerate(uniq_exc_nodes_DoF):
        Y_SEMM = np.insert(Y_SEMM, i, _all_exc_nodes_DoF[:, :, index], axis=2)

    # moved rows
    _all_resp_nodes_DoF = Y_SEMM[:, -len(uniq_resp_nodes_DoF):, :]
    Y_SEMM = Y_SEMM[:, :-len(uniq_resp_nodes_DoF), :]

    for index, i in enumerate(uniq_resp_nodes_DoF):
        Y_SEMM = np.insert(Y_SEMM, i, _all_resp_nodes_DoF[:, index, :], axis=1)

    return Y_SEMM
