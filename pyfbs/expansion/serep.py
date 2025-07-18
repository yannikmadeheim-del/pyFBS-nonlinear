from .tools import find_locations_in_data_frames
import numpy as np


def serep(eig_vec_num, eig_vec_exp, df_chn_num, df_chn_exp):
    """
    This function performs SEREP - and expanssion method in modal domain.

    :param eig_vec_num: Eigenvectors of the numerical model.
    :type eig_vec_num: array(float)
    :param eig_vec_exp: Eigenvectors of the experimental model.
    :type eig_vec_exp: array(float)
    :param df_chn_num: Response locations and directions for the numerical
        model.
    :type df_chn_num: pandas.DataFrame
    :param df_chn_exp: Response locations and directions for the experimental
        model.
    :type df_chn_exp: pandas.DataFrame

    :return: hybrid eigenvectors
    :rtype: array(float)
    """

    # Initialization data
    eig_vec_num = np.asarray(np.copy(eig_vec_num)).astype(complex)
    eig_vec_exp = np.asarray(np.copy(eig_vec_exp)).astype(complex)

    # Input data validation
    if df_chn_num.shape[0] != eig_vec_num.shape[0]:
        raise Exception(
            'Numerical model - Incompatible channel data and eigenvector '
            'shape.'
        )
    if df_chn_exp.shape[0] != eig_vec_exp.shape[0]:
        raise Exception(
            'Experimental model - Incompatible channel data and eigenvector '
            'shape.'
        )

    # Internal and boundary DoF partition
    maching_locations_chn = find_locations_in_data_frames(
        df_chn_num, df_chn_exp
    )
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception(
            'Not all experimental channel locations have a matching location '
            'in the numerical channel dataframe.'
        )

    b_dof_ind_num = np.copy(maching_locations_chn[:, 0])
    b_dof_ind_exp = np.copy(maching_locations_chn[:, 1])
    i_dof_ind_num = np.setdiff1d(
        np.arange(eig_vec_num.shape[0]), b_dof_ind_num
    )

    # T matrix generation
    psi_num_ir = np.copy(eig_vec_num)[i_dof_ind_num, :]
    psi_num_br = np.copy(eig_vec_num)[b_dof_ind_num, :]
    psi_exp_br = np.copy(eig_vec_exp)[b_dof_ind_exp, :]

    if psi_num_br.shape[0] < psi_num_br.shape[1]:
        print('Underdetermined: n_b < m_par')

    print('Condition number:' + "%.2f" % np.linalg.cond(psi_num_br))
    # T = np.vstack([
    #     psi_num_ir @ np.linalg.pinv(psi_num_br), np.eye(psi_num_br.shape[0])
    # ])
    T = np.vstack(
        [
            psi_num_ir @ np.linalg.pinv(psi_num_br),
            psi_num_br @ np.linalg.pinv(psi_num_br),
        ]
    )
    eig_vec_serep = T @ psi_exp_br

    # Reordering - input dof
    reord_ind = np.argsort(np.hstack([i_dof_ind_num, b_dof_ind_num]))

    return eig_vec_serep[reord_ind, :]
