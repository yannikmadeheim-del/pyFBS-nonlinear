from ..utility import tpinv, coh_frf
from .tools import find_locations_in_data_frames
import numpy as np
from tqdm import tqdm


def semm(
    y_num,
    y_exp,
    df_chn_num,
    df_imp_num,
    df_chn_exp,
    df_imp_exp,
    semm_type='fully-extend',
    red_comp=0,
    red_eq=0,
    additional_columns=[],
):
    """
    This function performs SEMM. It couples numerical (``y_num``) and
    experimental (``y_exp``) model to hybrid model.

    :param y_num: Numerical response matrix
    :type y_num: array(float)
    :param y_exp: Experimental response matrix
    :type y_exp: array(float)
    :param df_chn_num: Locations and directions of response in the ``y_num``
    :type df_chn_num: pandas.DataFrame
    :param df_imp_num: Locations and directions of excitation in the ``y_num``
    :type df_imp_num: pandas.DataFrame
    :param df_chn_exp: Locations and directions of response in the ``y_exp``
    :type df_chn_exp: pandas.DataFrame
    :param df_imp_exp: Locations and directions of excitation in the ``y_exp``
    :type df_imp_exp: pandas.DataFrame
    :param semm_type: Defined which type of SEMM will be performed - basic
        ("basic") or fully extended ("fully-extend") or fully extended with
        SVD truncation on compatibility or equilibrium ("fully-extend-svd")
    :type semm_type: str("basic" or "fully-extend" or "fully-extend-svd")
    :param red_comp: Defines how many maximum singular values will not be
        taken into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken
        into account in ensuring equilibrium conditions
    :type red_eq: int
    :param additional_columns: Aditional columns to check for maching between
        numerical and experimental model in defined data frames:
            ``df_chn_num``, ``df_imp_num``, ``df_chn_exp``, ``df_imp_exp``
    :type additional_columns: list
    :return: Hybrid model based on numerical and experimental data
    :rtype: array(float)

    The form of the FRFs in the numerical matrix must match the ``df_chn_num``
    and ``df_imp_num`` parameters.
    The ``df_chn_num`` parameter represents the rows (responses) of the numeric
    matrix, and the ``df_imp_num`` parameter represents the columns (excitaions)
    that are presented in the numerical model.
    The same guidelines must also be followed for the experimental model, the
    corresponding response locationsare defined in the parameter ``df_chn_exp``
    and the excitation locations in the parameter ``df_imp_exp``.

    The location and direction of an individual response point in the
    experimental model must coincide exactly with one location and direction
    of the response and in the numerical model. The same must be true also for
    the location and direction of excitation.
    """

    # Validation of input data
    y_num = np.asarray(y_num)
    if len(y_num.shape) != 3:
        raise Exception('Wrong shape of input numerical receptance matrix.')

    if len(y_exp.shape) != 3:
        raise Exception('Input experimental matrx must be 3D matrix.')

    if df_chn_exp.shape[0] != y_exp.shape[1]:
        raise Exception(
            'The input channel data frame must contain those DoFs that are '
            'represented in the experimental model.'
        )

    if df_imp_exp.shape[0] != y_exp.shape[2]:
        raise Exception(
            'The input impact data frame must contain those DoFs that are '
            'represented in the experimental model.'
        )

    if df_chn_num.shape[0] != y_num.shape[1]:
        raise Exception(
            'The input channel data frame must contain those DoFs that are '
            'represented in the numerical model.'
        )

    if df_imp_num.shape[0] != y_num.shape[2]:
        raise Exception(
            'The input impact data frame must contain those DoFs that are '
            'represented in the numerical model.'
        )

    # Initialization data
    y_num = np.asarray(np.copy(y_num)).astype(complex)
    y_exp = np.asarray(np.copy(y_exp)).astype(complex)

    # Data preparation for building parent, remowed and overlay model
    # Reviewing all experimental obtained DoFs
    maching_locations_chn = find_locations_in_data_frames(
        df_chn_num, df_chn_exp, additional_columns
    )
    if maching_locations_chn.shape[0] != df_chn_exp.shape[0]:
        raise Exception(
            'Not all locations in the channel data frame have their exact '
            'locations in the numeric channel data frame.'
        )

    maching_locations_imp = find_locations_in_data_frames(
        df_imp_num, df_imp_exp, additional_columns
    )
    if maching_locations_imp.shape[0] != df_imp_exp.shape[0]:
        raise Exception(
            'Not all locations in the impact data frame have their exact '
            'locations in the numeric impact data frame.'
        )

    chn_b_dof_ind_num = np.copy(maching_locations_chn[:, 0])
    chn_b_dof_ind_exp = np.copy(maching_locations_chn[:, 1])
    chn_i_dof_ind_num = np.setdiff1d(
        np.arange(y_num.shape[1]), chn_b_dof_ind_num
    )

    chn_n_b = chn_b_dof_ind_num.shape[0]

    imp_b_dof_ind_num = np.copy(maching_locations_imp[:, 0])
    imp_b_dof_ind_exp = np.copy(maching_locations_imp[:, 1])
    imp_i_dof_ind_num = np.setdiff1d(
        np.arange(y_num.shape[2]), imp_b_dof_ind_num
    )

    imp_n_b = imp_b_dof_ind_num.shape[0]

    # Parent model
    y_par = y_num[
        :,
        np.hstack([chn_i_dof_ind_num, chn_b_dof_ind_num])[:, np.newaxis],
        np.hstack([imp_i_dof_ind_num, imp_b_dof_ind_num]),
    ]
    # Removed model
    y_rem = y_num[:, chn_b_dof_ind_num[:, np.newaxis], imp_b_dof_ind_num]
    # Overlay model
    y_ov = y_exp[:, chn_b_dof_ind_exp[:, np.newaxis], imp_b_dof_ind_exp]

    if semm_type == "basic":
        # Single-line method SEMM - basic form - eq(21)
        y_semm = (
            y_par
            - y_par[:, :, -imp_n_b:]
            @ np.linalg.inv(y_rem)
            @ (y_rem - y_ov)
            @ np.linalg.pinv(y_rem)
            @ y_par[:, -chn_n_b:, :]
        )

    elif semm_type == "fully-extend" or "extended":
        # Single-line method SEMM - fully-extend form - eq(31)
        y_semm = (
            y_par
            - y_par
            @ np.linalg.pinv(y_par[:, -chn_n_b:, :])
            @ (y_rem - y_ov)
            @ np.linalg.pinv(y_par[:, :, -imp_n_b:])
            @ y_par
        )

    elif semm_type == "fully-extend-svd":
        y_semm = (
            y_par
            - y_par
            @ tpinv(y_par[:, -chn_n_b:, :], trunc=red_comp)
            @ (y_rem - y_ov)
            @ tpinv(y_par[:, :, -imp_n_b:], trunc=red_eq)
            @ y_par
        )

    # reordering
    chn_ind = np.argsort(np.hstack([chn_i_dof_ind_num, chn_b_dof_ind_num]))
    imp_ind = np.argsort(np.hstack([imp_i_dof_ind_num, imp_b_dof_ind_num]))

    return y_semm[:, chn_ind[:, np.newaxis], imp_ind]


def identification_algorithm(
    y_num,
    y_exp,
    df_num_chn,
    df_num_imp,
    df_exp_chn,
    df_exp_imp,
    axis=1,
    semm_type='fully-extend',
    red_comp=0,
    red_eq=0,
    additional_columns=[],
):
    """
    This function computes the coherence criterion for the identification of
    inconsistent measurements. The algorithm is based on the SEMM method.

    :param y_num: Numerical response matrix
    :type y_num: array(float)
    :param y_exp: Experimental response matrix
    :type y_exp: array(float)
    :param df_chn_num: Locations and directions of response in the ``y_num``
    :type df_chn_num: pandas.DataFrame
    :param df_imp_num: Locations and directions of excitation in the ``y_num``
    :type df_imp_num: pandas.DataFrame
    :param df_chn_exp: Locations and directions of response in the ``y_exp``
    :type df_chn_exp: pandas.DataFrame
    :param df_imp_exp: Locations and directions of excitation in the ``y_exp``
    :type df_imp_exp: pandas.DataFrame
    :param axis: Axis of eliminating measurements, 0 or 1
    :type axis: int
    :param semm_type: Defined which type of SEMM will be performed - basic
        ("basic") or fully extended ("fully-extend") or fully extended with
        SVD truncation on compatibility or equilibrium ("fully-extend-svd")
    :type semm_type: str("basic" or "fully-extend" or "fully-extend-svd")
    :param red_comp: Defines how many maximum singular values will not be taken
        into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken
        into account in ensuring equilibrium conditions
    :type red_eq: int
    :param additional_columns: Aditional columns to check for maching between
        numerical and experimental model in defined data frames:
        ``df_chn_num``, ``df_imp_num``, ``df_chn_exp``, ``df_imp_exp``
    :type additional_columns: list
    :return: Hybrid model based on numerical and experimental data
    :rtype: array(float)
    """
    all_exp_chn = np.arange(df_exp_chn.shape[0])
    all_exp_imp = np.arange(df_exp_imp.shape[0])

    sel_freq = np.arange(0, np.min([y_num.shape[0], y_exp.shape[0]]), 1)

    rconstructd_frf = np.zeros_like(y_exp, dtype='complex')

    if axis == 0:
        for i in tqdm(all_exp_chn):
            sel_chn = np.delete(all_exp_chn, i, axis=0)
            sel_imp = all_exp_imp

            chn_index = find_locations_in_data_frames(
                df_num_chn, df_exp_chn.iloc[[i]]
            )
            analsyed_chn = chn_index[:, 0]
            recast_chn = chn_index[:, 1]
            imp_index = find_locations_in_data_frames(
                df_num_imp, df_exp_imp.iloc[sel_imp]
            )
            analsyed_imp = imp_index[:, 0]
            recast_imp = imp_index[:, 1]

            rconstructd_frf[:, i, recast_imp] = semm(
                y_num,
                y_exp[np.ix_(sel_freq, sel_chn, sel_imp)],
                df_num_chn,
                df_num_imp,
                df_exp_chn.iloc[sel_chn],
                df_exp_imp.iloc[sel_imp],
                semm_type,
                red_comp,
                red_eq,
                additional_columns,
            )[:, analsyed_chn, analsyed_imp]
    elif axis == 1:
        for i in tqdm(all_exp_imp):
            sel_chn = all_exp_chn
            sel_imp = np.delete(all_exp_imp, i, axis=0)

            chn_index = find_locations_in_data_frames(
                df_num_chn, df_exp_chn.iloc[sel_chn]
            )
            analsyed_chn = chn_index[:, 0]
            recast_chn = chn_index[:, 1]
            imp_index = find_locations_in_data_frames(
                df_num_imp, df_exp_imp.iloc[[i]]
            )
            analsyed_imp = imp_index[:, 0]
            recast_imp = imp_index[:, 1]

            rconstructd_frf[:, recast_chn, i] = semm(
                y_num[sel_freq, :, :],
                y_exp[np.ix_(sel_freq, sel_chn, sel_imp)],
                df_num_chn,
                df_num_imp,
                df_exp_chn.iloc[sel_chn],
                df_exp_imp.iloc[sel_imp],
                semm_type,
                red_comp,
                red_eq,
                additional_columns,
            )[:, analsyed_chn, analsyed_imp]

    coh = coh_frf(y_exp, rconstructd_frf, return_average=False)
    return rconstructd_frf, coh
