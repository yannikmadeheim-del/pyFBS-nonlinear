import numpy as np
import copy
import matplotlib.pyplot as plt


def loc_of_exp(overlay, DoF):
    """
    Data preparation to find locations of corresponding experimental measurement in the numerical model.

    :param overlay: Defines connections between numerical and experimental model
    :type overlay: list(int)
    :param DoF: Number of DoFs per one node
    :type DoF: int

    :returns: tuple(``all_resp_nodes_DoF``, ``all_exc_nodes_DoF``, ``all_meas``)

        |  ``all_resp_nodes_DoF`` - Presents all locations of observed responses regarding numerical response matrix
        |  ``all_exc_nodes_DoF`` - Presents all locations of performed excitations regarding numerical response matrix 
        |  ``all_meas`` - Presents the order of the measurements used according to the overlay parameter
    :rtype: array(int), array(int), array(int)
    """
    all_resp_nodes_DoF = []
    all_exc_nodes_DoF = []
    all_meas = []
    for i in overlay:
        all_resp_nodes_DoF.append((i[0][0]-1)*DoF+i[1][0]-1)
        all_exc_nodes_DoF.append((i[0][1]-1)*DoF+i[1][1]-1)
        all_meas.append(i[2][0]-1)
    return np.asarray(all_resp_nodes_DoF), np.asarray(all_exc_nodes_DoF), np.asarray(all_meas)


def red_order(A, sv=0):
    """
    Filtration of input matrix ``A`` with singular value decomposition by reducing considered singular values.

    :param A: Matrix to be filtered by singular value decomposition
    :type A: array(float)
    :param sv: Number of singular values not taken into account by reconstruction of the matrix A
    :type sv: int
    :return: Filtered matrix A
    :rtype: array(float)
    """
    U, s, VT = np.linalg.svd(A)
    kk = s.shape[1] - sv
    Uk = U[:, :, :kk]
    Sk = np.zeros((A.shape[0], kk, kk))

    for i in range(A.shape[0]):
        Sk[i] = np.diag(s[i, :kk])
    Vk = VT[:, :kk, :]

    return Uk @ Sk @ Vk


def SEMM(Y_num, Y_exp, overlay, DoF, loc_Y_num=None, SEMM_type='fully-extend', red_comp=0, red_eq=0):
    """
    This function performs SEMM. It couples numerical (``Y_num``) and experimental (``Y_exp``) model to hybrid model. 
    Connections between both models are defined in the ``overlay`` parameter, the number of DoFs per one node is defined in parameter ``DoF``.

    :param Y_num: Numerical response matrix
    :type Y_num: array(float)
    :param Y_exp: Experimental response matrix
    :type Y_exp: array(float)
    :param overlay: Defines connections between numerical and experimental model
    :type overlay: list
    :param DoF: Number of DoFs per one node
    :type DoF: int
    :param loc_Y_num: Defines which nodes are represented in the numerical model if it is not a full square matrix
    :type loc_Y_num: list or ``None``, optional
    :param SEMM_type: Defined which type of SEMM will be performed - basic ("basic") or fully extended ("fully-extend") or fully extended with SVD truncation on compatibility or equilibrium ("fully-extended-svd")
    :type SEMM_type: str("basic" or "fully-extend" or "fully-extended-svd")
    :param red_comp: Defines how many maximum singular values will not be taken into account in ensuring compatibility conditions
    :type red_comp: int
    :param red_eq: Defines how many maximum singular values will not be taken into account in ensuring equilibrium conditions
    :type red_eq: int
    :return: ``Y_SEMM``

        |  Hybrid model based on numerical and experimental data
    :rtype: array(float)


    How to define the ``overlay`` parameter?

    The base model is presented by the numerical model. Let's assume, 
    that there are 2 DoFs at every node (deformation in *x* and *y* direction). 
    The numbering of the numerical model is presented in the bottom picture:

    ::

        1______2______3______4*
        |      |      |      |
        |      |      |      |
        |______|______|______|
        |5     |6     |7     |8
        |      |      |      |
        |______|______|______|
        9"     10"    11"    12"

    Experimental measurement was performed with the accelerometer on point 4 (marked with ``*``) in *x* direction.
    Excitation was made on points 9, 10, 11, 12 (marked with ``"``) also in *x* direction.

    Assume that the measurements in the ``Y_exp`` parameter belong to the responses in points 9, 10, 11 and 12. 
    This means that the measurement in the first place in the ``Y_exp`` parameter represents the excitation in point 9, 
    the measurements in the second place presents excitation in point 10, and so on.

    Now we can define overlay parameter. Every particular line belongs to one meassurement.

    ::

        overlay=[[[4,  9], [1, 1], [1]],
                 [[4, 10], [1, 1], [2]],
                 [[4, 11], [1, 1], [3]],
                 [[4, 12], [1, 1], [4]]]

    Let's look at the first line of ``overlay`` parameter. This means that response node 4 (with respect to numerical model) 
    and excitation point 9 (with respect to numerical model) with response direction 1 (*x* direction) and excitation
    direction 1 (*x* direction), belongs to the first measurement in the experimental model.

    List ``loc_Y_num`` holds information about nodes represented in the inputted numerical receptance matrix.
    This information is necessary only if the inputted numerical matrix is not square shape.
    Let's assume, that our numerical model consists of responses at nodes 2, 3 and 4 and excitations at
    points 1, 2, 3, 4 and 5. This means, that the inputted numerical matrix has dimensions 3x5. So list
    loc_Y_num has following form:

    ::

        loc_Y_num = [[2, 3, 4], [1, 2, 3, 4, 5]]
    """

    # Validation of input data
    Y_num = np.asarray(Y_num)
    if len(Y_num.shape) != 3:
        raise Exception('Wrong shape of input numerical receptance matrix.')

    if loc_Y_num == None:
        loc_Y_num = [np.arange(int(Y_num.shape[1]/DoF))+1,
                     np.arange(int(Y_num.shape[2]/DoF))+1]
        if Y_num.shape[1] != Y_num.shape[2]:
            raise Exception(
                "Input numerical model must be square matrix, or define parameter loc_Y_num.")
    else:
        if len(loc_Y_num) != 2:
            raise Exception(
                "List of locations loc_Y_num of inputed numerical matrix is not defined correctly.")
        if len(loc_Y_num[0])*DoF != Y_num.shape[1]:
            raise Exception(
                "Input numerical model and corespondind points of response must match.")
        if len(loc_Y_num[1])*DoF != Y_num.shape[2]:
            raise Exception(
                f"Input numerical model and corespondind points of excitation must match. {len(loc_Y_num[1])*DoF}=/={Y_num.shape[2]}")
        overlay = copy.deepcopy(overlay)
        for _, i in enumerate(overlay):
            try:
                overlay[_][0][0] = list(loc_Y_num[0]).index(i[0][0])+1
                overlay[_][0][1] = list(loc_Y_num[1]).index(i[0][1])+1
            except ValueError:
                raise Exception(
                    "Input numerical model does not have corresponding locations with respect to experimental model.")
    if len(Y_exp.shape) != 2:
        raise Exception('Input experimental matrx must be 2D matrix.')

    # Initialization data
    Y_par = np.copy(Y_num)
    Y_exp = np.asarray(Y_exp)

    # Data preparation for building parent, remowed and overlay model
    # Reviewing all experimental obtained DoFs
    all_resp_nodes_DoF, all_exc_nodes_DoF, all_meas = loc_of_exp(
        overlay, DoF)

    # Define unique locations of performed excitations and responses
    uniq_resp_nodes_DoF = np.unique(all_resp_nodes_DoF)
    uniq_exc_nodes_DoF = np.unique(all_exc_nodes_DoF)
    if len(np.unique(all_meas)) != len(np.sort(all_meas)):
        raise Exception(
            'One measuement can not be assign to multiple nodes.')

    # Define unique locations of performed excitations and responses started counting from 0
    all_resp_nodes_DoF_0 = [list(uniq_resp_nodes_DoF).index(i)
                            for i in all_resp_nodes_DoF]
    all_exc_nodes_DoF_0 = [list(uniq_exc_nodes_DoF).index(
        i) for i in all_exc_nodes_DoF]

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
    Y_ov = np.zeros((Y_par.shape[0], len(uniq_resp_nodes_DoF), len(
        uniq_exc_nodes_DoF)), dtype=complex)
    for i, j in enumerate(all_meas):
        Y_ov[:, all_resp_nodes_DoF_0[i],
             all_exc_nodes_DoF_0[i]] = Y_exp[:Y_par.shape[0], j]

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

    elif SEMM_type == "fully-extended-svd":
        Y_SEMM = Y_par - Y_par @ np.linalg.pinv(red_order(Y_par[:, -len(uniq_resp_nodes_DoF):, :], sv=red_comp))  @ (
            Y_rem - Y_ov) @ np.linalg.pinv(red_order(Y_par[:, :, -len(uniq_exc_nodes_DoF):], sv=red_eq)) @ Y_par

    #U, s, VT = np.linalg.svd(Y_par[:, -len(uniq_resp_nodes_DoF):, :])
    #U, s, VT = np.linalg.svd(Y_par[:, :, -len(uniq_exc_nodes_DoF):])
    # plt.semilogy(s)

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


if __name__ == '__main__':
    print("Test: Cat!")
