import numpy as np
import copy
import matplotlib.pyplot as plt




def LocOfExp(Overlay, DoF):
    """
    Preparation of data to find locations of corresponding experimental meassurement in numerical model.

    :param Overlay: Defines connections bettwen numerical and experimental model
    :type Overlay: list
    :param DoF: Number of DoFs per one node
    :type DoF: int

    :return: something cat like
    """
    AllResponseNodesDoF = []
    AllExcitationNodesDoF = []
    AllMeasurement = []
    for i in Overlay:
        AllResponseNodesDoF.append((i[0][0]-1)*DoF+i[1][0]-1)
        AllExcitationNodesDoF.append((i[0][1]-1)*DoF+i[1][1]-1)
        AllMeasurement.append(i[2][0]-1)

    AllResponseNodesDoF = np.asarray(AllResponseNodesDoF)
    AllExcitationNodesDoF = np.asarray(AllExcitationNodesDoF)
    AllMeasurement = np.asarray(AllMeasurement)
    return AllResponseNodesDoF, AllExcitationNodesDoF, AllMeasurement


def red_order(A, sv = 0):
    print(A.shape)
    U, s, VT = np.linalg.svd(A)
    kk = s.shape[1] - sv
    print(kk)
    Uk = U[:, :, :kk]
    Sk = np.zeros((A.shape[0], kk, kk))

    for i in range(A.shape[0]):
        Sk[i] = np.diag(s[i, :kk])
    Vk = VT[:, :kk, :]

    return Uk @ Sk @ Vk

def SEMM(Y_num, Y_exp, Overlay, DoF, Loc_Y_num=None, SEMM_type='fully-extend',red_comp = 0,red_eq = 0):
    """
    This function performs SEMM. It couples numerical (Y_num)
    and experimental (Y_exp) model to hybrid model. Connections bettwen
    both models are defined in Overlay parameter, number of DoFs per one
    node is defined in parameter DoF.

    Warning: numerical model must be squere matrx

    Arguments:
        Y_num {numpy.ndarray} -- numerical receptance matrix
        Y_exp {numpy.ndarray} -- experimental receptance matrix
        Overlay {list} -- defines connections bettwen numerical and experimental model
        DoF {int} -- number of DoFs per one node
        Loc_Y_num {list} -- defines which nodes are represented in numerical model if it is not full squere matrix
        SEMM_type {str} -- defined which type of SEMM will be performed - basic ("basic") or fully extended ("fully-extend") or fully extended with SVD truncation on compatibility or equilibrium ("full-extended-svd")

    Returns:
        numpy.ndarray -- hybrid model based on numerical and experiemral data


    How to define Overlay parameter:
    The base model is presented py numerical model. Let's assume, that there are 2 DoFs
    on every node (deformation in x and y direction). Numbering of numerical model is
    presented on bottom picture:
     1______2______3______4*
     |      |      |      |
     |      |      |      |
     |______|______|______|
    5|     6|     7|     8|
     |      |      |      |
     |______|______|______|
     9"     10"    11"    12"

    Experimental measurement was performed with accelometer on point 4 (marked with *) in x direction.
    Excitation was made on points 9, 10, 11, 12 (marked with ") also in x direction.

    Let's assume that meassurement in experimental model have order (9, 10, 11, 12).

    Now we can define Overlay parameter. Every particular line belongs to one meassurement.
    Overlay=[[[4,  9], [1, 1], [1]],
             [[4, 10], [1, 1], [2]],
             [[4, 11], [1, 1], [3]],
             [[4, 12], [1, 1], [4]]]
    This menas that response node 4 (with respect to numerical model) and excitation point 9
    (with respect to numerical model) with responce direction 1 (x direction) and excitation
    direction 1, belongs to the first measurment in experimentla model.

    List Loc_Y_num holds information about  nodes represented in inputed numerical receptance matrix.
    This inforamtion in nessesery only if inputed numerical matry is not squere.
    Let's assume, that our numerical model consist of responses at nodes 2 and 3 and excitations at
    points 1, 2, 3, 4 and 5. This mean, that inputed numerical matrix has dimensions 2x5. So list
    Loc_Y_num has form:
    Loc_Y_num=[[2, 3], [1, 2, 3, 4, 5]]

    """

    # Validation of input data
    Y_num = np.asarray(Y_num)
    if len(Y_num.shape) != 3:
        raise Exception('Wrong shape of input numerical receptance matrix')

    if Loc_Y_num == None:
        Loc_Y_num = [np.arange(int(Y_num.shape[1]/DoF))+1,
                     np.arange(int(Y_num.shape[2]/DoF))+1]
        if Y_num.shape[1] != Y_num.shape[2]:
            raise Exception(
                "Input numerical model must be square matrix, or define parameter Loc_Y_num.")
    else:
        if len(Loc_Y_num) != 2:
            raise Exception(
                "List of locations Loc_Y_num of inputed numerical matrix is not defined correctly.")
        if len(Loc_Y_num[0])*DoF != Y_num.shape[1]:
            raise Exception(
                "Input numerical model and corespondind points of response must match.")
        if len(Loc_Y_num[1])*DoF != Y_num.shape[2]:
            raise Exception(
                "Input numerical model and corespondind points of excitation must match. {0}=/={1}".format(len(Loc_Y_num[1])*DoF, Y_num.shape[2]))
        Overlay = copy.deepcopy(Overlay)
        for _, i in enumerate(Overlay):
            try:
                Overlay[_][0][0] = list(Loc_Y_num[0]).index(i[0][0])+1
                Overlay[_][0][1] = list(Loc_Y_num[1]).index(i[0][1])+1
            except ValueError:
                raise Exception(
                    "Input numerical model does not have corresponding locations with respect to experimental model")
    if len(Y_exp.shape) != 2:
        raise Exception('Input experimental matrx must be 2D matrix')

    # Initialization data
    Y_par = np.copy(Y_num)
    Y_exp = np.asarray(Y_exp)

    # Data preparation for building parent, remowed and overlay model
    # Reviewing all experimental obtained DoFs
    AllResponseNodesDoF, AllExcitationNodesDoF, AllMeasurement = LocOfExp(
        Overlay, DoF)

    # Define unique locations of performed excitations and responses
    UniqueResponseNodesDoF = np.unique(AllResponseNodesDoF)
    UniqueExcitationNodesDoF = np.unique(AllExcitationNodesDoF)
    if len(np.unique(AllMeasurement)) != len(np.sort(AllMeasurement)):
        raise Exception(
            'One experimental measuement can not be assign to multiple nodes')

    # Define unique locations of performed excitations and responses started counting from 0
    AllResponseNodesDoF_0 = [list(UniqueResponseNodesDoF).index(i)
                             for i in AllResponseNodesDoF]
    AllExcitationNodesDoF_0 = [list(UniqueExcitationNodesDoF).index(
        i) for i in AllExcitationNodesDoF]

    # Construction of parent model
    # moved collumns
    AllExcitationNodesDoF_ForAppend = Y_par[:, :, (UniqueExcitationNodesDoF)]
    Y_par = np.delete(Y_par, UniqueExcitationNodesDoF, axis=2)
    Y_par = np.concatenate((Y_par, AllExcitationNodesDoF_ForAppend), axis=2)

    # moved rows
    AllResponseNodesDoF_ForAppend = Y_par[:, (UniqueResponseNodesDoF), :]
    Y_par = np.delete(Y_par, UniqueResponseNodesDoF, axis=1)
    Y_par = np.concatenate((Y_par, AllResponseNodesDoF_ForAppend), axis=1)

    # Construction of removed model
    Y_rem = AllExcitationNodesDoF_ForAppend[:, (UniqueResponseNodesDoF), :]

    # Construction of overlay model
    Y_ov = np.zeros((Y_par.shape[0], len(UniqueResponseNodesDoF), len(
        UniqueExcitationNodesDoF)), dtype=complex)
    for i, j in enumerate(AllMeasurement):
        Y_ov[:, AllResponseNodesDoF_0[i],
             AllExcitationNodesDoF_0[i]] = Y_exp[:Y_par.shape[0], j]

    # Y_ov=Y_rem # ONLY FOR TESTING

    if SEMM_type == "basic":
        # Single-line method SEMM - basic form - eq(21)
        try:
            Y_SEMM = Y_par-Y_par[:, :, -len(UniqueExcitationNodesDoF):]@np.linalg.inv(Y_rem)@(
                Y_rem-Y_ov)@np.linalg.inv(Y_rem)@Y_par[:, -len(UniqueResponseNodesDoF):, :]
        except np.linalg.LinAlgError:
            Y_SEMM = Y_par-Y_par[:, :, -len(UniqueExcitationNodesDoF):]@np.linalg.pinv(Y_rem)@(
                Y_rem-Y_ov)@np.linalg.pinv(Y_rem)@Y_par[:, -len(UniqueResponseNodesDoF):, :]

    elif SEMM_type == "fully-extend":
        # Single-line method SEMM - fully-extend form - eq(31)
        Y_SEMM = Y_par-Y_par@np.linalg.pinv(Y_par[:, -len(UniqueResponseNodesDoF):, :])@(
            Y_rem-Y_ov)@np.linalg.pinv(Y_par[:, :, -len(UniqueExcitationNodesDoF):])@Y_par

    elif SEMM_type == "fully-extended-svd":
        Y_SEMM = Y_par - Y_par @ np.linalg.pinv(red_order(Y_par[:, -len(UniqueResponseNodesDoF):, :],sv = red_comp))  @ (Y_rem - Y_ov) @ np.linalg.pinv(red_order(Y_par[:, :, -len(UniqueExcitationNodesDoF):],sv = red_eq)) @ Y_par

    #U, s, VT = np.linalg.svd(Y_par[:, -len(UniqueResponseNodesDoF):, :])
    #U, s, VT = np.linalg.svd(Y_par[:, :, -len(UniqueExcitationNodesDoF):])
    #plt.semilogy(s)

    # rearranging SEMM model to input numerical form od DOFs
    # moved collumns
    AllExcitationNodesDoF_ForAppend = Y_SEMM[:,
                                             :, -len(UniqueExcitationNodesDoF):]
    Y_SEMM = Y_SEMM[:, :, :-len(UniqueExcitationNodesDoF)]

    for index, i in enumerate(UniqueExcitationNodesDoF):
        Y_SEMM = np.insert(
            Y_SEMM, i, AllExcitationNodesDoF_ForAppend[:, :, index], axis=2)

    # moved rows
    AllResponseNodesDoF_ForAppend = Y_SEMM[:, -len(UniqueResponseNodesDoF):, :]
    Y_SEMM = Y_SEMM[:, :-len(UniqueResponseNodesDoF), :]

    for index, i in enumerate(UniqueResponseNodesDoF):
        Y_SEMM = np.insert(
            Y_SEMM, i, AllResponseNodesDoF_ForAppend[:, index, :], axis=1)

    return Y_SEMM


if __name__ == '__main__':
    print("Test: Cat!")