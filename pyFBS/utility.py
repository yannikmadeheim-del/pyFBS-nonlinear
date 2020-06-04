import matplotlib.pyplot as plt
import numpy as np
from numpy import cross, eye
from scipy.linalg import expm, norm
import pandas as pd
from scipy.spatial.transform import Rotation as R


def response_sync_lstsq(response_vec):
    """
    Description

    :param response_vec:
    :return:
    """
    _mode = response_vec.flatten()
    z = np.arctan(np.average(np.imag(_mode) / np.real(_mode), weights=np.abs(_mode) ** 2))

    response_vec_norm = response_vec * (np.cos(-1 * z) + 1j * np.sin(-1 * z))
    return response_vec_norm


def modeshape_sync_lstsq(mode_shape_vec):
    """
    Creates a straight line fit in the complex plane and substracts the phase from mode shapes.
    """
    _n = np.zeros_like(mode_shape_vec)
    for i in range(np.shape(mode_shape_vec)[1]):
        _mode = mode_shape_vec[:,i]
        z = np.arctan(np.average(np.imag(_mode)/np.real(_mode),weights = np.abs(_mode)**2))
            
        _n[:,i] = _mode*(np.cos(-1*z)+1j*np.sin(-1*z))
    return _n

def modeshape_scaling_DP(mode_shape_vec, driving_point,sync = True):
    """
    Scales modeshape from driving point measurement.
    """
    
    _mode = mode_shape_vec
    for i in range(np.shape(mode_shape_vec)[1]):
        _mode[:,i] = _mode[:,i]/np.sqrt(mode_shape_vec[driving_point,i])
    
    if sync:
        _mode = modeshape_sync_lstsq(_mode)
    return _mode        

def MCF(mod):
    """
    Calculate Mode Complexity Factor (MCF)
    """
    sxx = np.real(mod).T@np.real(mod)
    syy = np.imag(mod).T@np.imag(mod)
    sxy = np.real(mod).T@np.imag(mod)
    mcf = (1 - ((sxx-syy)**2+4*sxy**2)/((sxx+syy)**2))
    return mcf




def flatten_FRFs(Y):
    """
    :param Y:
    :return:
    Flattens FRF matrix Y from shape (out,in,freq) in (out x in,freq)
    """
    new = np.zeros((Y.shape[0] * Y.shape[1], Y.shape[2]), dtype=complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new[_len * i:_len * (i + 1), :] = Y[i, :, :]

    return new

def unflattenFRFs(_modes_acc,Y):
    """
    :param _modes_acc:
    :param Y:
    :return:
    Reconstructs modeshapes from pyEMA.get_constants in 3D space based on the structure of Y
    """
    new_mode = np.zeros((Y.shape[0],Y.shape[1],_modes_acc.shape[1]),dtype = complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new_mode[i,:,:] = _modes_acc[i*_len:(i+1)*_len,:]
    return new_mode

def complex_plot_3D(mode_shape):
    """
    Plots modeshape on a radial plot.
    :param mode_shape:
    :return:
    """
    plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111,projection = "polar")

    for i,color in enumerate(["tab:red","tab:green","tab:blue"]):
        for x in mode_shape[:,i]:
            ax1.plot([0,np.angle(x)],[0,np.abs(x)],marker='.',color = color,alpha = 0.5)

    plt.yticks([])

def mode_animation(mode_shape,scale, no_points=60):
    """
    Create animation sequence
    """
    ann = np.zeros((mode_shape.shape[0], mode_shape.shape[1], no_points))

    for g, _t in enumerate(np.linspace(0, 2, no_points)):
        ann[:, :, g] = (np.real(mode_shape) * np.cos(2 * np.pi * _t) - np.imag(mode_shape) * np.sin(
            2 * np.pi * _t))
    ann = ann / np.max(ann) *scale
    return ann


def coh_frf(h_num, h_exp, check=False):
    """
    :param h_num: numerični kompleksni vektor FRF
    :param h_exp: eksperimentalni kompleksni vektor FRF

    :return: vrednost coherence kriterija
    """

    h_numk = np.conjugate(h_num)
    h_expk = np.conjugate(h_exp)

    def vector(h_, h_K):
        """
        :param h_: kompleksni vektor FRF
        :param h_K: konjugirani kompleksni vektor FRF

        :return: vektorski produkt
        """

        vec = np.dot(h_, h_K)
        return vec

    coh = np.abs(vector((h_num + h_exp), (h_numk + h_expk))) / 2 / (vector(h_numk, h_num) + vector(h_expk, h_exp))
    coh_abs = np.abs(coh)

    if check:
        return coh, coh_abs
    else:
        return coh_abs

def dict_animation(_modeshape,a_type,mesh= None,pts = None,fps = 30,r_scale = 10,no_points = 60, object_list = None):
    """
    Description

    :param _modeshape:
    :param pts:
    :param mesh:
    :param a_type:
    :param fps:
    :param r_scale:
    :param no_points:
    :param object_list:
    :return:
    """
    mode_dict = dict()

    mode_dict["animation_pts"] = mode_animation(_modeshape, r_scale, no_points=no_points)
    mode_dict["fps"] = fps

    if a_type == "modeshape":
        mode_dict["or_pts"] = pts
        mode_dict["mesh"] = mesh
        mode_dict["scalars"] = True

    elif a_type == "object":
        mode_dict["objects_list"] = object_list

    return mode_dict


def CMIF(FRF, singular_vectors=False):
    """
    Calculates a CMIF parameter on an FRF matrix

    :param FRF:
    :param singular_vector:
    :return:
    """
    _f = FRF.shape[0]
    val = np.min([FRF.shape[1], FRF.shape[2]])

    _S = np.zeros((_f, val))

    if singular_vectors:
        _U = np.zeros((_f, FRF.shape[1], FRF.shape[1]), dtype="complex")
        _V = np.zeros((_f, FRF.shape[2], FRF.shape[2]), dtype="complex")

    for i in range(_f):
        if singular_vectors:
            U, S, VH = np.linalg.svd(FRF[i, :, :], full_matrices=True, compute_uv=True)
            V = np.conj(VH).T
            _S[i, :] = S
            _U[i, :, :] = U
            _V[i, :, :] = V

        else:
            S = np.linalg.svd(FRF[i, :, :], full_matrices=True, compute_uv=False)
            _S[i, :] = S

    if singular_vectors:
        return _U, _S, _V
    else:
        return _S


def TSVD(matrix,reduction = 0):
    """
    Performs a TSVD on a suplied FRF matrix

    :param matrix:
    :param reduction: number of removed singular values
    :return:
    """
    U, s, VH = np.linalg.svd(matrix)
    kk = s.shape[1] - reduction
    Uk = U[:, :, :kk]
    Sk = np.zeros((matrix.shape[0], kk, kk))

    for i in range(matrix.shape[0]):
        Sk[i] = np.diag(s[i, :kk])
    Vk = VH[:, :kk, :]

    return Uk @ Sk @ Vk

def M(axis, theta):
    """
    Euler-Rodrigues formula
    """
    t = expm(cross(eye(3), axis / norm(axis) * (theta)))
    # print(theta)

    return t


def angle(vector1, vector2):
    """
    Returns the angle in radians between given vectors
    """
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return sign * np.arccos(dot_p)


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2

    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 += np.random.random(3) / 1e10  # just to avoid possible math errors
    vec2 += np.random.random(3) / 1e10  # just to avoid possible math errors

    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)

    if (np.abs(a) == np.abs(b)).all():
        return np.diag([1, 1, 1])
    else:
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        return rotation_matrix


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate_channels_from_sensors(df):
    """
    Description

    :param df:
    :return:
    """
    columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3"]
    df_ch = pd.DataFrame(columns=columns_chann)

    axes = ["x", "y", "z"]
    for s, angle in enumerate(df[["Orientation_1", "Orientation_2", "Orientation_3"]].to_numpy()):
        r = R.from_euler('xyz', angle, degrees=True)
        rot = r.as_matrix().T
        for i in range(3):
            data_chn = np.asarray([[df["Name"][s] + axes[i], df["Description"][s], df["Type"][s], None, None, None,
                                    None, None, df["Grouping"][s], df["Position_1"][s], df["Position_2"][s],
                                    df["Position_3"][s], rot[i][0], rot[i][1], rot[i][2]]])
            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df_ch = df_ch.append(df_row,ignore_index = True)

    return df_ch

def generate_sensors_from_channels(df):
    """
    Description

    :param df:
    :return:
    """
    columns_sen = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                     "Grouping", "Position_1", "Position_2", "Position_3", "Orientation_1", "Orientation_2", "Orientation_3"]
    df_sen = pd.DataFrame(columns=columns_sen)

    for i in range(int(len(df)/3)):
        sen_or = df[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()[3 * (i):3 * (i + 1)]
        sen_pos = df[["Position_1", "Position_2", "Position_3"]].to_numpy()[3 * (i)]
        #sen_name = df[["Name"]].to_numpy()[3 * (i):3 * (i + 1)]

        r = R.from_matrix(sen_or)
        r = r.inv()

        orient = r.as_euler('xyz', degrees=True)

        data_chn = np.asarray([["S"+str(i+1),None,None,None,None,None,None,None,None,sen_pos[0],sen_pos[1],sen_pos[2],orient[0],orient[1],orient[2]]])


        df_row = pd.DataFrame(data=data_chn, columns=columns_sen)
        df_sen = df_sen.append(df_row,ignore_index = True)

    return df_sen

def coh_on_FRF(FRF_matrix):
    """
    Description

    :param FRF_matrix:
    :return:
    """
    _out = FRF_matrix.shape[1]
    _in = FRF_matrix.shape[2]

    coh_crit = np.zeros((_out, _in))

    for i in range(_out):
        for j in range(_in):
            coh_crit[i, j] = coh_frf(FRF_matrix[:, i, j], FRF_matrix[:, j, i])

    return coh_crit