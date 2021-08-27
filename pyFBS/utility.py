import matplotlib.pyplot as plt
import numpy as np
from numpy import cross, eye
from scipy.linalg import expm, norm
import pandas as pd
from scipy.spatial.transform import Rotation as R
from pyts.decomposition import SingularSpectrumAnalysis
import altair as alt


alt.data_transformers.enable('json')
alt.data_transformers.enable('default', max_rows=None)


def modeshape_sync_lstsq(mode_shape_vec):
    """
    Creates a straight line fit in the complex plane and alligns the mode shape with the real-axis.

    :param mode_shape_vec: Mode shape vector
    :type mode_shape_vec: array(float)
    :return _n: Alligned mode shape vector
    """
    _n = np.zeros_like(mode_shape_vec)
    for i in range(np.shape(mode_shape_vec)[1]):
        _mode = mode_shape_vec[:,i]
        z = np.arctan(np.average(np.imag(_mode)/np.real(_mode),weights = np.abs(_mode)**1e4))
            
        _n[:,i] = _mode*(np.cos(-1*z)+1j*np.sin(-1*z))
    return _n

def modeshape_scaling_DP(mode_shape_vec, driving_point,sync = True):
    """
    Scales mode shapes according to the driving point measurement.

    :param mode_shape_vec: Mode shape vector
    :type mode_shape_vec: array(float)
    :param driving_point: Driving point location
    :type driving_point: int
    :param sync: Allign mode shape with the real-axis
    :type sync: bool, optional
    :return: Scalled mode shape
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

    :param mod: Mode shape
    :type mod: array(float)
    :return: Mode complexity factor
    """
    sxx = np.real(mod).T@np.real(mod)
    syy = np.imag(mod).T@np.imag(mod)
    sxy = np.real(mod).T@np.imag(mod)
    mcf = (1 - ((sxx-syy)**2+4*sxy**2)/((sxx+syy)**2))
    return mcf

def flatten_FRFs(Y):
    """
    Flattens input FRF matrix Y from shape (out,in,freq) in (out x in,freq)

    :param Y: Matrix of FRFs [out,in,f]
    :type Y: array(float)
    :return:  Matrix of FRFs [out x in,f]
    """
    new = np.zeros((Y.shape[0] * Y.shape[1], Y.shape[2]), dtype=complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new[_len * i:_len * (i + 1), :] = Y[i, :, :]

    return new

def unflatten_modes(_modes_acc, Y):
    """
    Unflattens mode shapes based on the shape of the input FRF matrix [out x in] in [out, in]

    :param _modes_acc: Mode shape [out x in]
    :type _modes_acc: array(float)
    :param Y:
    :return: Unflattened mode shape [out, in]
    """
    new_mode = np.zeros((Y.shape[0],Y.shape[1],_modes_acc.shape[1]),dtype = complex)

    _len = Y.shape[1]
    for i in range(Y.shape[0]):
        new_mode[i,:,:] = _modes_acc[i*_len:(i+1)*_len,:]
    return new_mode

def complex_plot(mode_shape, color = "k"):
    """
    Plots a mode shape on a radial plot.

    :param mode_shape: mode shape
    :type mode_shape: array(float)
    :param color: Color of the plot
    :type color: str
    """
    plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111,projection = "polar")

    for x in mode_shape:
        ax1.plot([0,np.angle(x)],[0,np.abs(x)],marker='.',color = color,alpha = 0.5)

    plt.yticks([])


def complex_plot_3D(mode_shape):
    """
    Plots a 3D mode shape on a radial plot.

    :param mode_shape: 3D mode shape
    :type mode_shape: array(float)
    """
    plt.figure(figsize = (3,3))
    ax1 = plt.subplot(111,projection = "polar")

    for i,color in enumerate(["tab:red","tab:green","tab:blue"]):
        for x in mode_shape[:,i]:
            ax1.plot([0,np.angle(x)],[0,np.abs(x)],marker='.',color = color,alpha = 0.5)

    plt.yticks([])

def mode_animation(mode_shape, scale, no_points=60, no_of_repetitions = 2, abs_scale = True, secondary_mode_shape = None, animate_secondary_mode_shape = False):
    """
    Creates an animation sequence from the mode shape and scales the displacemetns.
    It is also possible to add a secondary mode shape, which is displayed on a deformed 
    structure using colours based on values of secondary mode shape.
    Secondary mode shape could be rotational mode shape or strain mode shape, 
    any other parameter, which can be displayed on nodes.

    :param mode_shape: mode shape, must be 2D matrix
    :type mode_shape: array(float)
    :param scale: mode shape
    :type scale: float
    :param no_points: Number of points in the animation sequence
    :type no_points: int, optional
    :param no_of_repetitions: Number of repetitions of animated mode
    :type no_of_repetitions: int, optional
    :param abs_scale: Apply scaling on normalized mode
    :type abs_scale: bool, optional
    :param secondary_mode_shape: secondary mode shape, must be vector
    :type secondary_mode_shape: array(float), optional
    :param animate_secondary_mode_shape: If ``True``, secondary mode shape will be animated, if ``False`` still only initial mode shape will be animated
    :type animate_secondary_mode_shape: bool, optional
    :return: Animation sequence
    """
    if isinstance(mode_shape, np.ndarray):
        if mode_shape.ndim==2:
            ann = np.zeros((mode_shape.shape[0], mode_shape.shape[1], int(no_points)))
            ann_secondary = np.zeros((mode_shape.shape[0], int(no_points)))
        else:
            raise ValueError("Parameter mode_shape must be a 2D vector, where the first dimension presents all nodes and the second dimension 3 coordinates (x, y, z).")
    else:
        raise ValueError("To animate mode shape, a parameter mode_shape must be defined in form of 2D numpy array.")


    for g, _t in enumerate(np.linspace(0, int(no_of_repetitions), int(no_points))):
        ann[:, :, g] = (np.real(mode_shape) * np.cos(2 * np.pi * _t) - np.imag(mode_shape) * np.sin(2 * np.pi * _t))
        if animate_secondary_mode_shape:
            if isinstance(secondary_mode_shape, np.ndarray):
                if secondary_mode_shape.ndim==1:
                    ann_secondary[:, g] = (np.real(secondary_mode_shape) * np.cos(2 * np.pi * _t) - np.imag(secondary_mode_shape) * np.sin(2 * np.pi * _t))
                else:
                    raise ValueError("Parameter secondary_mode_shape must be 1D vector.")
            else:
                raise ValueError("To animate secondary mode shape, a parameter secondary_mode_shape must be defined in form of 1D numpy array.")

    if abs_scale:
        ann = ann / np.max(ann) * scale
    else:
        ann = ann * scale
    return ann, ann_secondary


def MAC(phi_1, phi_2, output_type = 'matrix'):
    """
    Calculates modal assurance criterion matrix.

    :param phi_1: modal matrix or modeshapes 1, shape: ``(n_locations, n_modes)``
    :type phi_1: array(float)
    :param phi_2: modal matrix or modeshapes 1, shape: ``(n_locations, n_modes)``
    :type phi_2: array(float)
    :param output_type: output type - 'matrix' or 'diagonal'
    :type output_type: str('matrix', 'diagonal')
    :return: MAC values
    """
    if phi_1.shape[0] != phi_2.shape[0]:
        raise Exception('Input dimensions are not compatible.')
    if phi_1.ndim == 1:
        phi_1 = phi_1[:,np.newaxis]
    if phi_2.ndim == 1:
        phi_2 = phi_2[:,np.newaxis]

    MAC_mat = np.abs(np.einsum('ri,ik->rk',np.conj(phi_1).T,phi_2))**2 / (np.einsum('ri,ir->r',np.conj(phi_1).T,phi_1)[:,np.newaxis] * np.einsum('ri,ir->r',np.conj(phi_2).T,phi_2))
    if output_type == 'matrix':
        return MAC_mat
    if output_type == 'diagonal':
        return np.diagonal(MAC_mat)
    else:
        raise Exception('Unknown output type.')

def coh_frf(Y_1, Y_2, return_average = True):
    """
    Calculates values of coherence between two FRFs.

    :param Y_1: FRF 1
    :type Y_1: array(float)
    :param Y_2: FRF 2
    :type Y_2: array(float)
    :return: coherence criterion
    """

    if Y_1.shape == Y_2.shape:
        if len(Y_1.shape) == 3:
            numerator = np.einsum("ijk,ijk->ijk", (Y_1+Y_2), (np.conj(Y_1)+np.conj(Y_2)))
            denumerator = 2*(np.einsum("ijk,ijk->ijk", Y_1, np.conj(Y_1)) + np.einsum("ijk,ijk->ijk", Y_2, np.conj(Y_2)))
            coh = np.einsum("ijk,ijk->ijk", numerator, 1/denumerator)
        elif len(Y_1.shape) == 2:
            numerator = np.einsum("ij,ij->ij", (Y_1+Y_2), (np.conj(Y_1)+np.conj(Y_2)))
            denumerator = 2*(np.einsum("ij,ij->ij", Y_1, np.conj(Y_1)) + np.einsum("ij,ij->ij", Y_2, np.conj(Y_2)))
            coh = np.einsum("ij,ij->ij", numerator, 1/denumerator)
        elif len(Y_1.shape) == 1:
            numerator = (Y_1+Y_2)*(np.conj(Y_1)+np.conj(Y_2))
            denumerator = 2*((Y_1*np.conj(Y_1)) + (Y_2*np.conj(Y_2)))
            coh = numerator/denumerator
        else:
            print("Wrong matrix shape")
        
        if return_average == True:
            return np.mean(np.abs(coh))
        else:
            return np.abs(coh)
    else:
        print("Wrong matrix shape")
        return None

def dict_animation(_modeshape, a_type, mesh= None, pts = None, fps = 30, r_scale = 10, no_points=60, no_of_repetitions = 2, object_list = None, abs_scale = True, secondary_mode_shape=None, animate_secondary_mode_shape = False):
    """
    Creates a predefined dictionary for animation sequency in the 3D display.

    :param _modeshape: A mode shape or response to be animated
    :type _modeshape: array(float)
    :param a_type: Animation type ("modeshape" or "object")
    :type a_type: str
    :param mesh: Mesh to be animated
    :type mesh: array(float), optional
    :param pts: Points to be animated
    :type pts: array(float), optional
    :param fps: Frames per second of the animation
    :type fps: int, optional
    :param r_scale: Relative scale of the displacement
    :type r_scale: float, optional
    :param no_points: Number of points in the animation sequence
    :type no_points: int, optional
    :param no_of_repetitions: Number of repetitions of animated mode
    :type no_of_repetitions: int, optional
    :param object_list: A list containing objects to be animated
    :type object_list: list, optional
    :param abs_scale: Apply scaling on normalized mode
    :type abs_scale: bool, optional
    :param secondary_mode_shape: secondary mode shape
    :type secondary_mode_shape: array(float), optional
    :param animate_secondary_mode_shape: If ``True``, secondary mode shape will be animated, if ``False`` still only initial mode shape will be animated
    :type animate_secondary_mode_shape: bool, optional
    :return: Dictionary of parameters for mode shape animation
    """
    mode_dict = dict()
    
    mode_animation_frames = mode_animation(_modeshape, r_scale, no_points = no_points, no_of_repetitions = no_of_repetitions, abs_scale = abs_scale, secondary_mode_shape = secondary_mode_shape, animate_secondary_mode_shape = animate_secondary_mode_shape)
    mode_dict["animation_pts"] = mode_animation_frames[0]
    mode_dict["animation_pts_secondary"] = mode_animation_frames[1]
    mode_dict["animate_secondary_mode_shape"] = animate_secondary_mode_shape
    mode_dict["fps"] = fps

    if a_type == "modeshape":
        mode_dict["or_pts"] = pts
        mode_dict["mesh"] = mesh
        mode_dict["scalars"] = True

    elif a_type == "object":
        mode_dict["objects_list"] = object_list

    return mode_dict


def CMIF(FRF, return_svector=False):
    """
    Calculates a CMIF parameter of input FRF matrix

    :param FRF: Input FRF matrix
    :type FRF: array(float)
    :param singular_vector: Return corresponding singular vectors
    :type singular_vector: bool, optional
    :return: CMIF parameters (singular values with or without left and right singular vectors)
    """
    _f = FRF.shape[0]
    val = np.min([FRF.shape[1], FRF.shape[2]])

    _S = np.zeros((_f, val))

    if return_svector:
        _U = np.zeros((_f, FRF.shape[1], FRF.shape[1]), dtype="complex")
        _V = np.zeros((_f, FRF.shape[2], FRF.shape[2]), dtype="complex")

    for i in range(_f):
        if return_svector:
            U, S, VH = np.linalg.svd(FRF[i, :, :], full_matrices=True, compute_uv=True)
            V = np.conj(VH).T
            _S[i, :] = S
            _U[i, :, :] = U
            _V[i, :, :] = V

        else:
            S = np.linalg.svd(FRF[i, :, :], full_matrices=True, compute_uv=False)
            _S[i, :] = S

    if return_svector:
        return _U, _S, _V
    else:
        return _S


def TSVD(matrix,reduction = 0):
    """
    Filters a FRF matrix  with a truncated singular value decomposition (TSVD) by removing the smallest singular values.

    :param matrix: Matrix to be filtered by singular value decomposition
    :type matrix: array(float)
    :param reduction: Number of singular values not taken into account by reconstruction of the matrix
    :type reduction: int, optional
    :return: Filtered matrix
    :rtype: array(float)
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
    Calculates rotational matrix based on the Euler-Rodrigues formula.

    :param axis: Axis of rotation
    :type axis: array(float)
    :param theta: Angle of rotation
    :type theta: float
    :return: Rotational matrix
    """
    t = expm(cross(eye(3), axis / norm(axis) * (theta)))
    return t


def angle(vector1, vector2):
    """
    Calculates angle of rotation between two 3D vectors.

    :param vector1: 3D vector
    :type vector1: array(float)
    :param vector2: 3D vector
    :type vector2: array(float)
    :return: angle
    """

    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(np.stack((v1_u[-2:], v2_u[-2:])))
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

    :param vec1: A 3D "source" vector
    :type vec1: array(float)
    :param vec2: A 3D "destination" vector
    :type vec2: array(float)
    :return: Rotational matrix which when applied to vec1, aligns it with vec2.
    """

    vec1 += np.random.random(3) / 1e20
    vec2 += np.random.random(3) / 1e20

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
    Returns the unit vector of input vector.

    :param vector: A 3D "source" vector
    :type vector: array(float)
    :return unit vector:
    """

    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Calculates angle of rotation between two 3D vectors.

    :param vector1: 3D vector
    :type vector1: array(float)
    :param vector2: 3D vector
    :type vector2: array(float)
    :return: angle
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def generate_channels_from_sensors(df):
    """
    Generates a set of channels based on the orientation of sensors. CUrrent implementation assumes that each sensor has
    three channels (i.e. tri-axial sensors).

    :param df: A DataFrame containing information on sensors
    :type df: pd.DataFrame
    :return: A DataFrame containing information on channels
    """

    columns_chann = ["Name", "Description", "Quantity", "Grouping",
                     "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2", "Direction_3"]
    df_ch = pd.DataFrame(columns=columns_chann)

    axes = ["x", "y", "z"]
    for s, angle in enumerate(df[["Orientation_1", "Orientation_2", "Orientation_3"]].to_numpy()):
        r = R.from_euler('xyz', angle, degrees=True)
        rot = r.as_matrix().T
        for i in range(3):
            data_chn = np.asarray([[df["Name"][s] + axes[i], df["Description"][s],
                                    None, df["Grouping"][s], df["Position_1"][s], df["Position_2"][s],
                                    df["Position_3"][s], rot[i][0], rot[i][1], rot[i][2]]])
            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df_ch = df_ch.append(df_row,ignore_index = True)

    return df_ch

def generate_sensors_from_channels(df):
    """
    Generates a set of sensors based on the supplied channel data. CUrrent implementation assumes that each sensor has
    three channels (i.e. tri-axial sensors).

    :param df: A DataFrame containing information on channels
    :type df: pd.DataFrame
    :return: A DataFrame containing information on sensors
    """

    columns_sen = ["Name", "Description", "Quantity", "Grouping",
                   "Position_1", "Position_2", "Position_3", "Orientation_1", "Orientation_2", "Orientation_3"]
    df_sen = pd.DataFrame(columns=columns_sen)

    for i in range(int(len(df)/3)):
        sen_or = df[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()[3 * (i):3 * (i + 1)]
        sen_pos = df[["Position_1", "Position_2", "Position_3"]].to_numpy()[3 * (i)]

        r = R.from_matrix(sen_or)
        r = r.inv()

        orient = r.as_euler('xyz', degrees=True)

        data_chn = np.asarray([["S"+str(i+1),None,None,None,sen_pos[0],sen_pos[1],sen_pos[2],orient[0],orient[1],orient[2]]])


        df_row = pd.DataFrame(data=data_chn, columns=columns_sen)
        df_sen = df_sen.append(df_row,ignore_index = True)

    return df_sen

def coh_on_FRF(FRF_matrix):
    """
    Evaluates a reciprocity on the whole FRF matrix.

    :param FRF_matrix: Matrix of FRFs [f,out,in]
    :type FRF_matrix: array(float)
    :return: A matrix of coherence criterion values on the reciprocal FRFs
    """

    _out = FRF_matrix.shape[1]
    _in = FRF_matrix.shape[2]

    coh_crit = np.zeros((_out, _in))

    for i in range(_out):
        for j in range(_in):
            coh_crit[i, j] = coh_frf(FRF_matrix[:, i, j], FRF_matrix[:, j, i])

    return coh_crit


def orient_in_global(mode, df_chn, df_acc):
    """
    Positions a response in 3D space based on the information of channel and sensor DataFrames

    :param mode: A mode shape or response to be animated
    :type mode: array(float)
    :param df_chn: A DataFrame containing information on channels
    :type df_chn: pd.DataFrame
    :param df_acc: A DataFrame containing information on sensors
    :type df_acc: pd.DataFrame
    :return: Oriented response in 3D
    """

    n_sen = len(df_acc)
    n_ax = 3

    empty = np.zeros((n_sen, n_ax), dtype=complex)

    _dir = df_chn[["Direction_1", "Direction_2", "Direction_3"]].to_numpy(dtype = float)

    for i in range(n_sen):
        for j in range(n_ax):
            sel = (i) * 3 + j
            empty[i, :] += _dir[sel:sel + 1, :].T @ np.asarray([mode[sel]])

    return empty

def orient_in_global_2(mode, df_imp):
    """
    Positions a response in 3D space based on the information of impact DataFrames (impact testing)

    :param mode: A mode shape or response to be animated
    :type mode: array(float)
    :param df_imp: A DataFrame containing information on impacts
    :type df_imp: pd.DataFrame
    :return: Oriented response in 3D
    """

    n_sen = len(df_imp)

    empty = np.zeros((n_sen, 3), dtype=complex)

    _dir = df_imp[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()
    for i in range(n_sen):
        sel = (i)
        empty[i, :] += _dir[sel:sel + 1, :].T @ np.asarray([mode[sel]])

    return empty


def MCC(mod):
    """
    Calculate a correlation coefficient MCC
    source: 10.1016/j.jsv.2013.01.039
    """
    Sxy = np.imag(mod).T @ np.real(mod)

    Sxx = np.real(mod).T @ np.real(mod)
    Syy = np.imag(mod).T @ np.imag(mod)
    MCC = Sxy ** 2 / (Sxx * Syy)
    return MCC


def MPC(mod, sel=0):
    """
    Calculate a modal phase collinearity coefficient MCC
    source: 10.1016/S0045-7949(03)00034-8
    """
    mod_t = mod

    _re = np.real(mod_t)
    _im = np.imag(mod_t)

    crr = _re.T @ _re
    cri = _re.T @ _im
    cii = _im.T @ _im

    MPC = ((cii - crr) ** 2 + 4 * cri ** 2) / (crr + cii) ** 2
    return MPC

def auralization(freq,FRF, load_case = None):
    """
    Auralization of FRFs, performs an IFFT and if the load case is supplied a convolution to obtain time response.

    :param freq: Frequency vector
    :type freq: array(float)
    :param FRF: Frequency Response Function
    :type FRF: array(float)
    :param load_case: Load vector
    :type load_case: array(float)
    :return: time vector, time response
    """

    s = np.fft.irfft(FRF).real
    fs = 1 / (freq[1] - freq[0])
    xt = np.linspace(0, fs, len(s), endpoint=False)

    if type(load_case) == type(np.asarray([])):
        s = (np.convolve(load_case, s, 'full').real)[:len(s)]

    return xt,s




def SSA_filter(time_series, no_sel, window_size=100):
    groups = [np.arange(0, no_sel), np.arange(no_sel, window_size)]
    transformer = SingularSpectrumAnalysis(window_size=window_size, groups=groups)

    X_new = transformer.transform(time_series.reshape(1, len(time_series)))

    signal = X_new[0, :]
    noise = X_new[1, :]

    return signal, noise


def SSA_evaluate(time_series, window_size=100):
    L = window_size
    N = len(time_series)
    K = N - L + 1

    # create trajectory matrix
    X_trajectory = np.column_stack([time_series[i:i + L] for i in range(0, K)])

    # compute singular values
    s = np.linalg.svd(X_trajectory, compute_uv=False)
    return s


def PRF(H1_main, n_sel):
    k = n_sel

    new_arr = H1_main.reshape(H1_main.shape[0], H1_main.shape[1] * H1_main.shape[2])
    u, s, vh = np.linalg.svd(new_arr, full_matrices=False)

    prfs = u @ np.diag(s)

    H1_rec = (u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]).reshape(H1_main.shape[0], H1_main.shape[1], H1_main.shape[2])

    return prfs, H1_rec

#if necessary, font properties can be changed
#def font():
#    font = "Sans Serif"
#    size = 12
#    
#    return {
#        "config" : {
#             "title": {
#                "font": font,
#                "fontSize": size
#            },
#             "axis": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             },
#             "header": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             },
#             "legend": {
#                "labelFont": font,
#                "titleFont": font,
#                "labelFontSize": size,
#                "titleFontSize": size
#             }
#        }
#    }
#
#alt.themes.register('font', font)
#alt.themes.enable('font')

def barchart(x, y, width=200, height=200, color='blue', title=''):
    """
    Wrapper function for plotting barcharts using Altair.
    :param x: The x coordinates of the bars.
    :type x: array
    :param y: The heights of the bars.
    :type y: array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param color: Color of the bars. CSS and HEX color codes supported.
    :type color: str, optional
    :param title: Title of the plot.
    :type title: str, optional
    """

    df = pd.DataFrame({'x':x, 'y':y})

    barchart = alt.Chart(df, title=title).mark_bar().encode(
            alt.X("x:O", axis=alt.Axis(title='No.')),
            alt.Y("y:Q", axis=alt.Axis(title='Value')),
            color=alt.value(color),
            tooltip=[alt.Tooltip('y:Q', format=".3f", title='Value')]
        ).properties(width=width, height=height)
    
    return barchart

def imshow(data, width=200, height=200, title='', cmap='turbo'):
    """
    Wrapper function for plotting images using Altair.
    :param data: Image data.
    :type x: 2D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param cmap: Colormap.
    :type cmap: str, optional
    """
    
    x, y = np.meshgrid(range(data.shape[1]), range(data.shape[0]))
    
    df = pd.DataFrame({'x': x.ravel(), 'y': y.ravel(), 'rec': data.ravel()})
    
    imshow = alt.Chart(df, title=title).mark_rect().encode(
            alt.X('x:O', axis=alt.Axis(title='Output DoFs')),
            alt.Y('y:O', axis=alt.Axis(title='Input DoFs')),
            color=alt.Color('rec:Q', scale=alt.Scale(scheme=cmap), legend=alt.Legend(title="Value")),
            tooltip=[alt.Tooltip('rec:Q', format=".3f", title='Value')]
        ).properties(width=width, height=height)
    
    return imshow

def plot_FRF(freq, FRF_data, width=500, height=400, circle_size=4E3):
    """
    Wrapper function for plotting Frequency Response Functions (magnitude and phase) using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param FRF_data: Admittance matrix to be displayed.
    :type FRF_data: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param circle_size: Size of the circles intendted for interactive selection of displayed FRFs.
    :type circle_size: int, optional
    """

    df = pd.DataFrame()
    _f = FRF_data.shape[0]
    for i in range(FRF_data.shape[1]):
        for j in range(FRF_data.shape[2]):

            df_temp = pd.DataFrame({"f" : freq, "A" : np.abs(FRF_data[:,i,j]),"ph" : np.angle(FRF_data[:,i,j]),"out" : [str(i)]*_f,\
                                    "in" : [str(j)]*_f,"out_in" : ['o'+str(i)+', i'+str(j)]*_f})
            df = df.append(df_temp)
            
    selector = alt.selection_multi(empty='all', fields=['out_in'])
    resize = alt.selection_interval(bind='scales')

    base = alt.Chart(df).properties(
        width=width,
        height=height
    ).add_selection(selector)

    points = base.mark_circle(size=circle_size).encode(
        alt.X('out', axis=alt.Axis(title='Output DoF')),
        alt.Y('in', axis=alt.Axis(title='Input DoF')),
        color=alt.condition(selector, 'out_in', alt.value('lightgray'), legend=None)
    )

    text = alt.Chart(df).mark_text(align='center', baseline='middle').encode(
        alt.X('out'),
        alt.Y('in'),
        text='out_in'
    )

    A = alt.Chart(df).mark_line().encode(
        alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
        alt.Y('A', axis=alt.Axis(title='Amplitude(Y)'), scale=alt.Scale(type='log',base=10)),
        color='out_in').properties(width=width,height=1/2*height).add_selection(
        resize
    ).transform_filter(
        selector
    )

    P = alt.Chart(df).mark_line().encode(
        alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
        alt.Y('ph', axis=alt.Axis(title='Phase(Y)')),
        color='out_in').properties(width=width,height=1/3*height).add_selection(
        resize
    ).transform_filter(
        selector
    )

    AP = alt.vconcat(A,P)

    return points + text | AP

def plot_frequency_response(freq, FR_data, width=500, height=400, labels=None):
    """
    Wrapper function for plotting frequency responses (magnitude and phase) using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param FR_data: Responses to be displayed.
    :type FRF_data: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param labels: Labels of the responses to be displayed in legend.
    :type labels: dict, optional
    """

    if labels==None:
        labels = []
        for k in range(int(FR_data.shape[1]*FR_data.shape[2])):
            labels.append('y%d'%(k+1))
#     else:
#         if int(y.shape[1]*y.shape[2]) == len(labels):
#             pass
#         else:
#             raise Exception('Labels dict does not match y shape.')

    df = pd.DataFrame()
    _f = FR_data.shape[0]
    k=0
    for i in range(FR_data.shape[1]):
        for j in range(FR_data.shape[2]):

            df_temp = pd.DataFrame({"f" : freq, "A" : np.abs(FR_data[:,i,j]),"ph" : np.angle(FR_data[:,i,j]),"out" : [str(i)]*_f,\
                                    "in" : [str(j)]*_f,"out_in" : [labels[k]]*_f})
            df = df.append(df_temp)
            k=k+1
    
    selection = alt.selection_multi(fields=['out_in'], bind='legend')
    resize = alt.selection_interval(bind='scales')

    A = alt.Chart(df).mark_line().encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('A', axis=alt.Axis(title='Amplitude'), scale=alt.Scale(type='log',base=10)),
            color=alt.Color('out_in', legend=alt.Legend(title="Click to highlight")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).properties(width=width,height=1/2*height
        ).add_selection(
            resize
        ).add_selection(
            selection
        )

    P = alt.Chart(df).mark_line().encode(
            alt.X("f", axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('ph', axis=alt.Axis(title='Phase')),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
            color='out_in').properties(width=width,height=1/3*height
        ).add_selection(
            resize
        ).add_selection(
            selection
        )

    AP = alt.vconcat(A,P)

    return AP

def comparison_plot(x, y, width=500, height=250, labels=None, title='', x_label='', y_label=''):
    """
    Wrapper function for plotting multiple responses using Altair.
    :param x: Data for x axis.
    :type freq: 1D array
    :param y: Responses to be displayed.
    :type y: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param labels: Labels of the responses to be displayed in legend.
    :type labels: dict, optional
    :param title: Title of the plot.
    :type title: str, optional
    :param x_label: Label of the x axis.
    :type x_label: str, optional
    :param y_label: Label of the y axis.
    :type y_label: str, optional
    """
    
    if labels==None:
        labels = []
        for k in range(int(y.shape[1]*y.shape[2])):
            labels.append('y%d'%(k+1))
#     else:
#         if int(y.shape[1]*y.shape[2]) == len(labels):
#             pass
#         else:
#             raise Exception('Labels dict does not match y shape.')

    df = pd.DataFrame()
    _x = y.shape[0]
    k=0
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):

            df_temp = pd.DataFrame({"x" : x, "y" : y[:,i,j],"out" : [str(i)]*_x,\
                                    "in" : [str(j)]*_x,"out_in" : [labels[k]]*_x})
            df = df.append(df_temp)
            k=k+1
    
    selection = alt.selection_multi(fields=['out_in'], bind='legend')
    resize = alt.selection_interval(bind='scales')
    
    A = alt.Chart(df, title=title).mark_line().encode(
            alt.X("x", axis=alt.Axis(title=x_label)),
            alt.Y('y', axis=alt.Axis(title=y_label), scale=alt.Scale()),
            color=alt.Color('out_in', legend=alt.Legend(title="Click to highlight")),
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))).properties(width=width,height=height
        ).add_selection(
            resize
        ).add_selection(
            selection
        )
        
    return A

def plot_coh(freq, coh_data, width=500, height=200, opacity=0.2, color='blue', title=''):
    """
    Wrapper function for plotting frequency dependable coherence using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Coherence data to be displayed.
    :type coh_data: 1D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param opacity: Opacity of the Area Fill between x axis and coherence data.
    :type opacity: int, optional
    :param color: Color of the line. CSS and HEX color codes supported.
    :type color: str, optional
    :param title: Title of the plot.
    :type title: str, optional
    """

    df = pd.DataFrame()
    _f = coh_data.shape[0]

    df_temp = pd.DataFrame({"f" : freq, "coh" : np.abs(coh_data),"out" : [str(0)]*_f, "avg_coh" : [str(np.round(np.average(coh_data),3))]*_f})
    df = df.append(df_temp)
            
    resize = alt.selection_interval(bind='scales')

    A = alt.Chart(df, title=title).mark_area(
        line={'color':'out'},
        color='out',
        opacity=opacity
    ).encode(
        alt.X('f', axis=alt.Axis(title='Frequency [Hz]')),
        alt.Y('coh', axis=alt.Axis(title='Coherence [/]')),
        color=alt.value(color)
    ).add_selection(
        resize
    ).properties(
        width=width,
        height=height
    )

    return A

def plot_coh_group(freq, coh_data, width=500, height=250, circle_size=4E3, opacity=0):
    """
    Wrapper function for plotting multiple frequency dependable coherence using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Coherence data to be displayed.
    :type coh_data: 3D array
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    :param circle_size: Size of the circles intendted for interactive selection of displayed FRFs.
    :type circle_size: int, optional
    :param opacity: Opacity of the Area Fill between x axis and coherence data.
    :type opacity: int, optional
    """

    df = pd.DataFrame()
    _f = coh_data.shape[0]
    for i in range(coh_data.shape[1]):
        for j in range(coh_data.shape[2]):

            df_temp = pd.DataFrame({"f" : freq, "coh" : np.abs(coh_data[:,i,j]),"out" : [str(i)]*_f,\
                                    "in" : [str(j)]*_f,"out_in" : [str(i)+str(j)]*_f, "avg_coh" : [str(np.round(np.average(coh_data[:,i,j]),3))]*_f})
            df = df.append(df_temp)
            
    selector = alt.selection_multi(empty='all', fields=['out_in'])
    resize = alt.selection_interval(bind='scales')

    base = alt.Chart(df).properties(
        width=width,
        height=height
    ).add_selection(selector)

    points = base.mark_circle().encode(
        alt.X('out', axis=alt.Axis(title='Output DoF')),
        alt.Y('in', axis=alt.Axis(title='Input DoF')),
        size=alt.Size('avg_coh', scale=alt.Scale(range=[circle_size/2, circle_size]), legend=None),
        color=alt.condition(selector, 'out_in', alt.value('lightgray'), legend=None)
    )

    text = alt.Chart(df).mark_text(align='center', baseline='middle').encode(
        alt.X('out'),
        alt.Y('in'),
        text='avg_coh'
    )

    A = base.mark_area(
        line={'color':'out_in'},
        color='out_in',
        opacity=opacity
    ).encode(
        alt.X('f', axis=alt.Axis(title='Frequency [Hz]')),
        alt.Y('coh', axis=alt.Axis(title='Coherence [/]')),
        color='out_in'
    ).transform_filter(
        selector
    ).add_selection(
        resize
    )

    return points + text | A

def tranfer_path(freq, u3_partial, width=700, height=150):
    """
    Wrapper function for plotting graphical presentation of tranfer paths contribution using Altair.
    :param freq: Frequency vector for x axis.
    :type freq: 1D array
    :param coh_data: Transfer paths contributions to be displayed.
    :type coh_data: 2D array (frequency X no_of_transfer_paths)
    :param width: Width of the plot.
    :type width: int, optional
    :param height: Height of the plot.
    :type height: int, optional
    """

    df = pd.DataFrame()
    _f = u3_partial.shape[0]
    for i in range(u3_partial.shape[1]):
        DoFs = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']        
        df_temp = pd.DataFrame({"f" : freq, "A" : np.log(np.abs(u3_partial[:,i])),"out" : [DoFs[i]]*_f})
        df = df.append(df_temp)
    
    A = alt.Chart(df).mark_rect().encode(
            alt.X('f:O', axis=alt.Axis(title='Frequency [Hz]')),
            alt.Y('out:O', axis=alt.Axis(title='DoF')),
            color=alt.Color('A:Q', scale=alt.Scale(scheme="turbo"), legend=None),
            tooltip=[alt.Tooltip('f:Q', title='Frequency')]
        ).configure_view(
            strokeWidth=0
        ).properties(width=width, height=height)
    
    return A