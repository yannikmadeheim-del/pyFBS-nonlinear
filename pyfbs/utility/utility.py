import numpy as np
from numpy import cross, eye
from scipy.linalg import expm, norm
import pandas as pd
from scipy.spatial.transform import Rotation
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
        _mode = mode_shape_vec[:, i]
        z = np.arctan(
            np.average(
                np.imag(_mode) / np.real(_mode), weights=np.abs(_mode) ** 1e4
            )
        )

        _n[:, i] = _mode * (np.cos(-1 * z) + 1j * np.sin(-1 * z))
    return _n


def modeshape_scaling_driving_point(mode_shape_vec, driving_point, sync=True):
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
        _mode[:, i] = _mode[:, i] / np.sqrt(mode_shape_vec[driving_point, i])

    if sync:
        _mode = modeshape_sync_lstsq(_mode)

    return _mode


def mcf(mod):
    """
    Calculate Mode Complexity Factor (MCF)

    :param mod: Mode shape
    :type mod: array(float)
    :return: Mode complexity factor
    """
    sxx = np.real(mod).T @ np.real(mod)
    syy = np.imag(mod).T @ np.imag(mod)
    sxy = np.real(mod).T @ np.imag(mod)
    _mcf = 1 - ((sxx - syy) ** 2 + 4 * sxy**2) / ((sxx + syy) ** 2)
    return _mcf


def flatten_frfs(frf):
    """
    Flattens input FRF matrix frf from shape (out,in,freq) in (out x in,freq)

    :param frf: Matrix of FRFs [out,in,f]
    :type frf: array(float)
    :return:  Matrix of FRFs [out x in,f]
    """
    new = np.zeros((frf.shape[0] * frf.shape[1], frf.shape[2]), dtype=complex)

    _len = frf.shape[1]
    for i in range(frf.shape[0]):
        new[_len * i : _len * (i + 1), :] = frf[i, :, :]

    return new


def unflatten_modes(_modes_acc, frf):
    """
    Unflattens mode shapes based on the shape of the input FRF matrix [out x in] in [out, in]

    :param _modes_acc: Mode shape [out x in]
    :type _modes_acc: array(float)
    :param frf:
    :return: Unflattened mode shape [out, in]
    """
    new_mode = np.zeros(
        (frf.shape[0], frf.shape[1], _modes_acc.shape[1]), dtype=complex
    )
    _len = frf.shape[1]
    for i in range(frf.shape[0]):
        new_mode[i, :, :] = _modes_acc[i * _len : (i + 1) * _len, :]
    return new_mode


def mode_animation(
    mode_shape,
    scale,
    no_points=60,
    no_of_repetitions=2,
    abs_scale=True,
    secondary_mode_shape=None,
    animate_secondary_mode_shape=False,
):
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
        if mode_shape.ndim == 2:
            ann = np.zeros(
                (mode_shape.shape[0], mode_shape.shape[1], int(no_points))
            )
            ann_secondary = np.zeros((mode_shape.shape[0], int(no_points)))
        else:
            raise ValueError(
                "Parameter mode_shape must be a 2D vector, where the first "
                "dimension presents all nodes and the second dimension 3 "
                "coordinates (x, y, z)."
            )
    else:
        raise ValueError(
            "To animate mode shape, a parameter mode_shape must be defined in form of 2D numpy array."
        )

    for g, _t in enumerate(
        np.linspace(0, int(no_of_repetitions), int(no_points))
    ):
        ann[:, :, g] = np.real(mode_shape) * np.cos(2 * np.pi * _t) - np.imag(
            mode_shape
        ) * np.sin(2 * np.pi * _t)
        if animate_secondary_mode_shape:
            if isinstance(secondary_mode_shape, np.ndarray):
                if secondary_mode_shape.ndim == 1:
                    ann_secondary[:, g] = np.real(
                        secondary_mode_shape
                    ) * np.cos(2 * np.pi * _t) - np.imag(
                        secondary_mode_shape
                    ) * np.sin(
                        2 * np.pi * _t
                    )
                else:
                    raise ValueError(
                        "Parameter secondary_mode_shape must be 1D vector."
                    )
            else:
                raise ValueError(
                    "To animate secondary mode shape, a parameter secondary_mode_shape must be defined in form of 1D numpy array."
                )

    if abs_scale:
        ann = ann / np.max(ann) * scale
    else:
        ann = ann * scale
    return ann, ann_secondary


def mac(phi_1, phi_2, output_type='matrix'):
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
        phi_1 = phi_1[:, np.newaxis]
    if phi_2.ndim == 1:
        phi_2 = phi_2[:, np.newaxis]

    mac_mat = (
        np.abs(np.einsum('ri,ik->rk', np.conj(phi_1).T, phi_2)) ** 2
        / (
            np.einsum('ri,ir->r', np.conj(phi_1).T, phi_1)[:, np.newaxis]
            * np.einsum('ri,ir->r', np.conj(phi_2).T, phi_2)
        )
    ).real
    if output_type == 'matrix':
        return mac_mat
    if output_type == 'diagonal':
        return np.diagonal(mac_mat)
    else:
        raise Exception('Unknown output type.')


def coh_frf(y_1, y_2, return_average=True):
    """
    Calculates values of coherence between two FRFs.

    :param y_1: FRF 1
    :type y_1: array(float)
    :param y_2: FRF 2
    :type y_2: array(float)
    :return: coherence criterion
    """
    return coh(y_1, y_2, return_average=return_average)


def coh(x, y, return_average=False):
    """
    Compute coherence of (complex or real-valued) signals
    x and y.

    Parameters
    ----------
    x, y : numpy.ndarray with identical shapes or at least
        shapes such that (x + y) returns a valid result.
    """
    coh_xy = (
        ((x + y) * (x + y).conj()) / (2 * (x.conj() * x + y.conj() * y))
    ).real
    if return_average:
        return np.mean(coh_xy)
    else:
        return coh_xy


def lac(x, y, return_average=False):
    """
    Compute LAC (local assurance criterion) of (complex or
    real-valued) signals x and y.
    -----------
    Parameters:
    -----------
    x, y : numpy.ndarray with identical shapes or at least
        shapes such that (x + y) returns a valid result.
    """

    lac_xy = (
        (2 * np.abs(x.conj() * y)) / ((x.conj() * x) + (y.conj() * y))
    ).real
    if return_average:
        return np.mean(lac_xy)
    else:
        return lac_xy


def dict_animation(
    _modeshape,
    a_type,
    mesh=None,
    pts=None,
    fps=30,
    r_scale=10,
    no_points=60,
    no_of_repetitions=2,
    object_list=None,
    abs_scale=True,
    secondary_mode_shape=None,
    animate_secondary_mode_shape=False,
):
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
    :param animate_secondary_mode_shape: If ``True``, secondary mode shape
        will be animated, if ``False`` still only initial mode shape will be
        animated
    :type animate_secondary_mode_shape: bool, optional
    :return: Dictionary of parameters for mode shape animation
    """
    mode_dict = dict()

    mode_animation_frames = mode_animation(
        _modeshape,
        r_scale,
        no_points=no_points,
        no_of_repetitions=no_of_repetitions,
        abs_scale=abs_scale,
        secondary_mode_shape=secondary_mode_shape,
        animate_secondary_mode_shape=animate_secondary_mode_shape,
    )
    mode_dict["animation_pts"] = mode_animation_frames[0]
    mode_dict["animation_pts_secondary"] = mode_animation_frames[1]
    mode_dict["animate_secondary_mode_shape"] = animate_secondary_mode_shape
    mode_dict["fps"] = fps

    if a_type == "modeshape":
        mode_dict["or_pts"] = pts.copy() if pts is not None else None
        mode_dict["mesh"] = mesh
        mode_dict["scalars"] = True

    elif a_type == "object":
        mode_dict["objects_list"] = object_list

    return mode_dict


def cmif(frf, return_svector=False):
    """
    Calculates a CMIF parameter of input FRF matrix

    :param frf: Input FRF matrix
    :type frf: array(float)
    :param singular_vector: Return corresponding singular vectors
    :type singular_vector: bool, optional
    :return: CMIF parameters (singular values with or without left and right singular vectors)
    """
    _f = frf.shape[0]
    val = np.min([frf.shape[1], frf.shape[2]])

    _s = np.zeros((_f, val))

    if return_svector:
        _u = np.zeros((_f, frf.shape[1], frf.shape[1]), dtype=complex)
        _v = np.zeros((_f, frf.shape[2], frf.shape[2]), dtype=complex)

    for i in range(_f):
        if return_svector:
            u, s, vh = np.linalg.svd(
                frf[i, :, :], full_matrices=True, compute_uv=True
            )
            v = np.conj(vh).T
            _s[i, :] = s
            _u[i, :, :] = u
            _v[i, :, :] = v

        else:
            s = np.linalg.svd(
                frf[i, :, :], full_matrices=True, compute_uv=False
            )
            _s[i, :] = s

    if return_svector:
        return _u, _s, _v
    else:
        return _s


def _tsvd(matrix, reduction=0):
    """
    Filters a FRF matrix  with a truncated singular value decomposition (TSVD) by removing the smallest singular values.

    :param matrix: Matrix to be filtered by singular value decomposition
    :type matrix: array(float)
    :param reduction: Number of singular values not taken into account by reconstruction of the matrix
    :type reduction: int, optional
    :return: Filtered matrix
    :rtype: array(float)
    """
    u, s, vh = np.linalg.svd(matrix)
    kk = s.shape[1] - reduction
    uk = u[:, :, :kk]
    sk = np.zeros((matrix.shape[0], kk, kk))

    for i in range(matrix.shape[0]):
        sk[i] = np.diag(s[i, :kk])
    vk = vh[:, :kk, :]

    return uk @ sk @ vk


def tsvd(
    matrix: np.ndarray,
    trunc: int = 0,
    mode: str = 'remove',
    return_components: bool = False,
):
    """
    Filters a FRF matrix  with a truncated singular value decomposition (TSVD)
    by removing the smallest singular values.

    :param matrix: Matrix to be filtered by singular value decomposition
    :type matrix: array(float)
    :param trunc: Number of singular values not taken into account by
        reconstruction of the matrix
    :type reduction: int, optional
    :param mode: Mode of operation, either 'remove' or 'keep'. Removal of
        smallest or retaining of largest `trunc` singular values
    :type mode: str
    :param return_components: If True, the right and left singular vectors
        and singular values are returned, otherwise reconstruction of the
        original matrix using the truncated svd is returned.
    :type return_components: bool
    :return: Filtered matrix
    :rtype: array(float)
    """
    u, s, vh = np.linalg.svd(matrix, full_matrices=False)
    if mode == 'remove':
        n = s.shape[-1] - trunc
    elif mode == 'keep':
        n = trunc
    else:
        raise ValueError("`mode` must be 'remove' or 'keep'")
    if n < 0:
        raise ValueError(
            "Reduction value is higher than the number of singular values"
        )
    if return_components:
        return u[..., :n], s[..., :n], vh[..., :n, :]
    else:
        return (u[..., :n] * s[..., None, :n]) @ vh[..., :n, :]


def tpinv(a: np.ndarray, trunc: int | None = None, mode: str = 'remove'):
    """
    Compute Moore-Penrose pseudo inverse using SVD with or without truncation.

    ----------
    Parameters
    ----------
    a : numpy.ndarray
        Array to be inverted.
    trunc : int or None
        Cutoff for singular values. If None, the inverse is calculated without
        truncation. Trunc specifies the amount of truncation prior to the
        inversion. If `mode` is 'remove', `trunc` specifies the number of
        smallest singular values to be set to zero. If `mode` is 'keep',
        `trunc` specifies the number of largest singular values to retain.
    mode : str
        Mode of operation, either 'remove' or 'keep'.
    """
    tu, ts, tvh = tsvd(a, trunc=trunc, mode=mode, return_components=True)
    return np.swapaxes(tvh.conj(), -2, -1) @ np.swapaxes(
        tu.conj() * 1 / ts[..., None, :], -2, -1
    )


def rotation_matrix(axis, theta):
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


def rotation_matrix_from_vectors(vec1, vec2, tol=1e-8):
    """
    Robust rotation matrix between two vectors.

    :param vec1: Source vector (3D)
    :param vec2: Target vector (3D)
    :param tol: Numerical tolerance
    :return: Rotation matrix aligning vec1 to vec2
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)

    # Handle parallel/anti-parallel cases
    if np.allclose(a, b, atol=tol):
        return np.eye(3)
    if np.allclose(a, -b, atol=tol):
        # 180° rotation around perpendicular axis
        axis = np.array([a[1], -a[0], 0])
        if np.linalg.norm(axis) < tol:  # Handle [0,0,z]
            axis = np.array([a[2], 0, -a[0]])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    # General case (Rodrigues' formula)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)

    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


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

    columns_chann = [
        "Name",
        "Description",
        "Quantity",
        "Grouping",
        "Position_1",
        "Position_2",
        "Position_3",
        "Direction_1",
        "Direction_2",
        "Direction_3",
    ]
    df_ch = pd.DataFrame(columns=columns_chann)

    axes = ["x", "y", "z"]
    for s, angle in enumerate(
        df[["Orientation_1", "Orientation_2", "Orientation_3"]].to_numpy()
    ):
        r = Rotation.from_euler('xyz', angle, degrees=True)
        rot = r.as_matrix().T
        for i in range(3):
            data_chn = np.asarray(
                [
                    [
                        str(df["Name"][s]) + axes[i],
                        df["Description"][s],
                        None,
                        df["Grouping"][s],
                        df["Position_1"][s],
                        df["Position_2"][s],
                        df["Position_3"][s],
                        rot[i][0],
                        rot[i][1],
                        rot[i][2],
                    ]
                ]
            )
            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df_ch = pd.concat([df_ch, df_row], ignore_index=True)

    return df_ch


def generate_sensors_from_channels(df):
    """
    Generates a set of sensors based on the supplied channel data. CUrrent implementation assumes that each sensor has
    three channels (i.e. tri-axial sensors).

    :param df: A DataFrame containing information on channels
    :type df: pd.DataFrame
    :return: A DataFrame containing information on sensors
    """

    columns_sen = [
        "Name",
        "Description",
        "Quantity",
        "Grouping",
        "Position_1",
        "Position_2",
        "Position_3",
        "Orientation_1",
        "Orientation_2",
        "Orientation_3",
    ]
    df_sen = pd.DataFrame(columns=columns_sen)

    for i in range(int(len(df) / 3)):
        sen_or = df[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()[
            3 * (i) : 3 * (i + 1)
        ]
        sen_pos = df[["Position_1", "Position_2", "Position_3"]].to_numpy()[
            3 * (i)
        ]

        r = Rotation.from_matrix(sen_or)
        r = r.inv()
        orient = r.as_euler('xyz', degrees=True)

        data_chn = np.asarray(
            [
                [
                    "S" + str(i + 1),
                    None,
                    None,
                    None,
                    sen_pos[0],
                    sen_pos[1],
                    sen_pos[2],
                    orient[0],
                    orient[1],
                    orient[2],
                ]
            ]
        )
        df_row = pd.DataFrame(data=data_chn, columns=columns_sen)
        df_sen = pd.concat([df_sen, df_row], ignore_index=True)

    return df_sen


def generate_vp_from_position(df):
    """
    Generates a DataFrame for full-DoF VP based on VP position determined using interactive positioning.
    VP is orientated in the direction of the global coordinate system.

    :param df: A DataFrame containing VPs positions
    :type df: pd.DataFrame
    :return df_vp: A Dataframe containg full DoF VPs channels
    :return df_vpref: A DataFrame containing full DoF VPs reference channels
    """

    columns_vp = [
        "Name",
        "Description",
        "Quantity",
        "Grouping",
        "Position_1",
        "Position_2",
        "Position_3",
        "Direction_1",
        "Direction_2",
        "Direction_3",
    ]

    desc_u = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz']
    desc_f = ['fx', 'fy', 'fz', 'mx', 'my', 'mz']

    quantity_u = np.tile(
        np.repeat(['Acceleration', 'Rotational Acceleration'], 3), 1
    )
    quantity_f = np.tile(np.repeat(['Force', 'Moment'], 3), 1)

    orientation = np.vstack((np.eye(3), np.eye(3)))

    df_vp = pd.DataFrame(columns=columns_vp)
    df_vpref = pd.DataFrame(columns=columns_vp)

    for i in range(df.shape[0]):

        for j in range(6):
            data_vp = np.asarray(
                [
                    [
                        df.iloc[i]['Name'],
                        desc_u[j],
                        quantity_u[j],
                        i + 1,
                        df.iloc[i]['Position_1'],
                        df.iloc[i]['Position_2'],
                        df.iloc[i]['Position_3'],
                        orientation[j][0],
                        orientation[j][1],
                        orientation[j][2],
                    ]
                ]
            )
            data_vpref = np.asarray(
                [
                    [
                        df.iloc[i]['Name'],
                        desc_f[j],
                        quantity_f[j],
                        i + 1,
                        df.iloc[i]['Position_1'],
                        df.iloc[i]['Position_2'],
                        df.iloc[i]['Position_3'],
                        orientation[j][0],
                        orientation[j][1],
                        orientation[j][2],
                    ]
                ]
            )

            df_row_vp = pd.DataFrame(data=data_vp, columns=columns_vp)
            df_row_vpref = pd.DataFrame(data=data_vpref, columns=columns_vp)

            df_vp = pd.concat([df_vp, df_row_vp], ignore_index=True).apply(
                pd.to_numeric, errors='ignore'
            )
            df_vpref = pd.concat(
                [df_vpref, df_row_vpref], ignore_index=True
            ).apply(pd.to_numeric, errors='ignore')

    return df_vp, df_vpref


def reciprocity(frf_matrix):
    """
    Evaluates a reciprocity on the whole FRF matrix.

    :param frf_matrix: Matrix of FRFs [f,out,in]
    :type frf_matrix: array(float)
    :return: A matrix of coherence criterion values on the reciprocal FRFs
    """

    _out = frf_matrix.shape[1]
    _in = frf_matrix.shape[2]

    coh_crit = np.zeros((_out, _in))

    for i in range(_out):
        for j in range(_in):
            coh_crit[i, j] = coh_frf(frf_matrix[:, i, j], frf_matrix[:, j, i])

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

    _dir = df_chn[["Direction_1", "Direction_2", "Direction_3"]].to_numpy(
        dtype=float
    )

    for i in range(n_sen):
        for j in range(n_ax):
            sel = (i) * 3 + j
            empty[i, :] += _dir[sel : sel + 1, :].T @ np.asarray([mode[sel]])

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
        sel = i
        empty[i, :] += _dir[sel : sel + 1, :].T @ np.asarray([mode[sel]])

    return empty


def mcc(mod):
    """
    Calculate a correlation coefficient MCC
    source: 10.1016/j.jsv.2013.01.039
    """
    s_xy = np.imag(mod).T @ np.real(mod)

    s_xx = np.real(mod).T @ np.real(mod)
    s_yy = np.imag(mod).T @ np.imag(mod)
    _mcc = s_xy**2 / (s_xx * s_yy)
    return _mcc


def mpc(mod, sel=0):
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

    _mpc = ((cii - crr) ** 2 + 4 * cri**2) / (crr + cii) ** 2
    return _mpc


def auralization(freq, frf, load_case=None):
    """
    Auralization of FRFs, performs an IFFT and if the load case is supplied a convolution to obtain time response.

    :param freq: Frequency vector
    :type freq: array(float)
    :param frf: Frequency Response Function
    :type frf: array(float)
    :param load_case: Load vector
    :type load_case: array(float)
    :return: time vector, time response
    """

    s = np.fft.irfft(frf)
    dt = 1 / (freq[1] - freq[0])
    xt = np.linspace(0, dt, len(s), endpoint=True)
    if type(load_case) == type(np.asarray([])):
        s = (np.convolve(load_case, s, 'full').real)[: len(load_case)]
        xt = np.linspace(0, dt, len(load_case), endpoint=False)

    return xt, s


def ssa_filter(time_series, no_sel, window_size=100):
    groups = [np.arange(0, no_sel), np.arange(no_sel, window_size)]
    transformer = SingularSpectrumAnalysis(
        window_size=window_size, groups=groups
    )

    x_new = transformer.transform(time_series.reshape(1, len(time_series)))

    signal = x_new[0, :]
    noise = x_new[1, :]

    return signal, noise


def ssa_evaluate(time_series, window_size=100):
    L = window_size
    N = len(time_series)
    K = N - L + 1

    # create trajectory matrix
    x_trajectory = np.column_stack(
        [time_series[i : i + L] for i in range(0, K)]
    )

    # compute singular values
    s = np.linalg.svd(x_trajectory, compute_uv=False)
    return s


def prf(h1_main, n_sel):
    k = n_sel

    new_arr = h1_main.reshape(
        h1_main.shape[0], h1_main.shape[1] * h1_main.shape[2]
    )
    u, s, vh = np.linalg.svd(new_arr, full_matrices=False)

    prfs = u @ np.diag(s)

    h1_rec = (u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]).reshape(
        h1_main.shape[0], h1_main.shape[1], h1_main.shape[2]
    )

    return prfs, h1_rec


def ods_frf(roving_responses, reference):
    '''
    roving_responses: roving responses not phase matched shaped in a form of (frequency X no. of responses)

    reference: refernce measurement in a form of (frequency)


    return ODS_FRFs: responses phase matched in a form of (frequency X no. of responses)
    '''

    gxx = np.einsum('ij,ij->ij', roving_responses, np.conj(roving_responses))
    gxy = np.einsum('ij,j->ij', roving_responses, np.conj(reference))

    ods_frfs = np.einsum('ij,ij->ij', np.sqrt(gxx), gxy / np.abs(gxy))

    return ods_frfs


def ods_frf_averaging(roving_responses, reference, no_of_avg):
    '''
    roving_responses: roving responses not phase matched shaped in a form of (samples X no. of responses)

    reference: refernce measurement in a form of (samples)


    return ODS_FRFs: responses phase matched in a form of (frequency X no. of responses)
    '''
    N = reference.shape[0]
    n = int(N / no_of_avg)
    gxx = np.zeros((int(n / 2) + 1, roving_responses.shape[1]), dtype=complex)
    gxy = np.zeros((int(n / 2) + 1, roving_responses.shape[1]), dtype=complex)

    for i in range(no_of_avg):
        roving_responses_ = np.fft.rfft(
            roving_responses[i * n : (i + 1) * n, :], axis=0
        )
        reference_ = np.fft.rfft(reference[i * n : (i + 1) * n])

        gxx += (
            np.einsum(
                'ij,ij->ij', roving_responses_, np.conj(roving_responses_)
            )
            / no_of_avg
        )
        gxy += (
            np.einsum('ij,i->ij', roving_responses_, np.conj(reference_))
            / no_of_avg
        )

    _ods_frfs = np.einsum('ij,ij->ij', np.sqrt(gxx), gxy / np.abs(gxy))

    return _ods_frfs


# if necessary, font properties can be changed
# def font():
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
# alt.themes.register('font', font)
# alt.themes.enable('font')

__all__ = [
    'modeshape_sync_lstsq',
    'modeshape_scaling_driving_point',
    'mcf',
    'flatten_frfs',
    'unflatten_modes',
    'mode_animation',
    'mac',
    'coh_frf',
    'coh',
    'lac',
    'dict_animation',
    'cmif',
    '_tsvd',
    'tsvd',
    'tpinv',
    'rotation_matrix',
    'angle',
    'rotation_matrix_from_vectors',
    'unit_vector',
    'angle_between',
    'generate_channels_from_sensors',
    'generate_sensors_from_channels',
    'generate_vp_from_position',
    'reciprocity',
    'orient_in_global',
    'orient_in_global_2',
    'mcc',
    'mpc',
    'auralization',
    'ssa_filter',
    'ssa_evaluate',
    'prf',
    'ods_frf',
    'ods_frf_averaging',
]
