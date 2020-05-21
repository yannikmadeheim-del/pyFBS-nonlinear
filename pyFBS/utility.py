import pyuff
from numpy import ndarray
import matplotlib.pyplot as plt
import math as mt
import cmath as cmt
import numpy as np
import math



def response_sync_lstsq(response_vec):
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


def eulerAnglesToRotationMatrix(theta):
    """
    Creates a rotational matrix based on the euler angles
    :param theta:
    :return:
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), math.sin(theta[0])],
                    [0, -math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, -math.sin(theta[1])],
                    [0, 1, 0],
                    [math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), math.sin(theta[2]), 0],
                    [-math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_x, np.dot(R_y, R_z))

    return R


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
    fig = plt.figure(figsize = (3,3))
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

    coh = (h_num + h_exp)

    coh = np.abs(vector((h_num + h_exp), (h_numk + h_expk))) / 2 / (vector(h_numk, h_num) + vector(h_expk, h_exp))
    coh_abs = np.abs(coh)

    if check:
        return coh, coh_abs
    else:
        return coh_abs

if __name__ == '__main__':
    print("Test: Cat!")