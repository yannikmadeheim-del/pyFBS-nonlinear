import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction as TF


def osi_var1(u_signal, f_signal, fs, nt, nt_cancel=5, t=1):
    '''
    Computes the Frequency Response Function (FRF) of an operating system
    using the Operational System Identification (OSI) method.

    Parameters
    ----------
    u_signal :  ndarray, float
                sampled response signal
    f_signal :  ndarray, float
                sampled force/ excitation signal
    fs :        scalar, int
                sampling frequency, number of data samples aquired per second
    t  :        scalar, float, optional
                measurement block length in seconds, time interval of the FFT
    nt_cancel : scalar, int
                number of measurement blocks that are dropped in the beginning,
                time that the sytem needs to reach the steady state
    nt :        scalar, int
                number of measurement blocks used for the OSI

    Returns
    -------
    y_osi   : ndarray, complex
              FRF determined using the OSI-method
    freq    : ndarray, float
              discrete frequencies associated to the FFT
    U_avg   : ndarray, complex
              FFT of the averaged response signal
    F_avg   : ndarray, complex
              FFT of the averaged force signal

    References:
    -----------
    [1] de Klerk, D. "Determination of an Operating Systems' Receptance FRF
        Data (Continued)." In: proceedings on the 26th  International Modal
        Analysis Conference (IMAC), Orlando, FL

    [2] de Klerk, D. "Dynamic Response Characterization of Complex Systems
        through Operational Identification and Dynamic Substructuring",
        Dissertation, 2009

    See Also:
    --------

    Examples:
    --------


    '''

    # maybe some helpful relationships:

    # time domain:
    t_tot = nt * t  # time interval used for the OSI
    t_tot_min = (
        t_tot + nt_cancel * t
    )  # minimum signal length, the first five measurement blocks are dropped
    N = t * fs  # total number of data samples per measurement block
    delta_t = 1 / fs  # = T/N ; sampling time
    # frequency domain:
    F_max = fs / 2.0  # bandwidth, highest frequency captured in the FFT
    delta_f = (
        1 / t
    )  # = F_max/SL ; frequency resolution, distance between spectral lines
    SL = N // 2  # spectral lines, number of samples in freq. domain

    # check if signal is long enough:
    if (t_tot_min * N) > len(u_signal):
        raise ValueError(
            'Signal is too short. Provide a longer signal or decrease \
                          the number of measurement blocks nt or measurement block length T.'
        )

    # averaging:

    # the OSI-method can be realized in two different ways: FFT of all measure-
    # ment blocks and then averaging in frequency domain or averaging in time
    # domain and then FFT of the averaged time signal. Here the 2nd variant is
    # chosen, because averaging of the time signal and performing one FFT is
    # much more cheaper than performing nt FFTs and averaging in frequency domain.

    u_avg = np.zeros(N)  # initialization of the averaged time domain data
    f_avg = np.zeros(N)
    err_u = np.zeros(nt)  # initialization of convergence vectors
    err_f = np.zeros(nt)

    for i in range(0, nt):
        start = (
            i + nt_cancel
        ) * N  # first and last index of the i-th measurement
        end = (
            i + 1 + nt_cancel
        ) * N  # block; the first nt_cancel measurement blocks are dropped
        u_block = u_signal[start:end]
        f_block = f_signal[start:end]
        u_avg += u_block
        f_avg += f_block

        # determination of a convergence measure for the averaged blocks:
        u_abs_i = (
            1 / (i + 1) * u_avg
        )  # total value / norm of i averaged measurement blocks
        u_abs_i = np.sum(np.abs(u_abs_i))
        f_abs_i = 1 / (i + 1) * f_avg
        f_abs_i = np.sum(np.abs(f_abs_i))

        if (
            i == 0
        ):  # the first measurement block is used as reference for normalization
            u_ref = u_abs_i
            f_ref = f_abs_i

        err_u[i] = u_abs_i / u_ref  # relative deviation, normalization
        err_f[i] = f_abs_i / f_ref

    u_avg = u_avg / nt
    f_avg = f_avg / nt

    # transformation into frequency domain:
    U_avg = fft(u_avg) / N
    U_avg = np.hstack((U_avg[0], 2 * U_avg[1:SL]))
    F_avg = fft(f_avg) / N
    F_avg = np.hstack((F_avg[0], 2 * F_avg[1:SL]))

    y_osi = np.divide(U_avg, F_avg)
    freq = np.arange(0, SL) * delta_f

    # plots:

    #    # convergence plot of the averaging process
    #    plt.semilogx(np.arange(1,nt+1), err_u, 'b', label='response u')
    #    plt.semilogx(np.arange(1,nt+1), err_f, 'r', label='force f')
    #    plt.grid()
    #    plt.ylim(0,1)
    #    plt.legend(loc='lower left')
    #    plt.xlabel("number of averaged measurement blocks [-]")
    #    plt.ylabel("convergence [-]")
    #    plt.show()

    #    # amplitudes of the FRF
    #    plt.plot(freq, np.abs(y_osi))
    #    plt.grid()
    #    plt.yscale("log")
    #    plt.xlabel("frequency [Hz]")
    #    plt.ylabel("amplitude [m/s^2]")
    #    plt.show()
    #
    #    # phase angle of the FRF
    #    plt.plot(freq, np.angle(y_osi,deg=True))
    #    plt.grid()
    #    plt.xlabel("frequency [Hz]")
    #    plt.ylabel("phase [°]")
    #    plt.show()

    return y_osi, freq, err_u, err_f


def osi_var2(u_signal, f_signal, fs, nt, nt_cancel=5, t=1):
    '''
    Computes the Frequency Response Function (FRF) of an operating system
    using the Operational System Identification (OSI) method.

    Parameters
    ----------
    u_signal :  ndarray, float
                sampled response signal
    f_signal :  ndarray (vector), float
                sampled force signal
    fs :        scalar, int
                sampling frequency, number of data samples aquired per second
    t  :        scalar, float, optional
                measurement block length in seconds, time interval of the FFT
    nt_cancel : scalar, int
                number of measurement blocks that are dropped in the beginning,
                time that the sytem needs to reach the steady state
    nt :        scalar, int
                number of measurement blocks

    Returns
    -------
    y_osi   : ndarray, complex
              FRF determined using the OSI-method
    freq    : ndarray, float
              discrete frequencies associated to the FFT
    U_avg   : ndarray, complex
              FFT of the averaged response signal
    F_avg   : ndarray, complex
              FFT of the averaged force signal

    References:
    -----------
    [1] de Klerk, D. "Determination of an Operating Systems' Receptance FRF
        Data (Continued)." In: proceedings on the 26th  International Modal
        Analysis Conference (IMAC), Orlando, FL

    [2] de Klerk, D. "Dynamic Response Characterization of Complex Systems
        through Operational Identification and Dynamic Substructuring",
        Dissertation, 2009

    See Also:
    --------

    Examples:
    --------


    '''

    # maybe some helpful relationships:

    # time domain:
    t_tot = nt * t  # time interval used for the OSI
    t_tot_min = (
        t_tot + nt_cancel * t
    )  # minimum signal length, the first five measurement blocks are dropped
    N = t * fs  # total number of data samples per measurement block
    delta_t = 1 / fs  # = T/N ; sampling time
    # frequency domain:
    F_max = fs / 2.0  # bandwidth, highest frequency captured in the FFT
    delta_f = (
        1 / t
    )  # = F_max/SL ; frequency resolution, distance between spectral lines
    SL = N // 2  # number of spectral lines, number of samples in freq. domain

    # check if signal is long enough:
    if (t_tot_min * N) > len(u_signal):
        raise ValueError(
            'Signal is too short. Provide a longer signal or decrease \
                          the number of measurement blocks nt or measurement block length T.'
        )

    # averaging:

    # the OSI-method can be realized in two different ways: FFT of all measurement blocks and then averaging in frequency
    # domain or averaging in time domain and then FFT of the averaged time signal. Here the 1st variant is chosen.

    U_avg = np.zeros(
        SL, dtype=complex
    )  # initialization of the averaged frequency domain data
    F_avg = np.zeros(SL, dtype=complex)
    err_U = np.zeros(nt)  # initialization of convergence vectors
    err_F = np.zeros(nt)

    for i in range(0, nt):
        start = (
            i + nt_cancel
        ) * N  # first and last index of the i-th measurement
        end = (
            i + 1 + nt_cancel
        ) * N  # block; the first five measurement blocks are dropped
        u_block = u_signal[start:end]
        f_block = f_signal[start:end]

        # transformation into frequency domain:
        U_block = fft(u_block) / N
        U_block = np.hstack((U_block[0], 2 * U_block[1:SL]))
        F_block = fft(f_block) / N
        F_block = np.hstack((F_block[0], 2 * F_block[1:SL]))

        U_avg += U_block
        F_avg += F_block

        # determination of a convergence measure for the averaged blocks:
        U_abs_i = (
            1 / (i + 1) * U_avg
        )  # total value / norm of i averaged measurement blocks
        U_abs_i = np.sum(np.abs(U_abs_i))
        F_abs_i = 1 / (i + 1) * F_avg
        F_abs_i = np.sum(np.abs(F_abs_i))

        if (
            i == 0
        ):  # the first measurement block is used as reference for normalization
            U_ref = U_abs_i
            F_ref = F_abs_i

        err_U[i] = U_abs_i / U_ref  # relative deviation, normalization
        err_F[i] = F_abs_i / F_ref

    U_avg = U_avg / nt
    F_avg = F_avg / nt

    y_osi = np.divide(U_avg, F_avg)
    freq = np.arange(0, SL) * delta_f

    # plots:

    #    # convergence plot of the averaging process
    #    plt.plot(np.arange(1,nt+1), err_U, 'b', label='response u')
    #    plt.plot(np.arange(1,nt+1), err_F, 'r', label='force f')
    #    plt.grid()
    #    plt.legend(loc='lower left')
    #    plt.xlabel("number of averaged measurement blocks [-]")
    #    plt.ylabel("convergence [-]")
    #    plt.show()

    #    # amplitudes of the FRF
    #    plt.plot(freq, np.abs(y_osi))
    #    plt.grid()
    #    plt.xlabel("frequency [Hz]")
    #    plt.ylabel("amplitude [m/s^2]")
    #    plt.show()
    #
    #    # phase angle of the FRF
    #    plt.plot(freq, np.angle(y_osi,deg=True))
    #    plt.grid()
    #    plt.xlabel("frequency [Hz]")
    #    plt.ylabel("phase [°]")
    #    plt.show()

    return y_osi, freq, err_U, err_F


def mimo_osi(responses, excitations, fs_=4096, nt_=300, nt_cancel_=5, t_=1):
    '''
    Application of the OSI method for multiple inputs and outputs

    Parameters
    ----------
    responses :   ndarray, float
                  sampled response signal of the accelerometers
                  1st dim: samples; 2nd dim: sensor channels, 3rd dim: excitation at different locations / different excitation points
                                                                      has to match with the 2nd dim of excitations
    excitations : ndarray, float
                  sampled force signal of the force transducer / impedance sensor
                  1st dim: samples; 2nd dim: excitation at different locations / different excitation points
    fs_ :         scalar, int
                  sampling frequency, number of data samples aquired per second
    t_  :         scalar, float, optional
                  measurement block length in seconds, time interval to perform a FFT
    nt_cancel_ :  scalar, int
                  number of measurement blocks that are dropped in the beginning,
                  time that the sytem needs to reach the steady state
    nt_ :         scalar, int
                  number of measurement blocks

    Returns
    -------
    y_uf   : ndarray, complex
             FRF matrix determined using the OSI-method
             1st dim: frequency response, 2nd dim: output channel / index, 3rd dim: input channel/ index
    freq   : ndarray, float
             discrete frequencies associated to the FRF

    References:
    -----------

    See Also:
    --------

    Examples:
    --------

    '''
    # check the correct dimensions:
    dim1_resp, dim2_resp, dim3_resp = np.shape(responses)
    dim1_exc, dim2_exc = np.shape(excitations)

    if dim1_resp != dim1_exc:
        raise ValueError(
            'Amount of time samples of response and excitation signals do not match!'
        )

    if dim3_resp != dim2_exc:
        raise ValueError(
            'Response array and excitation array have a different number of excitation locations!'
        )

    for i in range(0, dim3_resp):
        for j in range(0, dim2_resp):

            response = responses[:, j, i]
            force = excitations[:, i]
            y_osi, freq, _, _ = osi_var1(
                response, force, fs=fs_, nt=nt_, nt_cancel=nt_cancel_, t=t_
            )

            if i == 0 and j == 0:
                y_uf = np.zeros(
                    (np.size(freq), dim2_resp, dim3_resp), dtype=complex
                )

            y_uf[:, j, i] = y_osi

    return freq, y_uf
