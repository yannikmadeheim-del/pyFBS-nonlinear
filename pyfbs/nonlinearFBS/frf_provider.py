import numpy as np
from numpy import zeros, eye, array
from numpy.fft import rfft, irfft
from scipy.interpolate import CubicSpline
from abc import ABC, abstractmethod

from .frequency_domain import Fourier, FourierOmegaPoint


class FRFProvider(ABC):
    """
    Computes the per-harmonic admittance blocks Y_n(omega) and their omega-derivatives.
    Both are returned as a stack of shape (Nh, d, d): one (d, d) block per harmonic n.
    """

    @abstractmethod
    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        """Return the admittance stack Y, shape (Nh, d, d), complex."""

    @abstractmethod
    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        """Return the stack dY/domega, shape (Nh, d, d), complex."""


class NumericalFRF(FRFProvider):
    """
    FRF computed analytically from mass, damping and stiffness matrices.
        Z_n = -(n*omega)^2 * M + i*(n*omega)*C + K
        Y_n = Z_n^{-1}
        dY_n/domega = -Y_n @ dZ_n/domega @ Y_n,  dZ_n/domega = -2*n^2*omega*M + i*n*C
    """

    def __init__(self, M: array, C: array, K: array):
        self.M = M
        self.C = C
        self.K = K

    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        # n has shape (Nh, 1, 1) so that broadcasting against the (d, d) matrices
        # builds the whole (Nh, d, d) stack Z_n = -(n*omega)^2 M + i n*omega C + K
        # in one expression; the batched solve then inverts all blocks in one
        # LAPACK call instead of a Python loop.
        Nh = len(harmonics)
        n = np.asarray(harmonics, dtype=float).reshape(-1, 1, 1)
        Z = -(n * omega) ** 2 * self.M + 1j * (n * omega) * self.C + self.K
        return np.linalg.solve(Z, np.broadcast_to(eye(d), (Nh, d, d)))

    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        # dY_n = -Y_n dZ_n Y_n for all harmonics with one batched matmul;
        # Y_cache is the (Nh, d, d) stack returned by compute_FRF.
        n = np.asarray(harmonics, dtype=float).reshape(-1, 1, 1)
        dZ = -2.0 * n ** 2 * omega * self.M + 1j * n * self.C
        return -Y_cache @ dZ @ Y_cache


class ExperimentalFRF(FRFProvider):
    """
    FRF interpolated from measured frequency-domain data.

    Parameters
    ----------
    omega_frf : array, shape (N_freq,)
        Measured frequency points (positive, rad/s).
    Y : array, shape (N_freq, d, d)
        Complex admittance matrices at each frequency point.
    fd_step : float
        Step size for central-difference dY/domega.
    """

    def __init__(self, omega_frf: array, Y: array, fd_step: float = 1e-6):
        self.omega_frf = omega_frf
        self.Y = Y
        self.fd_step = fd_step
        self.interp_real = CubicSpline(omega_frf, Y.real)
        self.interp_imag = CubicSpline(omega_frf, Y.imag)

    def interpolate(self, omega) -> array:
        """Interpolate Y at scalar or vector omega. Handles Y(-omega) = conj(Y(omega))."""
        omega = np.asarray(omega)
        neg_mask = omega < 0
        omega_abs = np.abs(omega)

        if np.any(omega_abs > self.omega_frf[-1]):
            import warnings
            warnings.warn(
                f"omega outside FRF data range [0, {self.omega_frf[-1]:.4f}]. Extrapolating."
            )

        result = self.interp_real(omega_abs) + 1j * self.interp_imag(omega_abs)

        if np.ndim(omega) == 0:
            if bool(neg_mask):
                result = np.conj(result)
        elif np.any(neg_mask):
            result[neg_mask] = np.conj(result[neg_mask])

        return result

    def compute_FRF(self, omega: float, harmonics, d: int) -> array:
        omega_harmonics = harmonics * omega          # shape (Nh,)
        return self.interpolate(omega_harmonics)     # stack, shape (Nh, d, d)

    def compute_FRF_derivative(self, omega: float, harmonics, d: int,
                               Y_cache: array) -> array:
        h = self.fd_step
        dY_raw = (self.compute_FRF(omega + h, harmonics, d)
                  - self.compute_FRF(omega - h, harmonics, d)) / (2 * h)
        return dY_raw