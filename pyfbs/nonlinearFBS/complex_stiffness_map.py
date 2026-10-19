from abc import ABC, abstractmethod

import numpy as np


def _per_dof(values, amplitude):
    """Return values as a complex array with one entry per interface DoF."""
    return np.broadcast_to(np.asarray(values, dtype=complex), np.shape(amplitude))


class ComplexStiffnessMap(ABC):
    """
    Diagonal, amplitude-dependent complex joint stiffness K*(omega, a).

    Each interface DoF j has its own scalar law k*_j(omega, a_j), which depends
    only on the physical first-harmonic amplitude a_j of the same DoF.
    All methods return the n_int diagonal entries of K*, not the full matrix.
    """

    @abstractmethod
    def evaluate_Kstar(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        """
        Complex stiffness of every interface DoF.

        :param omega: angular frequency [rad/s]
        :param amplitude: physical amplitudes a_j, shape (n_int,) [m or rad]
        :return: k*_j(omega, a_j), complex, shape (n_int,)
        """

    @abstractmethod
    def evaluate_dKstar_damplitude(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        """
        Derivative of the complex stiffness with respect to the amplitude,
        at fixed angular frequency.

        :param omega: angular frequency [rad/s]
        :param amplitude: physical amplitudes a_j, shape (n_int,) [m or rad]
        :return: d k*_j / d a_j, complex, shape (n_int,)
        """

    @abstractmethod
    def evaluate_dKstar_domega(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        """
        Derivative of the complex stiffness with respect to the angular
        frequency, at fixed amplitude.

        :param omega: angular frequency [rad/s]
        :param amplitude: physical amplitudes a_j, shape (n_int,) [m or rad]
        :return: d k*_j / d omega, complex, shape (n_int,)
        """

class AnalyticalStiffnessMap(ComplexStiffnessMap):
    """
    Complex stiffness map from analytic functions (surrogate model).

    :param Kstar_function: function (omega, amplitude) -> k*_j, shape (n_int,)
    :param dKstar_damplitude_function: function (omega, amplitude) -> d k*_j / d a_j
    :param dKstar_domega_function: function (omega, amplitude) -> d k*_j / d omega
    """
    def __init__(self, Kstar_function, dKstar_damplitude_function, dKstar_domega_function):
        self._Kstar = Kstar_function
        self._dKstar_damplitude = dKstar_damplitude_function
        self._dKstar_domega = dKstar_domega_function

    def evaluate_Kstar(self, omega, amplitude):
        return _per_dof(self._Kstar(omega, amplitude), amplitude)

    def evaluate_dKstar_damplitude(self, omega, amplitude):
        return _per_dof(self._dKstar_damplitude(omega, amplitude), amplitude)

    def evaluate_dKstar_domega(self, omega, amplitude):
        return _per_dof(self._dKstar_domega(omega, amplitude), amplitude)