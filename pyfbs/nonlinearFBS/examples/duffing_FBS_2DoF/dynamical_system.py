# %%

import scipy as sp
import numpy as np
from numpy import cos, array, concatenate
from numpy.typing import ArrayLike
from pyfbs.nonlinearFBS import FBS_System


class System2DoF_FBS(FBS_System):
    """
    Implements the coupling of 2 Substructures with the dynamics
        m1*q1'' + c1*q1' + k1*q1 = P*cos(tau)
        m2*q2'' + c2*q2' + k2*q2 = P*cos(tau)
    Coupled by a cubic Spring
        fnl = beta*(q1-q2)**3
    """

    is_real_valued = True

    def __init__(self, c1=0.01, c2=0.01, k1=1.0, k2=1.0, k3=0.0, beta=1.0, alpha=0.0, P=1.0):
        """
		Initializes the linear FBS system.

		:param d: Damping coefficient (nearest-neighbour) [T^-1]
		:param k: Stiffness coefficient (nearest-neighbour) [T^-2]
		:param P: Amplitude of the external force on DOF 0 of subsystem A [L T^-2]
		"""
        M1 = [[1]]
        M2 = [[1]]
        C1 = [[c1]]
        C2 = [[c2]]
        K1 = [[k1]]
        K2 = [[k2]]

        self.c1 = c1
        self.c2 = c2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.beta = beta
        self.alpha = alpha
        self.P = P
        self.mass_matrix = sp.linalg.block_diag(M1, M2)
        self.damping_matrix = sp.linalg.block_diag(C1, C2)
        self.stiffness_matrix = sp.linalg.block_diag(K1, K2)
        self.B_coupling = np.array([[1, -1]])  # u_rel = q1 - q2
        self.total_dimension = 2
        self.dimension = 1
        self.polynomial_degree = 3

    def external_term(self, adimensional_time: np.ndarray) -> np.ndarray:
        """
        External force P*cos(tau) on mass 1 only.

        :param adimensional_time: Adimensional time, shape (Nt,)
        :return: External force array, shape (Nt, 2, 1)
        """
        f = np.zeros((len(adimensional_time), self.total_dimension, 1))
        f[:, 0, 0] = self.P * cos(adimensional_time)
        return f

    def interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Cubic interface force: lambda = beta * u_rel^3"""
        return self.k3 * u_rel + self.beta * np.power(u_rel, 3) + self.alpha * np.power(u_rel, 2) * udot_rel

    def jacobian_interface_force(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """d(lambda)/d(u_rel) = 3 * beta * u_rel^2, shape (Nt, 1, 1)"""
        J_int = self.k3 + 3 * self.beta * np.power(u_rel, 2) + 2 * self.alpha * u_rel * udot_rel
        return J_int

    def jacobian_interface_force_qdot(self, u_rel: np.ndarray, udot_rel: np.ndarray, adimensional_time: np.ndarray) -> np.ndarray:
        """No velocity dependence."""
        Jdot_int = self.alpha * np.power(u_rel, 2)
        return Jdot_int


class System2DoF_FBS_experimental(System2DoF_FBS):
    """
    Experimental variant of System2DoF_FBS.
    Pre-computes the uncoupled FRF Y(omega) = (-omega^2 * M + i*omega * C + K)^{-1}
    from the subsystem matrices and stores it as self.omega_frf, self.Y_frf
    for interpolation inside ExperimentalFRF (passed to FBSProblem in main.py).
    """

    def __init__(self, c1=0.01, c2=0.01, k1=1.0, k2=1.0, k3=0.0, beta=1.0, alpha=0.0, P=1.0):
        super().__init__(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)
        omega_start = 0.00
        omega_end = 5.0*9*2
        ome_density = 1000.0 # points per rad/s
        n_points = int((omega_end - omega_start) * ome_density)
        omega_frf = np.linspace(omega_start, omega_end, n_points)
        Y_frf = np.zeros((n_points, self.total_dimension, self.total_dimension), dtype=complex)
        for i, w in enumerate(omega_frf):
            Y_frf[i] = np.linalg.solve(
                -w ** 2 * self.mass_matrix + 1j * w * self.damping_matrix + self.stiffness_matrix,
                np.eye(self.total_dimension)
            )
        self.omega_frf = omega_frf
        self.Y_frf = Y_frf
