import numpy as np
from numpy.typing import ArrayLike


class FBS_System:
    """
    Base class for Frequency Based Substructuring systems.
    Residual: R = Q_rel - B @ Y^{A|B} @ F_ext + B @ Y^{A|B} @ B^T @ F_nl = 0

    Attributes (set by subclass):
        mass_matrix:       (dTotal, dTotal) — block-diagonal subsystem mass matrices
        damping_matrix:    (dTotal, dTotal) — block-diagonal subsystem damping matrices
        stiffness_matrix:  (dTotal, dTotal) — block-diagonal subsystem stiffness matrices
        B_coupling:        (n_int, dTotal)  — signed Boolean coupling matrix
        dimension:         int              — n_int (number of interface DOF pairs)
        polynomial_degree: int

    Subclasses must implement:
        external_term(tau)                                   -> (Nt, dTotal, 1)
        interface_force(u_rel, u_rel_dot, tau)               -> (Nt, n_int, 1)
        jacobian_interface_force(u_rel, u_rel_dot, tau)      -> (Nt, n_int, n_int)
        jacobian_interface_force_qdot(u_rel, u_rel_dot, tau) -> (Nt, n_int, n_int)
    """
    is_real_valued: bool = True

    def __init__(self):
        self.mass_matrix:      np.ndarray = np.eye(2)      # (dTotal, dTotal)
        self.damping_matrix:   np.ndarray = np.zeros((2, 2))
        self.stiffness_matrix: np.ndarray = np.eye(2)
        self.B_coupling:       np.ndarray = np.array([[1, -1]])  # (n_int, dTotal)
        self.dimension:        int = 1
        self.polynomial_degree: int = 3

    # --- Subclass interface (semantic names) ---

    def external_term(self, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement external_term.")

    def interface_force(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement interface_force.")

    def jacobian_interface_force(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement jacobian_interface_force.")

    def jacobian_interface_force_qdot(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement jacobian_interface_force_qdot.")

    # --- Framework wrappers (nicht überschreiben) ---

    def nonlinear_term(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.interface_force(u_rel, u_rel_dot, tau)

    def jacobian_nonlinear_term(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.jacobian_interface_force(u_rel, u_rel_dot, tau)

    def jacobian_nonlinear_term_qdot(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.jacobian_interface_force_qdot(u_rel, u_rel_dot, tau)

