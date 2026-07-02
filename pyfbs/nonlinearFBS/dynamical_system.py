import numpy as np
from numpy.typing import ArrayLike


class FBS_System:
    """
    Base class for Frequency Based Substructuring systems.
    Residual: R = Q_rel - B @ Y^{A|B} @ F_ext + B @ Y^{A|B} @ B^T @ F_nl = 0

    Attributes (set by subclass):
        B_coupling:        (n_int, dTotal)  — signed Boolean coupling matrix
        modal data (angular_eig_freq / eig_vec / zeta) — defines the uncoupled
            block-diagonal subsystem FRF used to build the FRF provider
        sample_number:     int              — number of AFT time samples

    The interface dimension (n_int = B rows) and total dimension (dTotal = FRF
    DOF count) are NOT defined here — the solver (FBSProblem) derives them.

    Subclasses must implement:
        external_term(tau)                                   -> (Nt, dTotal, 1)
        interface_force(u_rel, u_rel_dot, tau)               -> (Nt, n_int, 1)
        jacobian_interface_force(u_rel, u_rel_dot, tau)      -> (Nt, n_int, n_int)
        jacobian_interface_force_qdot(u_rel, u_rel_dot, tau) -> (Nt, n_int, n_int)
    """
    is_real_valued: bool = True

    def __init__(self):
        self.B_coupling:       np.ndarray = np.array([[1, -1]])  # (n_int, dTotal)
        self.sample_number: int = 400
        self.omega_ref: float = 1.0

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

