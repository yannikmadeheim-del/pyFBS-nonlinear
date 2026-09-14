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

    With DLFTContactAFT the four above stay the AFT law f^nl,A on the FULL interface
    (return zeros if the joint has none), and the friction law f^F on the n_F DOFs
    selected by L_F comes from the four below, f_normal being the DLFT normal force:
        friction_force(u_F_rel, u_F_rel_dot, f_normal, tau)                 -> (Nt, n_F, 1)
        jacobian_friction_force(u_F_rel, u_F_rel_dot, f_normal, tau)        -> (Nt, n_F, n_F)
        jacobian_friction_force_qdot(u_F_rel, u_F_rel_dot, f_normal, tau)   -> (Nt, n_F, n_F)
        jacobian_friction_force_normal(u_F_rel, u_F_rel_dot, f_normal, tau) -> (Nt, n_F, n_N)
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

    def friction_force(self, u_F_rel: ArrayLike, u_F_rel_dot: ArrayLike,
                       f_normal: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement friction_force.")

    def jacobian_friction_force(self, u_F_rel: ArrayLike, u_F_rel_dot: ArrayLike,
                                f_normal: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement jacobian_friction_force.")

    def jacobian_friction_force_qdot(self, u_F_rel: ArrayLike, u_F_rel_dot: ArrayLike,
                                     f_normal: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement jacobian_friction_force_qdot.")

    def jacobian_friction_force_normal(self, u_F_rel: ArrayLike, u_F_rel_dot: ArrayLike,
                                       f_normal: ArrayLike, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement jacobian_friction_force_normal.")

    # --- Framework wrappers (do not override) ---

    def nonlinear_term(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.interface_force(u_rel, u_rel_dot, tau)

    def jacobian_nonlinear_term(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.jacobian_interface_force(u_rel, u_rel_dot, tau)

    def jacobian_nonlinear_term_qdot(self, u_rel: ArrayLike, u_rel_dot: ArrayLike, tau: ArrayLike) -> np.ndarray:
        return self.jacobian_interface_force_qdot(u_rel, u_rel_dot, tau)

