import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline
import warnings

class FBSSetup:
    """
    Base class for Frequency Based Substructuring systems: the setup shared by
    every joint formulation. The joint law is added by a subclass
    (FBS_System: time domain, FBS_SystemComplexStiffness: frequency domain).

    Attributes (set by subclass):
        B_coupling:        (n_int, dTotal)  — signed Boolean coupling matrix
        modal data (angular_eig_freq / eig_vec / zeta) — defines the uncoupled
            block-diagonal subsystem FRF used to build the FRF provider
        sample_number:     int              — time samples per period (excitation
                                              DFT, coefficient scaling, AFT)
        omega_ref:         float            — reference angular frequency

    The interface dimension (n_int = B rows) and total dimension (dTotal = FRF
    DOF count) are NOT defined here — the solver (FBSProblem) derives them.

    Subclasses must implement:
        external_term(tau)                                   -> (Nt, dTotal, 1)
    """
    is_real_valued: bool = True

    def __init__(self):
        self.B_coupling:       np.ndarray = np.array([[1, -1]])  # (n_int, dTotal)
        self.sample_number: int = 400
        self.omega_ref: float = 1.0

    def external_term(self, tau: ArrayLike) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement external_term.")


class FBS_System(FBSSetup):
    """
    FBS system with a time-domain joint law, for AFT, DLFTContact and DLFTContactAFT.
    Residual: R = Q_rel - B @ Y^{A|B} @ F_ext + B @ Y^{A|B} @ B^T @ F_nl = 0

    Subclasses must implement external_term (see FBSSetup) and:
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

    # --- Subclass interface (semantic names) ---

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


class FBS_SystemComplexStiffness(FBSSetup):
    """
    FBS system with a frequency-domain joint law, for ComplexStiffness: a diagonal,
    amplitude-dependent complex stiffness k*_j(omega, a_j), first harmonic only.
    Residual: R_1 = Q_rel_1 + B @ Y @ B^T @ K* @ Q_rel_1 - B @ Y @ F_ext_1 = 0

    Subclasses must implement external_term (see FBSSetup) and, for the physical
    first-harmonic amplitudes a_j [m or rad] of shape (n_int,):
        complex_stiffness(omega, amplitude)                      -> (n_int,) complex
        complex_stiffness_derivative_amplitude(omega, amplitude) -> (n_int,) complex
        complex_stiffness_derivative_omega(omega, amplitude)     -> (n_int,) complex
    The derivatives are partial: d/da_j at fixed omega, d/domega at fixed a_j.
    A constant may be returned as a scalar; ComplexStiffness broadcasts it.
    """

    def complex_stiffness(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement complex_stiffness.")

    def complex_stiffness_derivative_amplitude(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement complex_stiffness_derivative_amplitude.")

    def complex_stiffness_derivative_omega(self, omega: float, amplitude: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement complex_stiffness_derivative_omega.")

class FBS_SystemComplexStiffnessExperimental(FBS_SystemComplexStiffness):
    """
    Diagonal complex joint stiffness from tabulated data, interpolated with
    cubic splines over (omega, amplitude), separately for each DoF.

    :param omega: frequency grid, strictly increasing, shape (P,) [rad/s]
    :param amplitude: amplitude grid of each DoF (column j for DoF j),
        strictly increasing per column, shape (M, n_int) [m or rad]
    :param K: complex stiffness k*_j(omega_p, a_mj), shape (P, M, n_int)
    """

    def __init__(self, omega, amplitude, K):
        super().__init__()
        self.omega_grid = np.asarray(omega, dtype=float)
        self.amplitude_grid = np.asarray(amplitude, dtype=float)
        self.K_table = np.asarray(K, dtype=complex)
        if self.omega_grid.ndim != 1 or self.amplitude_grid.ndim != 2 or self.K_table.ndim != 3:
            raise ValueError("omega must be 1D, amplitude 2D (M, n_int) and K 3D (P, M, n_int)")
        if self.K_table.shape != (len(self.omega_grid), *self.amplitude_grid.shape):
            raise ValueError(f"K must have shape (P, M, n_int) = "
                             f"({len(self.omega_grid)}, {self.amplitude_grid.shape[0]}, {self.amplitude_grid.shape[1]}), "
                             f"got {self.K_table.shape}")

        if not np.all(np.diff(self.omega_grid) > 0):
            raise ValueError("omega-table is not strictly increasing")
        if not np.all(np.diff(self.amplitude_grid, axis=0) > 0):
            raise ValueError("amplitude-table is not strictly increasing")

        if self.omega_grid.shape[0] < 4:
            raise ValueError("at least 4 samples along the omega axis needed")
        if self.amplitude_grid.shape[0] < 4:
            raise ValueError("at least 4 samples along the amplitude axis needed")
        self._splines = []  # one (real, imag) pair per DoF
        for j in range(self.K_table.shape[2]):
            a_j = self.amplitude_grid[:, j]  # this DoF's own amplitude grid
            K_j = self.K_table[:, :, j]  # this DoF's table, (P, M)
            self._splines.append((RectBivariateSpline(self.omega_grid, a_j, K_j.real),
                                  RectBivariateSpline(self.omega_grid, a_j, K_j.imag)))

    def _evaluate(self, omega, amplitude, d_omega=0, d_amplitude=0):
        """
        k*_j or one of its partial derivatives for every DoF, complex, shape (n_int,).

        Below the smallest tabulated amplitude the value of the smallest level is
        held (linear small-amplitude limit, zero amplitude derivative). A frequency
        outside the grid or an amplitude above the largest level raises ValueError.
        """
        if not self.omega_grid[0] <= omega <= self.omega_grid[-1]:
            raise ValueError(f"omega = {omega:.6g} rad/s is outside the table "
                             f"[{self.omega_grid[0]:.6g}, {self.omega_grid[-1]:.6g}] rad/s")
        above = amplitude > self.amplitude_grid[-1]
        if np.any(above):
            raise ValueError(f"amplitude above the largest tabulated level "
                             f"for DoFs {np.flatnonzero(above).tolist()}")

        below = amplitude < self.amplitude_grid[0]
        if np.any(below):
            warnings.warn("FBS_SystemComplexStiffnessExperimental: amplitude below the smallest "
                          "tabulated level - holding the value of the smallest level "
                          "(linear small-amplitude limit)")
        amplitude = np.maximum(amplitude, self.amplitude_grid[0])

        values = np.empty(len(self._splines), dtype=complex)
        for j, (spline_re, spline_im) in enumerate(self._splines):
            values[j] = (spline_re.ev(omega, amplitude[j], dx=d_omega, dy=d_amplitude)
                         + 1j * spline_im.ev(omega, amplitude[j], dx=d_omega, dy=d_amplitude))
        if d_amplitude:
            values[below] = 0.0
        return values

    def complex_stiffness(self, omega, amplitude):
        return self._evaluate(omega, amplitude)

    def complex_stiffness_derivative_amplitude(self, omega, amplitude):
        return self._evaluate(omega, amplitude, d_amplitude=1)

    def complex_stiffness_derivative_omega(self, omega, amplitude):
        return self._evaluate(omega, amplitude, d_omega=1)