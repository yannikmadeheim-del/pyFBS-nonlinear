import numpy as np
from numpy import eye, block, kron, diag, zeros_like
from abc import ABC, abstractmethod
import warnings

from .dynamical_system import FBS_SystemComplexStiffness
from .frequency_domain import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    JacobianFourier_Real, block_diag_stack_to_RI,
)


def _per_dof_broadcast(values, amplitude):
    """Return values as a complex array with one entry per interface DoF."""
    return np.broadcast_to(np.asarray(values, dtype=complex), np.shape(amplitude))


def _velocity_scaled_tangent(G: JacobianFourier_Real, Gdot: JacobianFourier_Real,
                             omega: float, n_cols: int) -> JacobianFourier_Real:
    """G_{n-m} + i ω m Gdot_{n-m}: tangent of a force law that depends on
    displacement AND velocity.  ∂qdot_k/∂Q_m = i ω m ∂q_k/∂Q_m, so the velocity
    part enters as the harmonic column scale ω m, and the factor i shuffles the
    RI blocks (Re <- -Im, Im <- Re).  n_cols is the column dimension of the two
    kernels: d_int for a law on the full interface, n_F for one on the friction
    DOFs only.
    """
    col_scale = omega * kron(diag(Fourier.harmonics), eye(n_cols))
    return JacobianFourier_Real(
        RR=G.RR + Gdot.RI @ col_scale,
        RI=G.RI - Gdot.RR @ col_scale,
        IR=G.IR + Gdot.II @ col_scale,
        II=G.II - Gdot.IR @ col_scale,
    )


def _to_RI(J: JacobianFourier_Real) -> np.ndarray:
    """A JacobianFourier block table as the dense real map on [Re Q; Im Q]."""
    return block([[J.RR, J.RI], [J.IR, J.II]])


def _localisation_to_RI(L: np.ndarray) -> np.ndarray:
    """RI embedding of a real, harmonic-independent localisation matrix: it acts
    identically on the real and the imaginary part of every harmonic."""
    LL = kron(eye(Fourier.number_of_harmonics), L)
    O = zeros_like(LL)
    return block([[LL, O], [O, LL]])


class NonlinearMethod(ABC):
    """
    Strategy that provides F_int, dF_int/dx_r and dF_int/dω in RI form.

    A method that needs problem-level data (e.g. admittance, coupling matrix)
    should override `bind` to capture a reference to its owning problem.
    """

    def bind(self, problem) -> None:
        """Capture problem-level context. Default: no-op."""
        pass

    @abstractmethod
    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """Nonlinear/contact force, shape (Nh*d, 1), complex."""

    @abstractmethod
    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """RI Jacobian dF_int/dx_r, shape (2*Nh*d, 2*Nh*d), real."""

    @abstractmethod
    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """RI omega-derivative of F_int, shape (2*Nh*d, 1), real."""


class AFT(NonlinearMethod):
    """Alternating Frequency-Time (AFT) scheme."""

    def bind(self, problem) -> None:
        self._problem = problem    # for n_int = problem.d_int (interface DOF count)

    def _get_nonlinear_term(self, x: FourierOmegaPoint, ode) -> Fourier_Real:
        if x.nonlinear_term_cache is None:
            Fourier_Real.compute_time_series(x.fourier)
            q    = x.fourier.time_series
            qdot = x.compute_time_series_derivative()
            fnl_ts = ode.nonlinear_term(q, qdot, Fourier.adimensional_time_samples)
            x.nonlinear_term_cache = Fourier_Real.new_from_time_series(fnl_ts)
        return x.nonlinear_term_cache

    def _get_Gdot(self, x: FourierOmegaPoint, ode) -> JacobianFourier_Real:
        if x.Gdot is None:
            self._get_nonlinear_term(x, ode)
            dfnldqdot_ts = ode.jacobian_nonlinear_term_qdot(
                x.fourier.time_series,
                x.time_series_derivative,
                Fourier.adimensional_time_samples,
            )
            x.Gdot = JacobianFourier_Real.new_from_time_series(dfnldqdot_ts)
        return x.Gdot

    def _get_jacobian_nonlinear_term(self, x: FourierOmegaPoint, ode) -> JacobianFourier_Real:
        self._get_nonlinear_term(x, ode)
        dfnldq_ts = ode.jacobian_nonlinear_term(
            x.fourier.time_series,
            x.time_series_derivative,
            Fourier.adimensional_time_samples,
        )
        G    = JacobianFourier_Real.new_from_time_series(dfnldq_ts)
        Gdot = self._get_Gdot(x, ode)
        return _velocity_scaled_tangent(G, Gdot, x.omega, self._problem.d_int)

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        fnl = self._get_nonlinear_term(x, ode)
        return fnl.coefficients.reshape(-1, 1)   # (Nh, d, 1) -> (Nh*d, 1) view

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        Jnl = self._get_jacobian_nonlinear_term(x, ode)
        return block([[Jnl.RR, Jnl.RI], [Jnl.IR, Jnl.II]])

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        qdot_adim = x.fourier.get_adimensional_time_derivative()
        Gdot   = self._get_Gdot(x, ode)
        qdot_R = qdot_adim.real.reshape(-1, 1)
        qdot_I = qdot_adim.imag.reshape(-1, 1)
        return np.concatenate((
            Gdot.RR @ qdot_R + Gdot.RI @ qdot_I,
            Gdot.IR @ qdot_R + Gdot.II @ qdot_I,
        ))


class DLFTContact(NonlinearMethod):
    r"""
    Dynamic Lagrangian Frequency-Time (DLFT) normal contact.

    Prediction:  λ_p = IDFT[Z_r (F_adm - x_r)] + ε (x_r - g₀)
    Correction:  λ   = max(0, λ_p)
    Force:       λ̃   = DFT[λ]

    The converged force is ε-independent: at the solution λ_p > 0 exactly where
    the interface penetrates, so ε only conditions the iteration, not the branch.

    Bound to an FBSProblem at construction; reads B, F_ext, the per-harmonic
    admittance caches, and the FRF provider through that reference.
    """

    def __init__(self, epsilon: float = 1.0, g_zero: float = 0.0):
        self.epsilon  = epsilon
        self.g_zero   = g_zero
        self._problem = None    # populated by bind()

    def bind(self, problem) -> None:
        self._problem = problem

    # --- internal helpers (use self._problem) ---

    def _get_Yr(self, x):
        return self._problem._get_Yr(x)         # (Nh, n_int, n_int) stack, cached on x

    def _get_Fext_admr(self, x):
        return self._problem._get_Fadm(x)       # (Nh, n_int, 1) stack, cached on x

    def _get_Zr_rhs(self, x):
        # z = Y_r^{-1} (F_adm - x_r), solved harmonic-by-harmonic (batched)
        if x.Zr_rhs is None:
            x.Zr_rhs = np.linalg.solve(self._get_Yr(x),
                                       self._get_Fext_admr(x) - x.fourier.coefficients)
        return x.Zr_rhs

    def _get_lambda_corrected(self, x):
        if x.lambda_corrected is None:
            # z_r = Y_r^{-1} (F_adm - x_r): the contact force the linear coupled
            # system would need to hold the interface at x_r ("balancing" force)
            zr_fourier = Fourier(self._get_Zr_rhs(x))   # already (Nh, n_int, 1)
            Fourier_Real.compute_time_series(zr_fourier)
            zr_t   = zr_fourier.time_series
            Fourier_Real.compute_time_series(x.fourier)
            q_rel  = x.fourier.time_series
            lambda_p = zr_t + self.epsilon * (q_rel - self.g_zero)
            x.contact_mask     = lambda_p > 0.0
            lambda_t_corr      = np.where(x.contact_mask, lambda_p, 0.0)

            lambda_x_corr      = Fourier_Real.new_from_time_series(lambda_t_corr)
            x.lambda_corrected = lambda_x_corr.coefficients.reshape(-1, 1)
        return x.lambda_corrected

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask
        n_int = self._problem.d_int
        contact_tangent = x.contact_mask * eye(n_int)    # (Nt, n_int, n_int)
        J_mask    = JacobianFourier_Real.new_from_time_series(contact_tangent)
        J_mask_RI = block([[J_mask.RR, J_mask.RI], [J_mask.IR, J_mask.II]])
        # Z_r = Y_r^{-1} per harmonic (batched) = dynamic interface stiffness;
        # M = eps*I - Z_r, block-diagonal
        Nh = Fourier.number_of_harmonics
        Zr = np.linalg.solve(self._get_Yr(x),
                             np.broadcast_to(eye(n_int), (Nh, n_int, n_int)))
        M  = self.epsilon * eye(n_int) - Zr              # broadcasts over harmonics
        return J_mask_RI @ block_diag_stack_to_RI(M)

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        problem = self._problem
        BdY = problem.B @ problem._get_dY(x)             # B dY/dω, (Nh, n_int, d_total)
        # rhs = B dY/dω (F_ext - B^T z_r): how the predicted multiplier's source
        # term shifts with ω at frozen x_r
        rhs = BdY @ (problem.F_ext - problem.B_f.T @ self._get_Zr_rhs(x))
        dlambda_pred = np.linalg.solve(self._get_Yr(x), rhs)   # dλ_p/dω, (Nh, n_int, 1)
        dlambda_pred_fourier = Fourier(dlambda_pred)
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        dlambda_t_corr = np.where(x.contact_mask, dlambda_pred_fourier.time_series, 0.0)
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_v      = dlambda_corr.coefficients.reshape(-1, 1)
        return np.concatenate((dlambda_v.real, dlambda_v.imag))


class DLFTContactAFT(NonlinearMethod):
    r"""
    DLFT normal contact and AFT force laws on ONE interface.

    Two user-supplied localisation matrices split the interface DOFs, and three
    forces act simultaneously per residue evaluation:

        f^N    = f^N(q_rel)                       DLFT prediction/correction
        f^F    = f^F(q^F_rel, qdot^F_rel, f^N)    AFT, driven by the normal force
        f^nl,A = f^nl,A(q_rel, qdot_rel)          AFT, on the FULL interface

        F_nl = L^N,T F^N + L^F,T F^F + F^nl,A

    so :class:`FBSProblem` assembles R = Q_rel + Y_r F_nl - F_adm unchanged.

    L^N L^F,T = 0 is MANDATORY.  Projecting the residual with L^N is what isolates
    F^N, and F^F only drops out of that projection when the friction DOFs are
    orthogonal to the normal ones; the scheme and its analytic Jacobian both rest
    on it.  f^nl,A carries no localisation matrix, does NOT drop out, and
    therefore enters the normal prediction.

    Evaluation order -- an explicit chain, no inner fixed point, because f^N
    depends on f^nl,A but not on f^F, while f^F depends on f^N:

        1. AFT   f^nl,A(t), F^nl,A
        2. DLFT  f^N,pred(t) = L^N [z_r(t) - f^nl,A(t)] + eps (L^N q_rel(t) - g0)
                 f^N(t) = max(0, f^N,pred(t)),  b_k = 1[f^N,pred_k > 0],  F^N = DFT[f^N]
        3. AFT   f^F(t) from (L^F q_rel, L^F qdot_rel, f^N),  F^F = DFT[f^F]
        4. F_nl  = L^N,T F^N + L^F,T F^F + F^nl,A

    with z_r(t) = iDFT[Y_r^{-1} (F_adm - Q_rel)] exactly as in :class:`DLFTContact`.

    Sign convention: pyFBS's, i.e. :class:`DLFTContact`'s.  The gap is measured
    positive in PENETRATION (contact when L^N q_rel > g0), so the penalty enters
    as +eps (L^N q_rel - g0) and every (Z_r + eps I) of the underlying derivation
    -- which measures the gap positive when open -- appears here as (eps I - Z_r).

    Every time series an AFT block sees is a TRUNCATED reconstruction over the
    retained harmonics: f^nl,A and f^N are both rebuilt from their Fourier
    coefficients rather than reused raw.  Anything else leaves the analytic
    Jacobian inconsistent with the residual as soon as a law generates harmonics
    above H -- max(0, .) always does.  The contact indicator b_k is piecewise
    constant, so no db_k term appears and the Newton scheme is SEMISMOOTH: at a
    switching sample the Jacobian is one element of the Clarke subdifferential.

    The force laws come from the FBS_System: ``interface_force`` and its two
    Jacobians are f^nl,A on the full interface (return zeros if there is none),
    ``friction_force`` and its three Jacobians are f^F on the friction DOFs.

    :param L_N: (n_N, n_int) normal localisation, orthonormal rows.
    :param L_F: (n_F, n_int) friction localisation, orthonormal rows, orthogonal
        to L_N.
    :param epsilon: DLFT penalty / Lagrangian factor; conditions the iteration
        only, the converged force is eps-independent.
    :param g_zero: normal gap offset g0, scalar or (n_N, 1); contact when
        L^N q_rel > g0.
    """

    def __init__(self, L_N, L_F, epsilon: float = 1.0, g_zero=0.0):
        self.L_N = np.asarray(L_N, dtype=float)
        self.L_F = np.asarray(L_F, dtype=float)
        self.epsilon = epsilon
        self.g_zero  = g_zero
        self.n_N, n_int = self.L_N.shape
        self.n_F = self.L_F.shape[0]

        if self.L_F.shape[1] != n_int:
            raise ValueError(
                f"L_N has {n_int} columns but L_F has {self.L_F.shape[1]}; both "
                f"localise the same relative interface vector q_rel.")
        if not np.allclose(self.L_N @ self.L_F.T, 0.0):
            raise ValueError(
                "L_N L_F^T must be zero. The normal force is isolated by projecting "
                "the residual with L_N, which removes the friction force only if the "
                "friction DOFs are orthogonal to the normal ones; the whole scheme "
                "and its Jacobian assume it.")
        if not np.allclose(self.L_N @ self.L_N.T, eye(self.n_N)):
            raise ValueError("L_N must have orthonormal rows (L_N L_N^T = I).")
        if not np.allclose(self.L_F @ self.L_F.T, eye(self.n_F)):
            raise ValueError("L_F must have orthonormal rows (L_F L_F^T = I).")

        self._aft     = AFT()    # the whole f^nl,A block, including its dF/dω
        self._problem = None     # populated by bind()

    def bind(self, problem) -> None:
        self._problem = problem
        self._aft.bind(problem)
        if self.L_N.shape[1] != problem.d_int:
            raise ValueError(
                f"L_N/L_F have {self.L_N.shape[1]} columns but the interface has "
                f"{problem.d_int} DOFs (B_coupling rows).")
        # the Jacobian kernels are indexed by n - r in [-2H, 2H]; below 4H+1
        # samples those bins alias onto each other
        Nt, H = Fourier.number_of_time_samples, Fourier.harmonic_truncation_order
        if Nt < 4 * H + 1:
            warnings.warn(
                f"sample_number={Nt} < 4H+1={4 * H + 1}: the Jacobian Toeplitz kernels "
                f"wrap around and the analytic Jacobian is no longer exact.")
        # the localisation matrices carry no frequency dependence, so their RI
        # embeddings are built once
        self.LN_RI = _localisation_to_RI(self.L_N)   # (2 Nh n_N, 2 Nh n_int)
        self.LF_RI = _localisation_to_RI(self.L_F)   # (2 Nh n_F, 2 Nh n_int)

    # --- internal helpers (use self._problem) ---

    def _get_Zr_rhs(self, x):
        # z_r = Y_r^{-1} (F_adm - Q_rel): the interface force the linear coupled
        # system would need to hold the interface at Q_rel
        if x.Zr_rhs is None:
            x.Zr_rhs = np.linalg.solve(
                self._problem._get_Yr(x),
                self._problem._get_Fadm(x) - x.fourier.coefficients)
        return x.Zr_rhs

    def _truncated_time_series(self, coefficients):
        """iDFT over the RETAINED harmonics only."""
        truncated = Fourier(coefficients)
        Fourier_Real.compute_time_series(truncated)
        return truncated.time_series

    def _get_F_int(self, x, ode):
        """One pass of the chain f^nl,A -> f^N -> f^F."""
        if x.F_int_cache is None:
            Nh, n_int = Fourier.number_of_harmonics, self._problem.d_int
            F_A    = self._aft.compute_F_int(x, ode).reshape(Nh, n_int, 1)
            q_t    = x.fourier.time_series            # populated by the AFT pass
            qdot_t = x.compute_time_series_derivative()

            f_pred = (self.L_N @ (self._truncated_time_series(self._get_Zr_rhs(x))
                                  - self._truncated_time_series(F_A))
                      + self.epsilon * (self.L_N @ q_t - self.g_zero))
            x.contact_mask = f_pred > 0.0
            F_N = Fourier_Real.new_from_time_series(
                np.where(x.contact_mask, f_pred, 0.0)).coefficients
            x.f_normal = self._truncated_time_series(F_N)

            F_F = Fourier_Real.new_from_time_series(
                ode.friction_force(self.L_F @ q_t, self.L_F @ qdot_t, x.f_normal,
                                   Fourier.adimensional_time_samples)).coefficients

            x.F_int_cache = (self.L_N.T @ F_N + self.L_F.T @ F_F + F_A).reshape(-1, 1)
        return x.F_int_cache

    def _friction_arguments(self, x, ode):
        self._get_F_int(x, ode)
        return (self.L_F @ x.fourier.time_series,
                self.L_F @ x.compute_time_series_derivative(),
                x.f_normal, Fourier.adimensional_time_samples)

    def _get_H(self, x, ode):
        if x.H_kernel is None:
            self._get_F_int(x, ode)                        # populates x.contact_mask
            indicator = x.contact_mask * eye(self.n_N)     # (Nt, n_N, n_N), diagonal
            x.H_kernel = JacobianFourier_Real.new_from_time_series(indicator)
        return x.H_kernel

    def _get_P(self, x, ode):
        if x.P_kernel is None:
            x.P_kernel = JacobianFourier_Real.new_from_time_series(
                ode.jacobian_friction_force_normal(*self._friction_arguments(x, ode)))
        return x.P_kernel

    def _get_GdotF(self, x, ode):
        if x.GdotF is None:
            x.GdotF = JacobianFourier_Real.new_from_time_series(
                ode.jacobian_friction_force_qdot(*self._friction_arguments(x, ode)))
        return x.GdotF

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_F_int(x, ode)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        Nh, n_int = Fourier.number_of_harmonics, self._problem.d_int
        J_A = self._aft.compute_J_int_RI(x, ode)           # G^A + i ω m Gdot^A

        # dF^N/dQ = H [ L^N (eps I - Z_r) - L^N dF^nl,A/dQ ], with Z_r = Y_r^{-1}
        # per harmonic (batched) the dynamic interface stiffness
        Zr = np.linalg.solve(self._problem._get_Yr(x),
                             np.broadcast_to(eye(n_int), (Nh, n_int, n_int)))
        LN_M_RI = block_diag_stack_to_RI(self.L_N @ (self.epsilon * eye(n_int) - Zr))
        J_N = _to_RI(self._get_H(x, ode)) @ (LN_M_RI - self.LN_RI @ J_A)

        # dF^F/dQ = (G^F + i ω m Gdot^F) L^F + P dF^N/dQ; i ω m is a per-harmonic
        # scalar, so it commutes with the right-multiplication by L^F
        G_F = JacobianFourier_Real.new_from_time_series(
            ode.jacobian_friction_force(*self._friction_arguments(x, ode)))
        J_F = (_to_RI(_velocity_scaled_tangent(G_F, self._get_GdotF(x, ode),
                                               x.omega, self.n_F)) @ self.LF_RI
               + _to_RI(self._get_P(x, ode)) @ J_N)

        return self.LN_RI.T @ J_N + self.LF_RI.T @ J_F + J_A

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        problem = self._problem
        dF_A = self._aft.compute_dF_int_domega_RI(x, ode)       # Gdot^A (i m Q_m)

        # how the DLFT source term of the prediction shifts with ω at frozen Q_rel:
        #   Z'_r (F_adm - Q_rel) + Z_r B Y'_m F_ext
        #     = solve(Y_r, B dY/dω (F_ext - B^T z_r))   [Z'_r = -Z_r B Y'_m B^T Z_r]
        BdY = problem.B @ problem._get_dY(x)                    # (Nh, n_int, n_in)
        dlambda_pred = np.linalg.solve(
            problem._get_Yr(x),
            BdY @ (problem.F_ext - problem.B_f.T @ self._get_Zr_rhs(x)))
        # dF^N/dω = H L^N [ that - dF^nl,A/dω ]: f^nl,A sits inside the normal
        # prediction, so it differentiates along with it
        dF_N = _to_RI(self._get_H(x, ode)) @ (
            self.LN_RI @ (Fourier.coefficients_to_RI(dlambda_pred) - dF_A))

        # dF^F/dω = Gdot^F L^F (i m Q_m) + P dF^N/dω
        qdot_adim = Fourier.coefficients_to_RI(x.fourier.get_adimensional_time_derivative())
        dF_F = (_to_RI(self._get_GdotF(x, ode)) @ (self.LF_RI @ qdot_adim)
                + _to_RI(self._get_P(x, ode)) @ dF_N)

        return self.LN_RI.T @ dF_N + self.LF_RI.T @ dF_F + dF_A


class ComplexStiffness(NonlinearMethod):
    """
    First-harmonic joint force from an amplitude-dependent complex stiffness,
    F_int = K*(omega, a) Q^rel_1.

    Assumptions: only the first harmonic is retained; K* is diagonal and each
    k*_j depends only on the physical amplitude a_j = 2/N_t |Q_j| of its own DoF.
    The law k*_j and its derivatives come from the FBS_SystemComplexStiffness.
    """

    def bind(self, problem) -> None:
        """Check that the system has a frequency-domain law and the retained harmonics are [1]."""
        if not isinstance(problem.ode, FBS_SystemComplexStiffness):
            raise TypeError("ComplexStiffness requires an FBS_SystemComplexStiffness, "
                            f"got {type(problem.ode).__name__}")
        if list(Fourier.harmonics) != [1]:
            raise ValueError("ComplexStiffness requires harmonics = [1], "
                             f"got {list(Fourier.harmonics)}")

    def _amplitude(self, x: FourierOmegaPoint) -> np.ndarray:
        """Physical first-harmonic amplitudes a_j = 2/N_t |Q_j|, real, shape (n_int,)."""
        return 2/Fourier.number_of_time_samples * np.abs(x.fourier.coefficients.reshape(-1))

    def _phasor_sq(self, x: FourierOmegaPoint) -> np.ndarray:
        """Squared unit phasors (Q_j/|Q_j|)^2, 0 where Q_j = 0, complex, shape (n_int,)."""
        Q =x.fourier.coefficients.reshape(-1)  # (n_int,) complex
        Qabs = np.abs(Q)
        phasor_sq = np.zeros_like(Q)
        nonzero = Qabs > 0  # boolean mask
        phasor_sq[nonzero] = (Q[nonzero] / Qabs[nonzero]) ** 2
        return phasor_sq

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """Joint force K* Q^rel_1, complex, shape (n_int, 1)."""
        Q = x.fourier.coefficients.reshape(-1)
        amplitude = self._amplitude(x)
        F_int = _per_dof_broadcast(ode.complex_stiffness(x.omega, amplitude), amplitude) * Q
        return F_int.reshape(-1, 1)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        """
        Realified force Jacobian [[Re(D+E), -Im(D-E)], [Im(D+E), Re(D-E)]] with
        D = diag(k* + a/2 dk*/da) and E = diag(a/2 dk*/da (Q/|Q|)^2),
        real, shape (2 n_int, 2 n_int).
        """
        amplitude = self._amplitude(x)
        phasor_sq = self._phasor_sq(x)
        dKstar_damplitude = _per_dof_broadcast(
            ode.complex_stiffness_derivative_amplitude(x.omega, amplitude), amplitude)
        Kstar = _per_dof_broadcast(ode.complex_stiffness(x.omega, amplitude), amplitude)
        D = np.diag(Kstar + amplitude/2 * dKstar_damplitude)
        E = np.diag(amplitude/2 * dKstar_damplitude * phasor_sq)
        DE_plus = D+E
        DE_minus = D-E
        return np.block([[DE_plus.real, -DE_minus.imag],[DE_plus.imag, DE_minus.real]])

    def compute_dF_int_domega_RI(self, x, ode) -> np.ndarray:
        """Realified omega-derivative (dK*/domega) Q^rel_1 as [Re; Im], real, shape (2 n_int, 1)."""
        Q = x.fourier.coefficients.reshape(-1)
        amplitude = self._amplitude(x)
        values = _per_dof_broadcast(
            ode.complex_stiffness_derivative_omega(x.omega, amplitude), amplitude) * Q
        return np.concatenate((values.real, values.imag)).reshape(-1, 1)
