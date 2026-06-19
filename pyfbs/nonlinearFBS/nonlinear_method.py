import numpy as np
from numpy import eye, vstack, block, kron, diag, zeros, zeros_like, einsum
from abc import ABC, abstractmethod

from .frequency_domain import (
    Fourier, Fourier_Real, FourierOmegaPoint,
    JacobianFourier_Real, block_diag_stack_to_RI,
)


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
        harmonics_term = kron(diag(Fourier.harmonics), eye(ode.dimension))
        col_scale = x.omega * harmonics_term
        return JacobianFourier_Real(
            RR=G.RR + Gdot.RI @ col_scale,
            RI=G.RI - Gdot.RR @ col_scale,
            IR=G.IR + Gdot.II @ col_scale,
            II=G.II - Gdot.IR @ col_scale,
        )

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
            # ALPHA = 1.0e6
            # soft_mask = 0.5 * (1.0 + np.tanh(ALPHA * lambda_p))
            # lambda_t_corr = soft_mask * lambda_p
            # # keep storing the discrete mask for the Jacobian too, or use the smoothed one
            # x.contact_mask = soft_mask  # was: lambda_p > 0

            lambda_x_corr      = Fourier_Real.new_from_time_series(lambda_t_corr)
            x.lambda_corrected = lambda_x_corr.coefficients.reshape(-1, 1)
        return x.lambda_corrected

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask
        n_int = ode.dimension
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
        rhs = BdY @ (problem.F_ext - problem.B.T @ self._get_Zr_rhs(x))
        dlambda_pred = np.linalg.solve(self._get_Yr(x), rhs)   # dλ_p/dω, (Nh, n_int, 1)
        dlambda_pred_fourier = Fourier(dlambda_pred)
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        dlambda_t_corr = np.where(x.contact_mask, dlambda_pred_fourier.time_series, 0.0)
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_v      = dlambda_corr.coefficients.reshape(-1, 1)
        return np.concatenate((dlambda_v.real, dlambda_v.imag))



class DLFTFriction(NonlinearMethod):
    r"""
    Dynamic Lagrangian Frequency-Time (DLFT) unilateral contact WITH Coulomb
    friction (Nacivet, Pierre, Thouverez, Jezequel, JSV 265(1), 2003) in the
    admittance / FBS form used throughout pyhbm.

    The primary Newton unknown is the multiharmonic relative-displacement vector
    ``x_r`` (NOT the contact force, which is slaved to ``x_r`` by a
    prediction-correction at every residue evaluation, exactly as in
    :class:`DLFTContact`).

    DOF layout (per-node ``[N, T...]`` blocks)
    ------------------------------------------
    The ``n_int = ode.dimension`` interface relative DOFs are grouped into
    ``n_contacts`` contiguous blocks of size ``n_dir = 1 + n_tangential``::

        node c  ->  slice(c*n_dir, (c+1)*n_dir),  index 0 = normal,
                    index 1 .. n_tangential = tangential component(s).

    ``n_tangential = 1`` is 1-D sliding (line contact), ``2`` is 2-D sliding
    (the friction cone becomes a disc; all formulae below are already correct in
    that case).

    Sign convention (matches :class:`DLFTContact`, NOT Nacivet's brief)
    ------------------------------------------------------------------
    The normal contact force is POSITIVE in compression and contact is detected
    when the predicted normal multiplier ``s = lambda_u^N > 0`` (separation when
    ``s <= 0``). Hence the friction bound is ``mu * s`` (Nacivet's ``mu*|s|``
    with ``s < 0`` maps to this), and the slip Jacobian N->T coupling is ``+mu*p_hat``
    (his ``-mu*p_hat`` flips with the sign convention).

    Master equations
    -----------------
        Z_r u_r-rhs:        z = solve(Y_r, F_adm - x_r)  = f_r - Z_r u_r
        predictor (time):   lambda_u = IDFT[z] + eps*u_r,   normal -= eps_N*g0
        corrector (time):   sequential stick/slip/separation sweep (see below)
        force:              lambda~  = DFT[lambda]
        residue (FBS):      r(x_r) = x_r + Y_r lambda~ - F_adm   (assembled by FBSProblem)
        Jacobian:           df_c/dx_r = (Gamma+ Jloc Gamma) (E - Z_r)

    Friction is path-dependent (``lambda_x^{k,T}`` depends on the previous
    sample), so the corrector is a SEQUENTIAL sweep over time samples. Following
    Nacivet Fig. 2 / Eqs. (21)-(33), the default is a SINGLE forward sweep per
    residue evaluation, initialized with ``lambda_x^{-1} = 0`` (Eq. 23); the
    period-boundary consistency is left for the outer nonlinear solver to drive
    out (at convergence ``X_r = U_r``, Eq. 16). Setting ``n_sweep > 1`` iterates
    the sweep toward an exactly-periodic tangential force -- a refinement BEYOND
    the paper, useful with plain Newton in heavier slip. The analytical ``Jloc``
    drops the history / periodicity coupling (standard practice,
    Nacivet/Salles/Petrov): the residue is still evaluated exactly, only the
    Newton contraction rate softens.

    NOTE ON THE SOLVER: Nacivet solves ``f({U_r}) = {F_r} - {lambda} - [K_r]{U_r}``
    (Eq. 15, Fig. 2) with a HYBRID POWELL trust-region algorithm (MINPACK hybrd),
    which does not rely on an analytical contact tangent and is robust through
    gross slip. pyhbm instead uses Newton (+ arc-length continuation) with the
    analytical Jacobian below; in 1-D gross slip its T->T block vanishes and
    Newton contracts poorly, so heavy-slip cases favor a Powell-type solver.

    :param epsilon_N: normal penalty / Lagrangian factor (stiffness units; large
        vs. the interface dynamic stiffness, e.g. ~1e2 * k_contact).
    :param epsilon_T: tangential penalty / Lagrangian factor.
    :param mu:        Coulomb friction coefficient.
    :param g_zero:    normal gap offset g0 (contact when x_r^N > g0).
    :param n_tangential: tangential components per contact node (1 or 2).
    :param n_sweep:   max periodicity passes of the sequential corrector.
        Default 1 = single sweep, exactly as Nacivet Fig. 2. >1 iterates the
        sweep toward a periodic tangential force (refinement beyond the paper).
    :param sweep_tol: convergence tolerance for the carried tangential state
        (only relevant when n_sweep > 1).
    """

    def __init__(self, epsilon_N: float = 1.0, epsilon_T: float = 1.0,
                 mu: float = 0.0, g_zero: float = 0.0, n_tangential: int = 1,
                 n_sweep: int = 1, sweep_tol: float = 1e-10):
        self.epsilon_N    = epsilon_N
        self.epsilon_T    = epsilon_T
        self.mu           = mu
        self.g_zero       = g_zero
        self.n_tangential = n_tangential
        self.n_dir        = 1 + n_tangential
        self.n_sweep      = n_sweep
        self.sweep_tol    = sweep_tol
        self._problem     = None    # populated by bind()
        self._eps_vec     = None    # per-DOF penalty vector, built lazily

    def bind(self, problem) -> None:
        self._problem = problem

    # --- DOF bookkeeping -------------------------------------------------

    def _get_eps_vec(self, n_int):
        """Per-DOF penalty vector: eps_N on each normal slot, eps_T on tangential."""
        if self._eps_vec is None:
            assert n_int % self.n_dir == 0, (
                f"n_int={n_int} not divisible by n_dir={self.n_dir} "
                f"(1 + n_tangential)")
            eps = np.empty(n_int)
            for c in range(n_int // self.n_dir):
                base = c * self.n_dir
                eps[base] = self.epsilon_N
                eps[base + 1:base + self.n_dir] = self.epsilon_T
            self._eps_vec = eps
        return self._eps_vec

    # --- internal helpers (use self._problem) ---

    def _get_Yr(self, x):
        return self._problem._get_Yr(x)         # (Nh, n_int, n_int) stack, cached on x

    def _get_Fext_admr(self, x):
        return self._problem._get_Fadm(x)       # (Nh, n_int, 1) stack, cached on x

    def _get_Zr_rhs(self, x):
        # z = solve(Y_r, F_adm - x_r) = f_r - Z_r u_r  (equilibrium contact force),
        # solved harmonic-by-harmonic (batched) on the (Nh, n_int, n_int) stack
        if x.Zr_rhs is None:
            x.Zr_rhs = np.linalg.solve(self._get_Yr(x),
                                       self._get_Fext_admr(x) - x.fourier.coefficients)
        return x.Zr_rhs

    def _corrector_sweep(self, lambda_u):
        """Sequential time-domain stick/slip/separation corrector.

        :param lambda_u: predicted multiplier in time, shape (Nt, n_int, 1).
        :returns: (lam, Jloc) with lam shape (Nt, n_int, 1) the corrected
            contact force, and Jloc shape (Nt, n_int, n_int) the block-diagonal
            per-sample contact tangent dlambda/dlambda_u (history coupling dropped).
        """
        Nt, n_int, _ = lambda_u.shape
        n_dir = self.n_dir
        n_tan = self.n_tangential
        mu    = self.mu
        n_contacts = n_int // n_dir

        lam  = zeros_like(lambda_u)
        Jloc = zeros((Nt, n_int, n_int))
        I_t  = eye(n_tan)

        lu = lambda_u[:, :, 0]   # (Nt, n_int) view for convenience

        for c in range(n_contacts):
            nN = c * n_dir                      # normal slot
            sT = slice(nN + 1, nN + n_dir)      # tangential slots
            lamx_prev_T = zeros(n_tan)          # carried tangential state

            for _pass in range(self.n_sweep):
                lamx_start = lamx_prev_T.copy()
                for k in range(Nt):
                    s = lu[k, nN]                       # predicted normal (>0 == contact)
                    p = lu[k, sT] - lamx_prev_T         # predicted tangential
                    if s <= 0.0:                        # SEPARATION
                        lam[k, nN, 0]   = 0.0
                        lam[k, sT, 0]   = 0.0
                        lamx_prev_T     = lu[k, sT].copy()
                        # Jloc block stays 0
                    else:
                        p_norm = np.sqrt(p @ p)
                        if p_norm < mu * s:             # STICK
                            lam[k, nN, 0] = s
                            lam[k, sT, 0] = p
                            # lamx_prev_T unchanged
                            Jloc[k, nN, nN] = 1.0
                            Jloc[k, sT, sT] = I_t
                        else:                           # SLIP
                            p_hat = p / p_norm
                            lam[k, nN, 0] = s
                            lam[k, sT, 0] = mu * s * p_hat
                            lamx_prev_T   +=  p * (1.0 - mu * s / p_norm)
                            Jloc[k, nN, nN] = 1.0
                            Jloc[k, sT, nN] = mu * p_hat
                            Jloc[k, sT, sT] = (mu * s / p_norm) * (I_t - np.outer(p_hat, p_hat))
                # periodicity check: state entering sample 0 == state leaving Nt-1
                if np.max(np.abs(lamx_prev_T - lamx_start)) < self.sweep_tol:
                    break
        return lam, Jloc

    def _get_lambda_corrected(self, x):
        if x.lambda_corrected is None:
            n_int   = self._problem.ode.dimension
            eps_vec = self._get_eps_vec(n_int)

            # z_r = Y_r^{-1} (F_adm - x_r): the contact force the linear coupled
            # system would need to hold the interface at x_r ("balancing" force)
            zr_fourier = Fourier(self._get_Zr_rhs(x))   # already (Nh, n_int, 1)
            Fourier_Real.compute_time_series(zr_fourier)
            zr_t = zr_fourier.time_series                 # (Nt, n_int, 1)
            Fourier_Real.compute_time_series(x.fourier)
            q_rel = x.fourier.time_series                 # (Nt, n_int, 1)

            # predictor:  lambda_u = IDFT[z] + eps*u_r,  normal slot -= eps_N*g0
            lambda_u = zr_t + eps_vec.reshape(1, n_int, 1) * q_rel
            lambda_u[:, ::self.n_dir, 0] -= self.epsilon_N * self.g_zero

            lam, Jloc = self._corrector_sweep(lambda_u)
            x.contact_mask = Jloc                         # cache per-sample tangent
            lambda_corr    = Fourier_Real.new_from_time_series(lam)
            x.lambda_corrected = vstack(lambda_corr.coefficients)
        return x.lambda_corrected

    # --- NonlinearMethod interface ---

    def compute_F_int(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        return self._get_lambda_corrected(x)

    def compute_J_int_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # populates x.contact_mask = Jloc(t)
        n_int = ode.dimension
        Jloc  = x.contact_mask                           # (Nt, n_int, n_int)
        J_mask    = JacobianFourier_Real.new_from_time_series(Jloc)
        J_mask_RI = block([[J_mask.RR, J_mask.RI], [J_mask.IR, J_mask.II]])
        # Z_r = Y_r^{-1} per harmonic (batched) = dynamic interface stiffness;
        # M = diag(eps) - Z_r, block-diagonal
        Nh = Fourier.number_of_harmonics
        Zr = np.linalg.solve(self._get_Yr(x),
                             np.broadcast_to(eye(n_int), (Nh, n_int, n_int)))
        M  = diag(self._get_eps_vec(n_int)) - Zr         # broadcasts over harmonics
        return J_mask_RI @ block_diag_stack_to_RI(M)

    def compute_dF_int_domega_RI(self, x: FourierOmegaPoint, ode) -> np.ndarray:
        self._get_lambda_corrected(x)                    # ensure x.contact_mask = Jloc(t)
        problem = self._problem
        BdY = problem.B @ problem._get_dY(x)             # B dY/dω, (Nh, n_int, d_total)
        # rhs = B dY/dω (F_ext - B^T z_r): how the predicted multiplier's source
        # term shifts with ω at frozen x_r
        rhs = BdY @ (problem.F_ext - problem.B.T @ self._get_Zr_rhs(x))
        dlambda_pred = np.linalg.solve(self._get_Yr(x), rhs)   # dλ_p/dω, (Nh, n_int, 1)
        dlambda_pred_fourier = Fourier(dlambda_pred)
        Fourier_Real.compute_time_series(dlambda_pred_fourier)
        # per-sample block multiply by the cached contact tangent Jloc(t)
        Jloc = x.contact_mask                            # (Nt, n_int, n_int)
        dlambda_t_corr = einsum('kij,kjl->kil', Jloc, dlambda_pred_fourier.time_series)
        dlambda_corr   = Fourier_Real.new_from_time_series(dlambda_t_corr)
        dlambda_v      = dlambda_corr.coefficients.reshape(-1, 1)
        return np.concatenate((dlambda_v.real, dlambda_v.imag))