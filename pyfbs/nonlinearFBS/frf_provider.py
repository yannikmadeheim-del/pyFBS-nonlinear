import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from numpy import zeros, eye, array
from numpy.fft import rfft, irfft
from scipy.interpolate import CubicSpline
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from ..mck.mck import Model, _active_static_dofs
from .frequency_domain import Fourier, FourierOmegaPoint


def _validate_static_correction(enabled):
    if not isinstance(enabled, (bool, np.bool_)):
        raise TypeError("static_correction must be True or False")
    return bool(enabled)


def _rigid_mode_mask(angular_eig_freq):
    eigenvalues = np.abs(np.asarray(angular_eig_freq)) ** 2
    tolerance = np.sqrt(np.finfo(float).eps) * np.max(eigenvalues)
    return eigenvalues <= tolerance


def _retained_static_flexibility(omega, phi_out, phi_in, flexible):
    return (phi_out[:, flexible] / np.asarray(omega)[flexible] ** 2) @ phi_in[:, flexible].T


def _residual_matrix(matrix, shape):
    if matrix is None:
        raise ValueError(
            "static_correction=True requires residual_flexibility when only "
            "modal data are supplied; use a model-based constructor to compute it.")
    matrix = np.asarray(matrix)
    if matrix.shape != shape:
        raise ValueError(f"residual_flexibility must have shape {shape}, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("residual_flexibility must contain finite values")
    return matrix.copy()


class _ModelStaticFlexibility:
    """Selected entries of constrained FE static flexibility; no dense FE inverse."""

    def __init__(self, model, rigid):
        if model.k is None:
            raise ValueError("Static correction requires the physical stiffness matrix k")
        self.dimension = model.eig_vec.shape[0]
        stiffness = sparse.csc_matrix(model.k)
        if stiffness.shape != (self.dimension, self.dimension):
            raise ValueError("Stiffness dimensions must match the physical modal basis")
        self.active = _active_static_dofs(stiffness, model.m)
        self.lookup = np.full(self.dimension, -1, dtype=int)
        self.lookup[self.active] = np.arange(len(self.active))
        stiffness = stiffness[self.active][:, self.active]
        self.matrix = stiffness
        if np.any(rigid):
            if model.m is None:
                raise ValueError("Free-free static correction requires the mass matrix")
            constraint = (sparse.csc_matrix(model.m) @ model.eig_vec[:, rigid])[self.active]
            self.matrix = sparse.bmat(
                [[stiffness, constraint], [constraint.conj().T, None]], format="csc")
        self.factor = None
        try:
            import pypardiso
        except ImportError:
            pypardiso = None
        self.pardiso = pypardiso

    def project(self, out_dofs, in_dofs):
        rhs = np.zeros((self.matrix.shape[0], len(in_dofs)), dtype=self.matrix.dtype)
        inp = self.lookup[in_dofs]
        rhs[inp[inp >= 0], np.flatnonzero(inp >= 0)] = 1.0
        if self.pardiso is not None and self.matrix.dtype == np.float64:
            try:
                solution = self.pardiso.spsolve(self.matrix.tocsr(), rhs, squeeze=False)
            finally:
                self.pardiso.ps.free_memory(everything=True)
        else:
            if self.factor is None:
                try:
                    self.factor = splu(self.matrix)
                except RuntimeError as error:
                    raise ValueError(
                        "Static correction requires a nonsingular constrained "
                        "stiffness system") from error
            solution = self.factor.solve(rhs)
        out = self.lookup[out_dofs]
        result = np.zeros((len(out_dofs), len(in_dofs)), dtype=solution.dtype)
        result[out >= 0] = solution[out[out >= 0]]
        return result


class FRFProvider(ABC):
    """
    Computes the per-harmonic admittance Y_n(omega) and its omega-derivative,
    restricted to requested output/input DoFs: a stack of shape (Nh, |out|, |in|),
    one block per harmonic n.  (out_dofs, in_dofs are index arrays into the full DoF
    set; the solver passes the small interface/excitation set in the Newton loop.)
    """

    @abstractmethod
    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        """Return Y[out_dofs, in_dofs] per harmonic, shape (Nh, |out|, |in|), complex."""

    @abstractmethod
    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        """Return dY/domega at [out_dofs, in_dofs], shape (Nh, |out|, |in|), complex."""

    @property
    @abstractmethod
    def n_dofs(self) -> int:
        """Number of physical DOFs in the underlying model (the full DoF set that
        out_dofs / in_dofs index into)."""


class NumericalFRF(FRFProvider):
    """
    FRF synthesized by mode superposition from a REAL modal basis.

    Build it from precomputed modal data with one of the constructors:
      - ``NumericalFRF.from_modal(angular_eig_freq, eig_vec, modal_damping)``
      - ``NumericalFRF.from_ansys_model(model, modal_damping)``
    The provider itself NEVER eigensolves -- the modes come from the caller (a
    system's own ``eigh``, or ``Model.from_ansys``).  Each query evaluates the
    modal-sum receptance (and its omega-derivative) at the EXACT requested
    frequencies via pyFBS's ``custom_frf_synth`` / ``custom_dfrf_domega_synth``.
    Exact for proportional/modal damping (constant or per-mode ratio).
    ``rayleigh_damping=(alpha, beta)`` instead uses the shared synthesis law
    ``C=alpha*M+beta*K``. Leave ``modal_damping=None`` in that case; rigid modes
    retain the mass-proportional term, without forming a singular damping ratio.
    The synthesis setting ``limit_modes`` (mode truncation) is stored on the
    provider and forwarded to every synthesis call.  The FRF type is fixed to
    ``receptance`` -- the displacement admittance :class:`FBSProblem` requires.

    ``static_correction=True`` adds frequency-independent residual flexibility.
    ``from_ansys_model`` computes requested entries from the constrained physical
    stiffness system and caches them. ``from_modal`` instead requires an explicit
    ``residual_flexibility`` matrix in the coordinates of ``eig_vec``. Static
    correction contributes zero to the analytical frequency derivative.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "NumericalFRF takes modal data, not (M, C, K). Use "
            "NumericalFRF.from_modal(angular_eig_freq, eig_vec, modal_damping) or "
            "NumericalFRF.from_ansys_model(model, modal_damping)."
        )

    @classmethod
    def from_modal(cls, angular_eig_freq, eig_vec, modal_damping=None,
                   limit_modes=None, static_correction=False,
                   residual_flexibility=None, rayleigh_damping=None):
        """Build from precomputed REAL modal data + a modal damping RATIO (scalar
        or per-mode). Skips the eigensolve -- proportional/modal damping only.

        :param limit_modes: number of modes used in the synthesis (``None`` = all)
        :type limit_modes: int or None
        :param static_correction: add the supplied residual flexibility.
        :param residual_flexibility: static flexibility minus the static modal
            contribution of the retained flexible modes, shape ``(n_dofs, n_dofs)``.
        :param rayleigh_damping: ``(alpha, beta)`` for ``C=alpha*M+beta*K``;
            mutually exclusive with ``modal_damping``. Units: 1/s, s.
        """
        self = cls.__new__(cls)
        self._set_modal(np.asarray(angular_eig_freq, dtype=float),
                        np.asarray(eig_vec), modal_damping,
                        limit_modes=limit_modes, rayleigh_damping=rayleigh_damping)
        self.static_correction = _validate_static_correction(static_correction)
        if self.static_correction:
            self.residual_flexibility = _residual_matrix(
                residual_flexibility, (self.n_dofs, self.n_dofs))
        return self

    @classmethod
    def from_ansys_model(cls, model, modal_damping=None,
                         limit_modes=None, static_correction=False,
                         rayleigh_damping=None):
        """Build from a pyFBS Model (e.g. Model.from_ansys), REUSING its cached
        eigensolution (model.angular_eig_freq / model.eig_vec) instead of
        re-solving. Proportional/modal damping only (constant ratio or per-mode).
        ``limit_modes`` as in :meth:`from_modal`. With ``static_correction=True``,
        compute static flexibility from ``model.k`` (and ``model.m`` to constrain
        free-free rigid modes), subtract the retained flexible-mode contribution,
        and cache requested blocks."""
        if getattr(model, "damped_solver", False):
            raise NotImplementedError(
                "from_ansys_model supports proportional (real-mode) damping only; "
                "this model has a complex/damped eigensolution."
            )
        enabled = _validate_static_correction(static_correction)
        provider = cls.from_modal(model.angular_eig_freq, model.eig_vec,
                                  modal_damping, limit_modes=limit_modes,
                                  rayleigh_damping=rayleigh_damping)
        if enabled:
            rigid = _rigid_mode_mask(model.angular_eig_freq)
            provider.static_correction = True
            provider._flexible_modes = ~rigid
            provider._static_model = _ModelStaticFlexibility(model, rigid)
        return provider

    def _set_modal(self, angular_eig_freq, eig_vec, modal_damping,
                   limit_modes=None, rayleigh_damping=None):
        count = len(angular_eig_freq) if limit_modes is None else limit_modes
        Model._modal_damping_rates(angular_eig_freq[:count], modal_damping, rayleigh_damping)
        self.angular_eig_freq = angular_eig_freq                    # Omega (n_modes,)
        self.eig_vec = eig_vec                                      # (d, n_modes)
        self.modal_damping = modal_damping                         # ratio: scalar or (n_modes,)
        self.rayleigh_damping = rayleigh_damping                    # (alpha [1/s], beta [s])
        self.limit_modes = limit_modes                             # mode truncation (None = all)
        self.frf_type = "receptance"
        self.static_correction = False
        self.residual_flexibility = None
        self._static_model = None
        self._static_cache = {}

    def _get_static_correction(self, out_dofs, in_dofs):
        if self.residual_flexibility is not None:
            return self.residual_flexibility[np.ix_(out_dofs, in_dofs)]
        key = (tuple(out_dofs), tuple(in_dofs))
        if key not in self._static_cache:
            count = len(self.angular_eig_freq) if self.limit_modes is None else self.limit_modes
            exact = self._static_model.project(out_dofs, in_dofs)
            retained = _retained_static_flexibility(
                self.angular_eig_freq[:count], self.eig_vec[out_dofs, :count],
                self.eig_vec[in_dofs, :count], self._flexible_modes[:count])
            self._static_cache[key] = exact - retained
        return self._static_cache[key]

    @property
    def n_dofs(self) -> int:
        return self.eig_vec.shape[0]                 # modal basis is (d, n_modes)

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        omegas = np.asarray(harmonics, dtype=float) * omega
        _, frf = Model.custom_frf_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=omegas,
            limit_modes=self.limit_modes, frf_type=self.frf_type,
            rayleigh_damping=self.rayleigh_damping,
        )
        if self.static_correction:
            frf += self._get_static_correction(out_dofs, in_dofs)[None]
        return frf                                                 # (Nh, |out|, |in|)

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        n = np.asarray(harmonics, dtype=float)
        _, dfrf = Model.custom_dfrf_domega_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=n * omega,
            limit_modes=self.limit_modes, frf_type=self.frf_type,
            rayleigh_damping=self.rayleigh_damping,
        )
        return dfrf * n[:, None, None]          # (Nh, |out|, |in|)


class ModalVPFRF(FRFProvider):
    """
    Virtual-point admittance by mode superposition, with the Virtual Point
    Transformation (VPT) FOLDED INTO THE MODE SHAPES.

    Because the VPT (Tu on responses, Tf on loads) is a constant spatial
    projection and the modal denominator is diagonal in the modes and
    independent of DoF, Tu/Tf commute through the modal sum::

        Y_vp = Tu (Phi_c D(w) Phi_i^T) Tf = (Tu Phi_c) D(w) (Tf^T Phi_i)^T
             =     Psi_c            D(w)      Psi_i^T

    So we store the VP-projected participation matrices ``Psi_c = Tu @ Phi_c``
    (outputs) and ``Psi_i = Tf^T @ Phi_i`` (inputs); the poles ``angular_eig_freq``
    and ``modal_damping`` are the UNCHANGED physical ones (the VPT moves mode
    shapes, not poles). Each query synthesizes DIRECTLY in the N_vp virtual-point
    space -- the physical admittance ``Phi_c D Phi_i^T`` is never assembled, so the
    cost is O(n_modes * N_vp^2), flat in mesh size and below even forming Y_phys.

    ``n_dofs == N_vp`` equals the column count of the FBS Boolean matrix, so a
    plain :class:`FBSProblem` consumes it unchanged (no VPT branch in the solver).

    ``from_substructures(..., static_correction=True)`` computes residual
    flexibility once per substructure and transforms it with the same Tu/Tf.
    Direct construction requires the already transformed ``residual_flexibility``
    when static correction is enabled. Its frequency derivative is zero.
    ``rayleigh_damping=(alpha, beta)`` passes the same physical damping law to
    the shared synthesis and its analytical derivative, before or after VPT.
    """

    def __init__(self, angular_eig_freq, eig_vec_chn, eig_vec_imp, modal_damping=None,
                 static_correction=False, residual_flexibility=None,
                 rayleigh_damping=None):
        self.angular_eig_freq = np.asarray(angular_eig_freq, dtype=float)  # Omega (n_modes,)
        Model._modal_damping_rates(self.angular_eig_freq, modal_damping, rayleigh_damping)
        self.eig_vec_chn = np.asarray(eig_vec_chn)   # Psi_c (N_vp, n_modes): VP-projected OUTPUT modes
        self.eig_vec_imp = np.asarray(eig_vec_imp)   # Psi_i (N_vp, n_modes): VP-projected INPUT  modes
        self.modal_damping = modal_damping           # ratio: scalar or (n_modes,)
        self.rayleigh_damping = rayleigh_damping     # (alpha [1/s], beta [s])
        self.frf_type = "receptance"
        # Only the mode count has to agree. The DoF dimension may differ: with
        # non-collocated channel/impact grids the admittance is rectangular.
        if self.eig_vec_chn.shape[1] != self.eig_vec_imp.shape[1]:
            raise ValueError(
                "eig_vec_chn (Psi_c) and eig_vec_imp (Psi_i) must share the mode "
                f"count; got {self.eig_vec_chn.shape} and {self.eig_vec_imp.shape}")
        self.static_correction = _validate_static_correction(static_correction)
        self.residual_flexibility = None
        if self.static_correction:
            self.residual_flexibility = _residual_matrix(
                residual_flexibility, (self.n_dofs_out, self.n_dofs_in))

    @classmethod
    def from_substructures(cls, subsystems, modal_damping=None,
                           limit_modes=None, static_correction=False,
                           rayleigh_damping=None):
        """Fold each substructure's VPT into its mode shapes, then block-diagonal-stack.

        The block-diagonal stack keeps the substructures uncoupled (a substructure's
        modes get zero participation on the others' VP DoFs), exactly mirroring the
        block-diagonal ``Y = diag(Y_A, Y_B)`` assembled in the current example.

        :param subsystems: list of ``(model, vpt, df_chn, df_imp)`` per substructure:
            ``model`` a pyFBS Model with a REAL modal solution; ``vpt`` a
            ``pyfbs.interface.VPT`` built from ``(df_chn, df_imp)``; ``df_chn``/``df_imp``
            the location-updated physical channel/impact dataframes the VPT consumes.
        :param modal_damping: modal damping ratio (scalar or per-mode) for synthesis.
        :param limit_modes: number of modes kept PER SUBSTRUCTURE (``None`` = all).
            Truncation must happen here, before stacking: the stacked basis lists
            substructure A's modes first, then B's, so a global "first k" cut
            after stacking would drop whole substructures instead of high modes.
        :param static_correction: compute physical residual flexibility from each
            model's stiffness matrix and project it with that substructure's VPT.
        :param rayleigh_damping: common ``(alpha, beta)`` for all substructures;
            use instead of modal ratios. Units: 1/s, s.
        """
        enabled = _validate_static_correction(static_correction)
        Psi_c, Psi_i, Omega, damp, residuals = [], [], [], [], []
        for model, vpt, df_chn, df_imp in subsystems:
            if rayleigh_damping is not None and getattr(model, "damped_solver", False):
                raise ValueError("Rayleigh damping requires real, undamped mode shapes")
            if enabled and getattr(model, "damped_solver", False):
                raise ValueError("Static correction requires real, undamped mode shapes")
            eigval2, d_modal, Phi_c = model.transform_modal_parameters(
                df_chn, limit_modes=limit_modes, modal_damping=modal_damping,
                return_channel_only=True)
            _, _, Phi_i = model.transform_modal_parameters(
                df_imp, limit_modes=limit_modes, modal_damping=modal_damping,
                return_channel_only=True)
            Psi_c.append(vpt.tu @ Phi_c)        # (N_vp_s, n_modes_s)
            Psi_i.append(vpt.tf.T @ Phi_i)      # (N_vp_s, n_modes_s)
            Omega.append(np.sqrt(eigval2))
            damp.append(np.atleast_1d(d_modal))
            if enabled:
                rigid = _rigid_mode_mask(model.angular_eig_freq)
                exact = model._projected_static_flexibility(
                    df_chn, df_imp, rigid_modes=model.eig_vec[:, rigid])
                retained = _retained_static_flexibility(
                    np.sqrt(eigval2), Phi_c, Phi_i, ~rigid[:len(eigval2)])
                residuals.append(vpt.tu @ (exact - retained) @ vpt.tf)
        return cls(np.concatenate(Omega),
                   block_diag(*Psi_c), block_diag(*Psi_i),
                   np.concatenate(damp) if rayleigh_damping is None else modal_damping,
                   static_correction=enabled,
                   residual_flexibility=block_diag(*residuals) if enabled else None,
                   rayleigh_damping=rayleigh_damping)

    @property
    def n_dofs(self) -> int:
        return self.eig_vec_chn.shape[0]             # N_vp, alias of n_dofs_out

    @property
    def n_dofs_out(self) -> int:
        return self.eig_vec_chn.shape[0]             # channels (rows)

    @property
    def n_dofs_in(self) -> int:
        return self.eig_vec_imp.shape[0]             # impacts (columns)

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        omegas = np.asarray(harmonics, dtype=float) * omega
        _, frf = Model.custom_frf_synth(
            self.angular_eig_freq, self.eig_vec_chn[out_dofs], self.eig_vec_imp[in_dofs],
            modal_damping=self.modal_damping, omegas=omegas,
            frf_type=self.frf_type,
            rayleigh_damping=self.rayleigh_damping,
        )
        if self.static_correction:
            frf += self.residual_flexibility[np.ix_(out_dofs, in_dofs)][None]
        return frf                                   # (Nh, |out|, |in|), already in VP space

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        n = np.asarray(harmonics, dtype=float)
        _, dfrf = Model.custom_dfrf_domega_synth(
            self.angular_eig_freq, self.eig_vec_chn[out_dofs], self.eig_vec_imp[in_dofs],
            modal_damping=self.modal_damping, omegas=n * omega,
            frf_type=self.frf_type,
            rayleigh_damping=self.rayleigh_damping,
        )
        return dfrf * n[:, None, None]               # (Nh, |out|, |in|)


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
    interpolation : {'cubic', 'none'}
        'none' requires every harmonic query to coincide with a stored sample.
        It is intended for fixed-frequency sweeps and supplies no derivative.
    """

    def __init__(self, omega_frf: array, Y: array, fd_step: float = 1e-6,
                 interpolation: str = "cubic"):
        self.omega_frf = np.asarray(omega_frf, dtype=float)
        self.Y = np.asarray(Y)
        if (self.omega_frf.ndim != 1 or not len(self.omega_frf)
                or not np.all(np.isfinite(self.omega_frf))
                or np.any(self.omega_frf < 0) or np.any(np.diff(self.omega_frf) <= 0)):
            raise ValueError("omega_frf must be finite, nonnegative and strictly increasing")
        if self.Y.ndim != 3 or self.Y.shape[0] != len(self.omega_frf):
            raise ValueError("Y must have shape (N_freq, N_output, N_input)")
        if interpolation not in ("cubic", "none"):
            raise ValueError("interpolation must be 'cubic' or 'none'")
        if interpolation == "cubic" and len(self.omega_frf) < 2:
            raise ValueError("Cubic interpolation requires at least two samples")
        self.interpolation = interpolation
        self.fd_step = fd_step
        self._spline_cache = {}   # (out_key, in_key) -> (CubicSpline real, CubicSpline imag)

    @property
    def n_dofs(self) -> int:
        return self.Y.shape[1]                       # alias of n_dofs_out

    @property
    def n_dofs_out(self) -> int:
        return self.Y.shape[1]                       # channels (rows)

    @property
    def n_dofs_in(self) -> int:
        return self.Y.shape[2]                       # impacts (columns)

    def _splines(self, out_dofs, in_dofs):
        # project-then-interpolate: spline ONLY the requested Y[:, out, in] channels
        # (cached per DoF set), so the per-query cost is flat in the measured DoF count.
        key = (tuple(np.atleast_1d(out_dofs)), tuple(np.atleast_1d(in_dofs)))
        if key not in self._spline_cache:
            Yr = self.Y[:, out_dofs][:, :, in_dofs]              # (N_freq, |out|, |in|)
            self._spline_cache[key] = (CubicSpline(self.omega_frf, Yr.real),
                                       CubicSpline(self.omega_frf, Yr.imag))
        return self._spline_cache[key]

    def _interpolate(self, omega, out_dofs, in_dofs) -> array:
        """Interpolate the reduced channels at omega; Y(-omega) = conj(Y(omega))."""
        ir, ii = self._splines(out_dofs, in_dofs)
        omega = np.asarray(omega)
        neg_mask = omega < 0
        omega_abs = np.abs(omega)

        if np.any(omega_abs > self.omega_frf[-1]):
            import warnings
            warnings.warn(
                f"omega outside FRF data range [0, {self.omega_frf[-1]:.4f}]. Extrapolating."
            )

        result = ir(omega_abs) + 1j * ii(omega_abs)

        if np.ndim(omega) == 0:
            if bool(neg_mask):
                result = np.conj(result)
        elif np.any(neg_mask):
            result[neg_mask] = np.conj(result[neg_mask])

        return result

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        if self.interpolation == "none":
            queries = np.asarray(harmonics, dtype=float) * omega
            values = np.abs(queries)
            upper = np.clip(np.searchsorted(self.omega_frf, values), 0, len(self.omega_frf) - 1)
            lower = np.maximum(upper - 1, 0)
            indices = np.where(abs(self.omega_frf[upper] - values) <=
                               abs(self.omega_frf[lower] - values), upper, lower)
            if not np.all(np.isclose(self.omega_frf[indices], values, rtol=1e-10, atol=1e-10)):
                raise ValueError("interpolation='none' requires every harmonic frequency to match an FRF sample")
            result = self.Y[indices][:, np.atleast_1d(out_dofs)][:, :, np.atleast_1d(in_dofs)].copy()
            result[queries < 0] = result[queries < 0].conj()
            return result
        return self._interpolate(np.asarray(harmonics, dtype=float) * omega,
                                 out_dofs, in_dofs)              # (Nh, |out|, |in|)

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        if self.interpolation == "none":
            raise ValueError("Sample-only FRFs have no frequency derivative; use solve_frequency_sweep")
        h = self.fd_step
        return (self.compute_FRF(omega + h, harmonics, out_dofs, in_dofs)
                - self.compute_FRF(omega - h, harmonics, out_dofs, in_dofs)) / (2 * h)


