import numpy as np
from numpy import zeros, eye, array
from numpy.fft import rfft, irfft
from scipy.interpolate import CubicSpline
from scipy.linalg import block_diag
from abc import ABC, abstractmethod
from ..mck.mck import Model
from .frequency_domain import Fourier, FourierOmegaPoint


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
      - ``NumericalFRF.from_modal(angular_eig_freq, eig_vec, modal_damping, omega_ref=None)``
      - ``NumericalFRF.from_ansys_model(model, modal_damping, omega_ref=None)``
    The provider itself NEVER eigensolves -- the modes come from the caller (a
    system's own ``eigh``, or ``Model.from_ansys``).  Each query evaluates the
    modal-sum receptance (and its omega-derivative) at the EXACT requested
    frequencies via pyFBS's ``custom_frf_synth`` / ``custom_dfrf_domega_synth``.
    Exact for proportional/modal damping (constant or per-mode ratio).
    The synthesis setting ``limit_modes`` (mode truncation) is stored on the
    provider and forwarded to every synthesis call.  The FRF type is fixed to
    ``receptance`` -- the displacement admittance :class:`FBSProblem` requires.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "NumericalFRF takes modal data, not (M, C, K). Use "
            "NumericalFRF.from_modal(angular_eig_freq, eig_vec, modal_damping) or "
            "NumericalFRF.from_ansys_model(model, modal_damping)."
        )

    @classmethod
    def from_modal(cls, angular_eig_freq, eig_vec, modal_damping,
                   limit_modes=None):
        """Build from precomputed REAL modal data + a modal damping RATIO (scalar
        or per-mode). Skips the eigensolve -- proportional/modal damping only.

        :param limit_modes: number of modes used in the synthesis (``None`` = all)
        :type limit_modes: int or None
        """
        self = cls.__new__(cls)
        self._set_modal(np.asarray(angular_eig_freq, dtype=float),
                        np.asarray(eig_vec), modal_damping,
                        limit_modes=limit_modes)
        return self

    @classmethod
    def from_ansys_model(cls, model, modal_damping,
                         limit_modes=None):
        """Build from a pyFBS Model (e.g. Model.from_ansys), REUSING its cached
        eigensolution (model.angular_eig_freq / model.eig_vec) instead of
        re-solving. Proportional/modal damping only (constant ratio or per-mode).
        ``limit_modes`` as in :meth:`from_modal`."""
        if getattr(model, "damped_solver", False):
            raise NotImplementedError(
                "from_ansys_model supports proportional (real-mode) damping only; "
                "this model has a complex/damped eigensolution."
            )
        return cls.from_modal(model.angular_eig_freq, model.eig_vec,
                              modal_damping,
                              limit_modes=limit_modes)

    def _set_modal(self, angular_eig_freq, eig_vec, modal_damping,
                   limit_modes=None):
        self.angular_eig_freq = angular_eig_freq                    # Omega (n_modes,)
        self.eig_vec = eig_vec                                      # (d, n_modes)
        self.modal_damping = modal_damping                         # ratio: scalar or (n_modes,)
        self.limit_modes = limit_modes                             # mode truncation (None = all)
        self.frf_type = "receptance"

    @property
    def n_dofs(self) -> int:
        return self.eig_vec.shape[0]                 # modal basis is (d, n_modes)

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        omegas = np.asarray(harmonics, dtype=float) * omega
        _, frf = Model.custom_frf_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=omegas,
            limit_modes=self.limit_modes, frf_type=self.frf_type,
        )
        return frf                                                 # (Nh, |out|, |in|)

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        n = np.asarray(harmonics, dtype=float)
        _, dfrf = Model.custom_dfrf_domega_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=n * omega,
            limit_modes=self.limit_modes, frf_type=self.frf_type,
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
    """

    def __init__(self, angular_eig_freq, eig_vec_chn, eig_vec_imp, modal_damping):
        self.angular_eig_freq = np.asarray(angular_eig_freq, dtype=float)  # Omega (n_modes,)
        self.eig_vec_chn = np.asarray(eig_vec_chn)   # Psi_c (N_vp, n_modes): VP-projected OUTPUT modes
        self.eig_vec_imp = np.asarray(eig_vec_imp)   # Psi_i (N_vp, n_modes): VP-projected INPUT  modes
        self.modal_damping = modal_damping           # ratio: scalar or (n_modes,)
        self.frf_type = "receptance"
        if self.eig_vec_chn.shape != self.eig_vec_imp.shape:
            raise ValueError(
                "eig_vec_chn (Psi_c) and eig_vec_imp (Psi_i) must share shape "
                f"(N_vp, n_modes); got {self.eig_vec_chn.shape} and {self.eig_vec_imp.shape}")

    @classmethod
    def from_substructures(cls, subsystems, modal_damping,
                           limit_modes=None):
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
        """
        Psi_c, Psi_i, Omega, damp = [], [], [], []
        for model, vpt, df_chn, df_imp in subsystems:
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
        return cls(np.concatenate(Omega),
                   block_diag(*Psi_c), block_diag(*Psi_i),
                   np.concatenate(damp))

    @property
    def n_dofs(self) -> int:
        return self.eig_vec_chn.shape[0]             # N_vp

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        omegas = np.asarray(harmonics, dtype=float) * omega
        _, frf = Model.custom_frf_synth(
            self.angular_eig_freq, self.eig_vec_chn[out_dofs], self.eig_vec_imp[in_dofs],
            modal_damping=self.modal_damping, omegas=omegas,
            frf_type=self.frf_type,
        )
        return frf                                   # (Nh, |out|, |in|), already in VP space

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        n = np.asarray(harmonics, dtype=float)
        _, dfrf = Model.custom_dfrf_domega_synth(
            self.angular_eig_freq, self.eig_vec_chn[out_dofs], self.eig_vec_imp[in_dofs],
            modal_damping=self.modal_damping, omegas=n * omega,
            frf_type=self.frf_type,
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
    """

    def __init__(self, omega_frf: array, Y: array, fd_step: float = 1e-6):
        self.omega_frf = omega_frf
        self.Y = Y
        self.fd_step = fd_step
        self._spline_cache = {}   # (out_key, in_key) -> (CubicSpline real, CubicSpline imag)

    @property
    def n_dofs(self) -> int:
        return self.Y.shape[1]                       # Y is (N_freq, d, d)

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
        return self._interpolate(np.asarray(harmonics, dtype=float) * omega,
                                 out_dofs, in_dofs)              # (Nh, |out|, |in|)

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        h = self.fd_step
        return (self.compute_FRF(omega + h, harmonics, out_dofs, in_dofs)
                - self.compute_FRF(omega - h, harmonics, out_dofs, in_dofs)) / (2 * h)


