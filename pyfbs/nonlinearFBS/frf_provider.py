import numpy as np
from numpy import zeros, eye, array
from numpy.fft import rfft, irfft
from scipy.interpolate import CubicSpline
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
    Receptance only (no frf_type argument is exposed).
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "NumericalFRF takes modal data, not (M, C, K). Use "
            "NumericalFRF.from_modal(angular_eig_freq, eig_vec, modal_damping) or "
            "NumericalFRF.from_ansys_model(model, modal_damping)."
        )

    @classmethod
    def from_modal(cls, angular_eig_freq, eig_vec, modal_damping, omega_ref=None):
        """Build from precomputed REAL modal data + a modal damping RATIO (scalar
        or per-mode). Skips the eigensolve -- proportional/modal damping only."""
        self = cls.__new__(cls)
        self._set_modal(np.asarray(angular_eig_freq, dtype=float),
                        np.asarray(eig_vec), modal_damping, omega_ref)
        return self

    @classmethod
    def from_ansys_model(cls, model, modal_damping, omega_ref=None):
        """Build from a pyFBS Model (e.g. Model.from_ansys), REUSING its cached
        eigensolution (model.angular_eig_freq / model.eig_vec) instead of
        re-solving. Proportional/modal damping only (constant ratio or per-mode)."""
        if getattr(model, "damped_solver", False):
            raise NotImplementedError(
                "from_ansys_model supports proportional (real-mode) damping only; "
                "this model has a complex/damped eigensolution."
            )
        return cls.from_modal(model.angular_eig_freq, model.eig_vec,
                              modal_damping, omega_ref=omega_ref)

    def _set_modal(self, angular_eig_freq, eig_vec, modal_damping, omega_ref):
        self.angular_eig_freq = angular_eig_freq                    # Omega (n_modes,)
        self.eig_vec = eig_vec                                      # (d, n_modes)
        self.modal_damping = modal_damping                         # ratio: scalar or (n_modes,)
        # auto reference: first non-rigid (nonzero) natural frequency, so the
        # continuation runs in nondimensional omega_hat = omega / omega_ref
        # (O(1) axis -> arc-length conditioning). eigh returns Omega ascending.
        if omega_ref is None:
            mx = float(angular_eig_freq.max()) if angular_eig_freq.size else 0.0
            nz = angular_eig_freq[angular_eig_freq > 1e-6 * mx] if mx > 0 else angular_eig_freq
            self.omega_ref = float(nz[0]) if nz.size else 1.0
        else:
            self.omega_ref = float(omega_ref)

    @property
    def n_dofs(self) -> int:
        return self.eig_vec.shape[0]                 # modal basis is (d, n_modes)

    def compute_FRF(self, omega: float, harmonics, out_dofs, in_dofs) -> array:
        # synthesize only Y[out_dofs, in_dofs] at nu_n = n * omega * omega_ref
        # (omega is the nondimensional omega_hat) -> cost flat in mesh size.
        omegas = np.asarray(harmonics, dtype=float) * omega * self.omega_ref
        _, frf = Model.custom_frf_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=omegas,
        )
        return frf                                                 # (Nh, |out|, |in|)

    def compute_FRF_derivative(self, omega: float, harmonics, out_dofs, in_dofs,
                               Y_cache: array) -> array:
        # dY_n/d(omega_hat) = (n * omega_ref) * dH/d(nu) at nu = n*omega*omega_ref
        n = np.asarray(harmonics, dtype=float)
        _, dfrf = Model.custom_dfrf_domega_synth(
            self.angular_eig_freq, self.eig_vec[out_dofs], self.eig_vec[in_dofs],
            modal_damping=self.modal_damping, omegas=n * omega * self.omega_ref,
        )
        return dfrf * (n * self.omega_ref)[:, None, None]          # (Nh, |out|, |in|)


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