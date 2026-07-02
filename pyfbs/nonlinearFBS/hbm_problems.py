import numpy as np
from numpy import eye

from .frequency_domain import Fourier, Fourier_Real, FourierOmegaPoint, block_diag_stack_to_RI
from .frf_provider import FRFProvider
from .nonlinear_method import NonlinearMethod


class FBSProblem:
    """
    HBM problem for Frequency Based Substructuring (FBS).

    Newton unknown: Q_rel — interface DOFs, size n_int = fbs.B_coupling.shape[0].
    All frequency-domain operators are kept as per-harmonic stacks: the
    admittance Y is (Nh, d_total, d_total), the interface admittance
    Y_r = B Y B^T is (Nh, n_int, n_int). Dense matrices are only assembled
    at interface size for the Newton solver.

    Residual:  R = Q_rel + Y_r F_nl - F_adm,      F_adm = B Y F_ext
    Jacobian:  J = I + (Y_r)_RI J_nl_RI
    dR/dω:     B dY (B^T F_nl - F_ext) + Y_r dF_nl/dω   (complex, then to RI)
    """

    def __init__(self, fbs, frf_provider: FRFProvider, method: NonlinearMethod):
        self.ode = fbs
        self.frf_provider = frf_provider
        self.method = method

        # The residual R = Q_rel + Y_r F_nl - F_adm lives in displacements, so
        # the provider must synthesize receptance; mobility/accelerance would
        # silently solve a different (wrong) problem.
        frf_type = getattr(frf_provider, "frf_type", "receptance")
        if frf_type != "receptance":
            raise ValueError(
                f"FBSProblem requires an FRF provider with frf_type='receptance', "
                f"got '{frf_type}'. Non-receptance synthesis is meant for "
                f"standalone FRF evaluation, not for the FBS solver.")

        self.d_int   = fbs.B_coupling.shape[0]      # n_int: interface DOFs, from the Boolean matrix
        self.d_total = frf_provider.n_dofs          # full DOF count, from the FRF/Ansys modal data
        Nh = Fourier.number_of_harmonics
        self.complex_dimension = Nh * self.d_int
        self.real_dimension    = 2 * self.complex_dimension
        omega_ref = float(getattr(fbs, 'omega_ref', 1.0))
        self.Dscale = np.ones(self.real_dimension + 1)
        self.Dscale[-1] = omega_ref

        external_ts = fbs.external_term(Fourier.adimensional_time_samples)
        self.external_term = Fourier_Real.new_from_time_series(external_ts)
        B_full     = fbs.B_coupling                          # (n_int, d_total)
        F_ext_full = self.external_term.coefficients         # (Nh, d_total, 1)

        # the Boolean matrix and the FRF data must agree on the full DOF count
        assert B_full.shape[1] == self.d_total, (
            f"B_coupling has {B_full.shape[1]} columns but the FRF provider "
            f"exposes {self.d_total} DOFs")

        # Reduce to the DoFs the interface (B columns) and excitation (F_ext rows)
        # actually touch.  The FRF provider then synthesizes only this small
        # (in_dofs x in_dofs) block per eval -> cost flat in mesh size.  The full
        # response (compute_full_response) requests Y[all_dofs, in_dofs].
        idof = np.nonzero(np.any(B_full != 0.0, axis=0))[0]              # interface DoFs
        xdof = np.nonzero(np.any(F_ext_full[:, :, 0] != 0.0, axis=0))[0] # excitation DoFs
        self.in_dofs  = np.union1d(idof, xdof)                           # sorted unique
        self.all_dofs = np.arange(self.d_total)
        self.B     = B_full[:, self.in_dofs]                             # (n_int, n_in)
        self.F_ext = F_ext_full[:, self.in_dofs, :]                      # (Nh, n_in, 1)

        self.method.bind(self)

    def _get_FRF(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.Y_cache is None:
            x.Y_cache = self.frf_provider.compute_FRF(
                x.omega, Fourier.harmonics, self.in_dofs, self.in_dofs
            )                                                   # (Nh, n_in, n_in)
        return x.Y_cache

    def _get_BY(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.BY_cache is None:
            x.BY_cache = self.B @ self._get_FRF(x)          # (Nh, n_int, d_total)
        return x.BY_cache

    def _get_Yr(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.Yr_cache is None:
            x.Yr_cache = self._get_BY(x) @ self.B.T         # (Nh, n_int, n_int)
        return x.Yr_cache

    def _get_Fadm(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.Fext_admr_cache is None:
            x.Fext_admr_cache = self._get_BY(x) @ self.F_ext   # (Nh, n_int, 1)
        return x.Fext_admr_cache

    def _get_BYBT_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.BYBT_RI_cache is None:
            x.BYBT_RI_cache = block_diag_stack_to_RI(self._get_Yr(x))
        return x.BYBT_RI_cache

    def _get_dY(self, x: FourierOmegaPoint) -> np.ndarray:
        if x.dY_cache is None:
            x.dY_cache = self.frf_provider.compute_FRF_derivative(
                x.omega, Fourier.harmonics, self.in_dofs, self.in_dofs, self._get_FRF(x))
        return x.dY_cache

    def compute_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Q_rel = x.fourier.coefficients                                  # (Nh, n_int, 1)
        Fnl   = self.method.compute_F_int(x, self.ode).reshape(Q_rel.shape)
        R     = Q_rel + self._get_Yr(x) @ Fnl - self._get_Fadm(x)       # per harmonic
        return np.concatenate((R.real.reshape(-1, 1), R.imag.reshape(-1, 1)))

    def compute_jacobian_of_residue_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        J_nl_RI = self.method.compute_J_int_RI(x, self.ode)
        return eye(self.real_dimension) + self._get_BYBT_RI(x) @ J_nl_RI

    def compute_derivative_wrt_omega_RI(self, x: FourierOmegaPoint) -> np.ndarray:
        Fnl = self.method.compute_F_int(x, self.ode).reshape(
            Fourier.number_of_harmonics, self.d_int, 1)
        BdY = self.B @ self._get_dY(x)                                  # (Nh, n_int, d_total)
        dF_nl_dw_RI = self.method.compute_dF_int_domega_RI(x, self.ode)
        # reinterpret the RI vector [Re; Im] as a complex per-harmonic stack
        dF_nl_dw = (dF_nl_dw_RI[:self.complex_dimension]
                    + 1j * dF_nl_dw_RI[self.complex_dimension:]).reshape(Fnl.shape)
        dR = BdY @ (self.B.T @ Fnl - self.F_ext) + self._get_Yr(x) @ dF_nl_dw
        return np.concatenate((dR.real.reshape(-1, 1), dR.imag.reshape(-1, 1)))

    def compute_full_response(self, fourier: Fourier, omega: float) -> Fourier:
        """Post-processing: full response for all d_total DOFs from a converged x_r.

        Uses Y[all_dofs, in_dofs] (one extra provider call, once per converged point)
        so every physical DOF is recovered while the Newton loop stays reduced.
        """
        x   = FourierOmegaPoint(fourier, omega)
        Fnl = self.method.compute_F_int(x, self.ode).reshape(
            Fourier.number_of_harmonics, self.d_int, 1)
        Y_full = self.frf_provider.compute_FRF(
            omega, Fourier.harmonics, self.all_dofs, self.in_dofs)     # (Nh, d_total, n_in)
        Q_full = Y_full @ (self.F_ext - self.B.T @ Fnl)               # (Nh, d_total, 1)
        return Fourier(Q_full)

