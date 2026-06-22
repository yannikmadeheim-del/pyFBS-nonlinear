# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
from __future__ import annotations
import numpy as np
import warnings
from numpy import array, concatenate, unique, hstack, array_split, vstack, einsum, pi, linspace, zeros, eye, kron, diag, where, block, zeros_like, vdot, sqrt
from numpy.fft import rfft, irfft, fft, ifft

from scipy.interpolate import CubicSpline, make_interp_spline

# %%
class Fourier(object):
    
    harmonics = unique(array([1,3])) # list of relevant harmonics3
    sample_number = 400
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(abs(harmonics))
    number_of_time_samples = sample_number
    adimensional_time_samples = linspace(0, 2*pi, number_of_time_samples, endpoint=False)
    
    @staticmethod
    def update_class_variables(harmonics: array, sample_number: int):
        indexes = sorted(unique(harmonics, return_index=True)[1])
        Fourier.harmonics = array(harmonics)[indexes] # list of relevant harmonics
        Fourier.sample_number = sample_number

        Fourier.number_of_harmonics = len(Fourier.harmonics)
        Fourier.harmonic_truncation_order = max(abs(Fourier.harmonics))
        Fourier.number_of_time_samples = sample_number
        Fourier.adimensional_time_samples = linspace(0, 2*pi, Fourier.number_of_time_samples, endpoint=False)
        
    def __init__(self, coefficients: array) -> None:
        """
        Creates a Fourier instance from a set of Fourier coefficients
        """
        assert coefficients.shape[0] == Fourier.number_of_harmonics, \
            f"Number of harmonics is {Fourier.number_of_harmonics}, but {len(coefficients)} coefficients were provided."

        self.coefficients = coefficients
        # self.real_part = coefficients.real
        # self.imaginary_part = coefficients.imag
        self.time_series = None
        self.adimensional_time_derivative = None

    def compute_adimensional_time_derivative(self):
        self.adimensional_time_derivative = einsum('i,ijk->ijk', Fourier.harmonics, self.coefficients) * 1j
    
    def get_adimensional_time_derivative(self):
        if self.adimensional_time_derivative is None:
            self.compute_adimensional_time_derivative()
        return self.adimensional_time_derivative
    
    def new_from_time_series(time_series: array):
        pass
    
    def compute_time_series(self):
        pass

    def compute_time_series_derivative(self, x: FourierOmegaPoint):
        pass

    def __add__(self, other):
        return Fourier(coefficients = self.coefficients + other.coefficients)

    def __sub__(self, other):
        return Fourier(coefficients = self.coefficients - other.coefficients)

    def matmul(self, other):
        return Fourier(coefficients = self.coefficients @ other.coefficients)

    def __mul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)

    def __rmul__(self, other: float):
        return Fourier(coefficients = self.coefficients * other)
    
    def __array__(self):
        # (Nh, d, 1) -> (Nh*d, 1) is a plain C-order flatten, so reshape (a free
        # view) replaces the per-harmonic vstack copy.
        R = self.coefficients.real.reshape(-1, 1)
        I = self.coefficients.imag.reshape(-1, 1)
        return concatenate((R, I))
    
    @staticmethod
    def new_from_RI(RI: array):
        complex_dimension = len(RI) // 2
        fourier_C = RI[:complex_dimension] + 1j*RI[complex_dimension:]
        return Fourier(array(array_split(fourier_C, Fourier.number_of_harmonics)))
    
    @staticmethod
    def coefficients_to_RI(coeffs):
        # (Nh, d, 1) -> (Nh*d, 1) flatten via reshape view, then stack [Re; Im].
        R = coeffs.real.reshape(-1, 1)
        I = coeffs.imag.reshape(-1, 1)
        return concatenate((R, I))
    
    @staticmethod
    def zeros(dimension: int):
        return Fourier(zeros((Fourier.number_of_harmonics, dimension, 1), dtype=complex))
    
    @staticmethod
    def new_from_first_harmonic(first_harmonic: array):
        assert 1 in Fourier.harmonics, "Fourier: Harmonic 1 is not in the list of harmonics"
        z1 = first_harmonic * 0.5 * Fourier.number_of_time_samples
        zz = zeros_like(z1)
        zz_fill = [zz if h !=1 else z1 for h in Fourier.harmonics]
        return Fourier(array(zz_fill))
    
class Fourier_Real(Fourier):
    def new_from_time_series(time_series: array):
        """
        Computes the Fourier coefficients (Fourier instance) of a time series by executing the Real Fast Fourier transform (rFFT)
        """
        all_coefficients = rfft(time_series, axis=0)
        new = Fourier(all_coefficients[Fourier.harmonics])
        new.time_series = time_series
        return new
    
    def compute_time_series(self) -> None:
        if self.time_series is None:
            shape = list(self.coefficients.shape)
            shape[0] = Fourier.harmonic_truncation_order + 1
            new_coeff = zeros(shape, dtype=complex)
            new_coeff[Fourier.harmonics] = self.coefficients
            # inverse of Real FFT
            self.time_series = irfft(new_coeff, axis=0, n=Fourier.number_of_time_samples)
        return self.time_series

#%%

def block_diag_stack_to_RI(blocks: array) -> array:
    """Embed a block-diagonal complex operator, given as a (Nh, a, b) stack of
    per-harmonic blocks, into the dense real [[Re, -Im], [Im, Re]] form used by
    the Newton solver. Only interface-sized operators should pass through here:
    the result is a dense (2*Nh*a, 2*Nh*b) matrix.
    """
    Nh, a, b = blocks.shape
    Re = zeros((Nh * a, Nh * b))
    Im = zeros((Nh * a, Nh * b))
    for k in range(Nh):
        Re[k * a:(k + 1) * a, k * b:(k + 1) * b] = blocks[k].real
        Im[k * a:(k + 1) * a, k * b:(k + 1) * b] = blocks[k].imag
    return block([[Re, -Im], [Im, Re]])


class FourierOmegaPoint(object):
    def __init__(self, fourier: Fourier, omega: float):
        self.fourier: Fourier = fourier
        self.omega: float = omega
        self.RI = None
        self.time_series_derivative = None
        self.second_adimensional_time_derivative = None
        self.Gdot = None
        self.Y_cache = None
        self.dY_cache = None         # dY/dω — reused by dR/dω and DLFT dF_int/dω
        self.Z_cache = None
        self.nonlinear_term_cache = None
        self.BY_cache = None        # B @ Y  (FBSProblem: (Nh, n_int, d_total) stack; legacy classes: dense)
        self.BYBT_RI_cache = None   # dense RI form of B @ Y @ B.T  (reused by Jacobian + dR/dω)
        self.Yr_cache = None         # Y_r = B Y B^T  (FBSProblem: (Nh, n_int, n_int) stack; legacy: dense)
        self.Fext_admr_cache = None  # F_adm = B Y f_ext  ((Nh, n_int, 1) stack) — reused by residue and Zr_rhs
        self.Zr_rhs = None           # solve(Y_r, F_adm − Q_rel)  ((Nh, n_int, 1) stack in DLFT methods)
        self.lambda_corrected = None # corrected contact force λ̃ = DFT[max(0, λ_p)]
        self.contact_mask = None     # time-domain Boolean mask m = (λ_p > 0)

    @staticmethod
    def new_from_RI_omega(RI_omega: array):
        # in case omega is not included in the array
        if len(RI_omega) % 2 == 0:
            RI_omega = vstack((RI_omega, 0))
            
        omega = RI_omega[-1,0]
        fourier = Fourier.new_from_RI(RI_omega[:-1])
        return FourierOmegaPoint(fourier=fourier, omega=omega)

    def __add__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier + other.fourier, self.omega + other.omega)

    def __sub__ (self, other):
        if isinstance(other, np.ndarray):
            other = FourierOmegaPoint.new_from_RI_omega(other)
        
        return FourierOmegaPoint(self.fourier - other.fourier, self.omega - other.omega)
    
    def __mul__ (self, other: float| complex):
        return FourierOmegaPoint(self.fourier*other, self.omega*other)
    
    def __array__(self):
        if self.RI is None:
            self.RI = vstack((self.fourier.__array__(), self.omega))
        return self.RI
    
    def adimensional_time_derivative_RI(self) -> array:
        adimensional_time_derivative = self.fourier.get_adimensional_time_derivative()
        R = vstack(adimensional_time_derivative.real)
        I = vstack(adimensional_time_derivative.imag)
        return vstack((R, I, 0.0))
    
    def zero_amplitude(dimension: int, omega: float):
        return FourierOmegaPoint(Fourier.zeros(dimension), omega)
    
    def new_from_first_harmonic(first_harmonic: array, omega: float):
        return FourierOmegaPoint(Fourier.new_from_first_harmonic(first_harmonic), omega)

    def compute_time_series_derivative(self):
        if self.time_series_derivative is None:
            qdot_coefficients = self.omega * self.fourier.get_adimensional_time_derivative()
            qdot_fourier = Fourier_Real(qdot_coefficients)
            Fourier_Real.compute_time_series(qdot_fourier)
            self.time_series_derivative = qdot_fourier.time_series
        return self.time_series_derivative

    def compute_second_adimensional_time_derivative(self):
        if self.second_adimensional_time_derivative is None:
            self.second_adimensional_time_derivative = einsum('i,ijk->ijk', Fourier.harmonics**2, self.fourier.coefficients) * -1
        return self.second_adimensional_time_derivative

#%%

class JacobianFourier(object):

    harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
    harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
    harmonics = unique(concatenate((unique(harmonics_state), unique(harmonics_state_conj))))
    number_of_harmonics = len(harmonics)
    harmonic_truncation_order = max(abs(harmonics))
    
    @staticmethod
    def update_class_variables():
        JacobianFourier.harmonics_state = Fourier.harmonics[:, None] - Fourier.harmonics
        JacobianFourier.harmonics_state_conj = Fourier.harmonics[:, None] + Fourier.harmonics
        JacobianFourier.harmonics = unique(concatenate((unique(JacobianFourier.harmonics_state), unique(JacobianFourier.harmonics_state_conj))))
        JacobianFourier.number_of_harmonics = len(JacobianFourier.harmonics)
        JacobianFourier.harmonic_truncation_order = max(JacobianFourier.harmonics)

    def __init__(self, RR: array, RI: array, IR: array, II: array) -> None:
        self.RR = RR # Derivative of real part wrt real part
        self.RI = RI # Derivative of real part wrt imag part
        self.IR = IR # Derivative of imag part wrt real part
        self.II = II # Derivative of imag part wrt imag part
        
    def new_from_time_series(time_series: array):
        pass
    
    def new_given_all_coefficients(all_coefficients: array):
        pass

class JacobianFourier_Real(JacobianFourier):

    def new_from_time_series(time_series: array):
        """
        Computes the JacobianFourier coefficients given a time series by executing the fast Fourier transform (FFT)
        """
        all_coefficients = fft(time_series, axis=0)
        return JacobianFourier_Real.new_given_all_coefficients(all_coefficients)

    def new_given_all_coefficients(all_coefficients: array):

        state = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state]) # row by row
        state_conj = array([all_coefficients[harmonics] for harmonics in JacobianFourier.harmonics_state_conj])

        plus  = state + state_conj
        minus = state - state_conj

        # Fix: halve the DC column (harmonic m=0). With the rfft convention used
        # throughout pyhbm, the DC bin c_0 = N_t * a_0 has NO factor of 2,
        # unlike c_k = (N_t/2) * (a_k - i b_k) for k >= 1. The Hankel doubling
        # G_{n-m} + G_{n+m} is correct for m >= 1; for m = 0 the two terms are
        # identical (G_n) and naive addition over-counts by factor 2.
        # Identified via FD-Jacobian check in examples/sdof_vibroimpact_validation.
        if 0 in Fourier.harmonics:
            m0 = list(Fourier.harmonics).index(0)
            plus[:, m0, :, :]  *= 0.5
            minus[:, m0, :, :] *= 0.5   # already zero by construction; kept for symmetry

        state_real = hstack(concatenate(plus,  axis=1)) / Fourier.number_of_time_samples
        state_imag = hstack(concatenate(minus, axis=1)) / Fourier.number_of_time_samples

        return JacobianFourier_Real(RR = state_real.real, RI = -state_imag.imag, IR = state_real.imag, II = state_imag.real)
