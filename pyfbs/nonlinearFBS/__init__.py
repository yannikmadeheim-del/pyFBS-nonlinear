# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
"""Nonlinear FBS solver ported from pyhbm: multi-harmonic balance + numerical
path continuation with AFT and DLFT normal contact, consuming FRFs through the
FRFProvider interface (NumericalFRF / ExperimentalFRF)."""

from .dynamical_system import FBS_System

from .frequency_domain import (
    Fourier,
    Fourier_Real,
    FourierOmegaPoint,
    JacobianFourier,
    JacobianFourier_Real,
    block_diag_stack_to_RI,
)

from .frf_provider import (
    FRFProvider,
    NumericalFRF,
    ExperimentalFRF,
    ModalVPFRF,
)

from .nonlinear_method import (
    NonlinearMethod,
    AFT,
    DLFTContact,
    DLFTFriction,
)

from .hbm_problems import FBSProblem

from .numerical_continuation.corrector_step import (
    NewtonRaphson,
    CorrectorParameterization,
    OrthogonalParameterization,
    ArcLengthParameterization,
)

from .numerical_continuation.predictor_step import (
    Predictor,
    TangentPredictorRobust,
    TangentPredictorOne,
    TangentPredictorTwo,
    TangentPredictorBordered,
    StepLengthAdaptation,
    ExponentialAdaptation,
    BiExponentialAdaptation,
)

from .core import (
    SolutionSet,
    HarmonicBalanceMethod,
)

__all__ = [
    "FBS_System",
    "Fourier", "Fourier_Real", "FourierOmegaPoint",
    "JacobianFourier", "JacobianFourier_Real", "block_diag_stack_to_RI",
    "FRFProvider", "NumericalFRF", "ExperimentalFRF", "ModalVPFRF",
    "NonlinearMethod", "AFT", "DLFTContact", "DLFTFriction",
    "FBSProblem",
    "NewtonRaphson", "CorrectorParameterization",
    "OrthogonalParameterization", "ArcLengthParameterization",
    "Predictor", "TangentPredictorRobust", "TangentPredictorOne",
    "TangentPredictorTwo", "TangentPredictorBordered",
    "StepLengthAdaptation", "ExponentialAdaptation", "BiExponentialAdaptation",
    "SolutionSet", "HarmonicBalanceMethod",
]
