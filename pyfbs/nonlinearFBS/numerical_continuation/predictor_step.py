# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
import numpy as np
from numpy import vdot, sign
from numpy.linalg import norm
from scipy.linalg import null_space


def _scaled_inputs(jacobian, reference_direction, dscale):
    """Physical J, r -> J*diag(sigma), r/sigma."""
    size = jacobian.shape[1]
    scale = np.ones((size, 1)) if dscale is None else np.asarray(dscale, float).reshape(-1, 1)
    if scale.shape != (size, 1) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("dscale must contain one finite positive scale per unknown")
    reference = np.asarray(reference_direction, float).reshape(size, 1) / scale
    return np.asarray(jacobian) * scale.T, reference, scale


def _physical_tangent(tangent, reference, scale):
    """Normalise and orient in scaled coordinates, then return physical t."""
    length = norm(tangent)
    if not np.isfinite(length) or length == 0:
        return None
    tangent = tangent / length
    alignment = float(np.vdot(reference, tangent).real)
    if not np.isfinite(alignment) or alignment == 0:
        return None
    return scale * tangent * (1.0 if alignment > 0 else -1.0)


class Predictor(object):
    def compute_predictor_vector():
        pass


class TangentPredictorRobust(Predictor):
    """Scaled null-space tangent, removing autonomous phase directions."""
    autonomous = True

    @staticmethod
    def compute_predictor_vector(jacobian, reference_direction,
                                 remove_direction=np.array([[]]), dscale=None, **_):
        matrix, reference, scale = _scaled_inputs(jacobian, reference_direction, dscale)
        basis = null_space(matrix)
        dimension = basis.shape[1]
        if dimension == 0:
            return None
        if dimension != 1:
            remove = np.asarray(remove_direction)
            if remove.size == 0:
                return None
            basis = TangentPredictorRobust.filter_directions(basis, dimension, remove / scale)
        if basis is None:
            return None
        return _physical_tangent(basis, reference, scale)

    @staticmethod
    def filter_directions(predictor_vector, dimension_of_kernel, remove_direction):
        if dimension_of_kernel > remove_direction.shape[1] + 1:
            return None
        coordinates = null_space(remove_direction.T @ predictor_vector)
        return predictor_vector @ coordinates if coordinates.shape[1] == 1 else None


class TangentPredictorOne(Predictor):
    """Scaled null-space tangent for a one-dimensional forced branch."""
    autonomous = False

    @staticmethod
    def compute_predictor_vector(jacobian, reference_direction, rcond=None,
                                 dscale=None, **_):
        matrix, reference, scale = _scaled_inputs(jacobian, reference_direction, dscale)
        basis = null_space(matrix, rcond=rcond)
        if basis.shape[1] != 1:
            return None
        return _physical_tangent(basis, reference, scale)


class TangentPredictorTwo(Predictor):
    """Scaled null-space tangent for an autonomous branch with one phase mode."""
    autonomous = True

    @staticmethod
    def compute_predictor_vector(jacobian, reference_direction, remove_direction,
                                 rcond=None, dscale=None, **_):
        matrix, reference, scale = _scaled_inputs(jacobian, reference_direction, dscale)
        basis = null_space(matrix, rcond=rcond)
        if basis.shape[1] != 2:
            return None
        basis = TangentPredictorRobust.filter_directions(
            basis, 2, np.asarray(remove_direction) / scale)
        if basis is None:
            return None
        return _physical_tangent(basis, reference, scale)


class TangentPredictorBordered(Predictor):
    """Solve [J*diag(sigma); (r/sigma).T] t_scaled = [0; 1].

    Normalisation and orientation use the scaled coordinates. The returned
    physical tangent is sigma*t_scaled, so ||t/sigma||=1. A singular bordered
    solve falls back to the null-space projection of the reference direction.
    """
    autonomous = False

    @staticmethod
    def compute_predictor_vector(jacobian, reference_direction, dscale=None, **_):
        matrix, reference, scale = _scaled_inputs(jacobian, reference_direction, dscale)
        rhs = np.zeros((matrix.shape[1], 1))
        rhs[-1] = 1.0
        try:
            tangent = np.linalg.solve(np.vstack((matrix, reference.T)), rhs)
        except np.linalg.LinAlgError:
            basis = null_space(matrix)
            if basis.shape[1] == 0:
                return None
            tangent = basis @ (basis.T @ reference)
        return _physical_tangent(tangent, reference, scale)


class StepLengthAdaptation(object):
    def update_step_length() -> int:
        """Adapt the step length from the last corrector's iteration count.
        Returns 1 when the step is clamped at minimum_step_length, else 0 --
        the driver counts consecutive 1s to declare continuation failure."""
        pass

class ExponentialAdaptation(StepLengthAdaptation):
    """Multiply the step by base**(goal_number_of_iterations - iterations)
    after every corrector solve, clamped to [minimum, maximum] step length."""
    def __init__(self, base, maximum_step_length, minimum_step_length, goal_number_of_iterations, initial_step_length=None):

        assert base > 1.0, "base must be greater than 1"
        self.base = base

        self.max_step_length = maximum_step_length
        self.min_step_length = minimum_step_length
        self.goal_number_of_iterations = goal_number_of_iterations
        
        if initial_step_length is None:
            self.step_length = maximum_step_length
        else:
            self.step_length = initial_step_length

    def update_step_length(self, iterations) -> int:
        delta_iterations = self.goal_number_of_iterations - iterations
        if delta_iterations == 0: return 0
        self.step_length = self.step_length * (self.base**delta_iterations)

        if self.step_length > self.max_step_length:
            self.step_length = self.max_step_length
        
        if self.step_length < self.min_step_length:
            self.step_length = self.min_step_length
            return 1
        
        return 0
        
class BiExponentialAdaptation(StepLengthAdaptation):
    """ExponentialAdaptation with separate growth (base_increase) and shrink
    (base_decrease) bases."""
    def __init__(self, base_increase, maximum_step_length, minimum_step_length, goal_number_of_iterations, initial_step_length=None, base_decrease=None):

        assert base_increase > 1.0, "bases must be greater than 1"
        self.base_increase = base_increase
        
        if base_decrease is None:
            self.base_decrease = base_increase
        else:
            assert base_decrease > 1.0, "bases must be greater than 1"
            self.base_decrease = base_decrease

        self.max_step_length = maximum_step_length
        self.min_step_length = minimum_step_length
        self.goal_number_of_iterations = goal_number_of_iterations
        
        if initial_step_length is None:
            self.step_length = maximum_step_length
        else:
            self.step_length = initial_step_length

    def update_step_length(self, iterations)  -> int:
        delta_iterations = self.goal_number_of_iterations - iterations
        if delta_iterations == 0:
            return 0
        elif delta_iterations > 0:
            self.step_length = self.step_length * (self.base_increase**(delta_iterations))
        else:
            self.step_length = self.step_length / (self.base_decrease**(-delta_iterations))

        if self.step_length > self.max_step_length:
            self.step_length = self.max_step_length
        
        if self.step_length < self.min_step_length:
            self.step_length = self.min_step_length
            return 1
        
        return 0