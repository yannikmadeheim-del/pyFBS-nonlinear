# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
import numpy as np
from numpy import vdot, sign
from numpy.linalg import norm
from scipy.linalg import null_space

class Predictor(object):
    def compute_predictor_vector():
        pass

class TangentPredictorRobust(Predictor):
    """
    Robust tangent predictor
    The dimension of the kernel is verified
    It is not the fastest if the dimension of the kernel is known apriori
    """
    autonomous = True
    @staticmethod
    def compute_predictor_vector(jacobian: np.ndarray, 
                                 reference_direction: np.ndarray, 
                                 remove_direction=np.array([[]])) -> np.ndarray:
        
        predictor_vector: np.ndarray = null_space(jacobian)
        dimension_of_kernel: int = predictor_vector.shape[1]
        
        if dimension_of_kernel != 1:
            
            if dimension_of_kernel == 0:
                print(f"TangentPredictorRobust: could not find any predictor directions")
                return None
            
            predictor_vector = TangentPredictorRobust.filter_directions(predictor_vector, 
                                                                        dimension_of_kernel, 
                                                                        remove_direction)
        
        if predictor_vector is None:
            return None
            
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector * step_length
    
    @staticmethod
    def filter_directions(predictor_vector: np.ndarray, 
                          dimension_of_kernel: int, 
                          remove_direction: np.ndarray):
        
        remove_dimension: int = remove_direction.shape[1]   
         
        if dimension_of_kernel > remove_dimension + 1:
            print(f"TangentPredictorRobust: encoutered {dimension_of_kernel} possible predictor directions and could only remove {remove_dimension}")
            return None
            
        alignment_to_remove: np.ndarray = remove_direction.T @ predictor_vector
        coordinates_filter: np.ndarray = null_space(alignment_to_remove)
        
        if coordinates_filter.shape[1] == 1:
            return predictor_vector @ coordinates_filter
        
        print(f"TangentPredictorRobust: encoutered {dimension_of_kernel} possible predictor directions. \
                After filering, {coordinates_filter.shape[1]} directions remained because the remove directions were not independent")
        
        return None
        
class TangentPredictorOne(Predictor):
    """
    Less robust than TangentPredictorRobust 
    The dimension of the kernel is always assumed to be 1
    It is designed for non-autonomous ODEs
    Use at your own risk
    """
    autonomous = False
    @staticmethod
    def compute_predictor_vector(jacobian: np.ndarray, 
                                 reference_direction: np.ndarray,
                                 rcond: float = None) -> np.ndarray:
        
        predictor_vector: np.ndarray = null_space(jacobian, rcond=rcond)
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector
        
class TangentPredictorTwo(Predictor):
    """
    Less robust than TangentPredictorRobust 
    The dimension of the kernel is always assumed to be 2
    It is designed for autonomous ODEs
    Use at your own risk
    """
    autonomous = True
    @staticmethod
    def compute_predictor_vector(jacobian: np.ndarray, 
                                 reference_direction: np.ndarray, 
                                 remove_direction: np.ndarray,
                                 rcond: float = None) -> np.ndarray:
        """
        rcond : float, optional
            -> Relative condition number. Singular values s smaller than rcond * max(s) are considered zero. Default: floating point eps * max(M,N).
        """
        
        predictor_vector: np.ndarray = null_space(jacobian, rcond=rcond)
        dimension_of_kernel: int = predictor_vector.shape[1]
        if not dimension_of_kernel == 2: 
            print(f"TangentPredictorTwo: for rcond={rcond}, dimension_of_kernel={dimension_of_kernel}")
            return None

        # remove one direction
        alignment_to_remove: np.ndarray = remove_direction.T @ predictor_vector
        predictor_vector = predictor_vector @ np.array([-alignment_to_remove[:,1], alignment_to_remove[:,0]])
            
        # normalize predictor_vector
        predictor_vector /= norm(predictor_vector)
        # align predictor_vector with reference
        predictor_vector *= sign(vdot(reference_direction, predictor_vector))
        # scale to match step length
        return predictor_vector

class TangentPredictorBordered(Predictor):
    """Keller (1983) §6: tangent from the bordered system.
    Solve  [A  B; ṫ_prevᵀ]  [t_x; t_λ]  =  [0; 1]
    where ṫ_prev is the previous tangent. Stable when A becomes
    rank-deficient at limit points."""
    autonomous = False

    @staticmethod
    def compute_predictor_vector(jacobian, reference_direction, **_):
        # jacobian = [∂r/∂x_r | ∂r/∂ω], shape (N, N+1)
        # reference_direction = previous tangent, shape (N+1, 1)
        n_plus_one = jacobian.shape[1]
        bordered = np.vstack((jacobian, reference_direction.T))  # (N+1, N+1)

        rhs = np.zeros((n_plus_one, 1))
        rhs[-1, 0] = 1.0

        try:
            tangent = np.linalg.solve(bordered, rhs)
        except np.linalg.LinAlgError:
            return None    # 𝒜 itself singular: true bifurcation, fall through

        tangent /= np.linalg.norm(tangent)
        tangent *= np.sign(np.vdot(reference_direction, tangent))
        return tangent



#%%

class StepLengthAdaptation(object):
    def update_step_length() -> int:
        pass

class ExponentialAdaptation(StepLengthAdaptation):
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
        
        # self.step_length = min(max(new_step_length, self.min_step_length), self.max_step_length)
        
        if self.step_length > self.max_step_length:
            self.step_length = self.max_step_length
        
        if self.step_length < self.min_step_length:
            self.step_length = self.min_step_length
            return 1
        
        return 0
        
class BiExponentialAdaptation(StepLengthAdaptation):
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
            
        # self.step_length = min(max(new_step_length, self.min_step_length), self.max_step_length)
        
        if self.step_length > self.max_step_length:
            self.step_length = self.max_step_length
        
        if self.step_length < self.min_step_length:
            self.step_length = self.min_step_length
            return 1
        
        return 0