# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
import numpy as np
from numpy import vdot, asarray
from numpy.linalg import norm, solve

class NewtonRaphson(object):
	def __init__(self, func, jacobian, maximum_iterations: int, absolute_tolerance: float,
				 jacobian_update_frequency: int = 3,
				 jacobian_reuse_delta_threshold: float = 1e-3, relative_tolerance: float = 0.0, stagnation_tolerance: float = 0.0):
		self.compute_residue = func
		self.compute_jacobian = jacobian
		self.maximum_iterations = maximum_iterations
		self.absolute_tolerance = absolute_tolerance
		self.jacobian_update_frequency = jacobian_update_frequency
		self.jacobian_reuse_delta_threshold = jacobian_reuse_delta_threshold

		self.relative_tolerance = relative_tolerance
		self.stagnation_tolerance = stagnation_tolerance
		self._residue0_norm = None

	def is_converged(self) -> bool:
		if norm(self.residue) < self.absolute_tolerance:
			return True
		if self.relative_tolerance > 0 and self._residue0_norm is not None:
			if norm(self.residue) < self.relative_tolerance * self._residue0_norm:
				return True
		return False

	def is_stagnated(self) -> bool:
		if self.stagnation_tolerance > 0:
			return norm(self.delta) < self.stagnation_tolerance * norm(self.x)
		return False
	
	def update_jacobian(self, iteration: int):
		if self._should_update_jacobian(iteration):
			self.jacobian = self.compute_jacobian(self.x)
			self._last_jacobian_update_iter = iteration
	
	def _should_update_jacobian(self, iteration: int) -> bool:
     
		if (iteration - self._last_jacobian_update_iter) >= self.jacobian_update_frequency:
			return True
		
		return norm(self.delta) >= self.jacobian_reuse_delta_threshold * norm(self.x)
	
	def compute_increment(self):
		self.delta = solve(self.jacobian, self.residue)
	
	def update_solution(self):
		"""Backtracking line search (Armijo-style). Falls back to full step
		    if no improvement is found within max_backtracks tries."""
		alpha = 1.0
		norm_old = norm(self.residue)
		max_backtracks = 10
		for _ in range(max_backtracks):
			x_trial = self.x - alpha * self.delta
			try:
				r_trial = self.compute_residue(x_trial)
				if norm(r_trial) < norm_old:
					self.x = x_trial
					self.residue = r_trial  # reuse — saves one func call next iter
					return
			except (np.linalg.LinAlgError, ValueError, FloatingPointError):
				pass
			alpha *= 0.5
		self.x = self.x - self.delta
	
	def get_converged_result(self, iteration: int, return_jacobian: bool):
		if not return_jacobian:
			return self.x, iteration, True
		if iteration == 0:
			self.jacobian = self.compute_jacobian(self.x)
		return self.x, iteration, True, self.jacobian
	
	def get_failed_result(self, return_jacobian: bool):
		return (self.x, self.maximum_iterations, False, self.jacobian) if return_jacobian \
				else (self.x, self.maximum_iterations, False)
	
	def solve(self, initial_guess, return_jacobian: bool = False):
		self.x = initial_guess
		self._last_jacobian_update_iter = -self.jacobian_update_frequency
		self._residue0_norm = None
		
		for iteration in range(self.maximum_iterations):
			self.residue = self.compute_residue(self.x)
			
			if self._residue0_norm is None:
				self._residue0_norm = norm(self.residue)

			if self.is_converged():
				return self.get_converged_result(iteration, return_jacobian)
			
			self.update_jacobian(iteration)
			self.compute_increment()
			self.update_solution()

			if self.is_stagnated():
				return self.get_converged_result(iteration, return_jacobian)
		
		print(f"Newton-Raphson: maximum iterations reached ({self.maximum_iterations})")
		return self.get_failed_result(return_jacobian)

#%%

class CorrectorParameterization(object):
	"""
		Augments the residual with one scalar equation that pins the Newton solution
		to a specific point on the solution curve. Subclasses pick which information
		they actually use from the common kwargs.

		Common kwargs (all four passed by HarmonicBalanceMethod):
		    predictor_vector   : tangent direction along the curve (column vector)
		    predicted_solution : predictor extrapolation point (last + step * predictor_vector)
		    last_solution      : previously converged point
		    step_size          : signed arc-length step magnitude
		"""

	def __init__(self, *, predictor_vector=None, predicted_solution=None,
	             last_solution=None, step_size=None, **_):
		self.predictor_vector = predictor_vector
		self.predicted_solution = predicted_solution
		self.last_solution = last_solution
		self.step_size = step_size

	def compute_parameterization(**kwargs):
		pass

	def compute_jacobian_parameterization(**kwargs):
		pass

class OrthogonalParameterization(CorrectorParameterization):
	"""
	Linear constraint: correction lies in the hyperplane through `predicted_solution`
	orthogonal to `predictor_vector`. Robust through turning points where omega
	is non-monotonic in arc length; quadratic-free, so Newton converges fast.

	g(x) = <predictor_vector, x - predicted_solution>
	"""
	def compute_parameterization(self, point, *args):
		return vdot(self.predictor_vector, point - self.predicted_solution)

	def compute_jacobian_parameterization(self, *args):
		return self.predictor_vector.T

class ArcLengthParameterization(CorrectorParameterization):
	"""
	Keller pseudo-arclength: corrected solution lies on the sphere of radius
	`step_size` around `last_solution`. Quadratic in x — Newton may need more
	iterations at sharp bends.

	g(x) = ||x - last_solution||^2 - step_size^2
	"""
	def compute_parameterization(self, point, *args):
		delta = point - self.last_solution
		return vdot(delta, delta) - self.step_size**2

	def compute_jacobian_parameterization(self, point, *args):
		delta = point - self.last_solution
		return 2 * delta.T
