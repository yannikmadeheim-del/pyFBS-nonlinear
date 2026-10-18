# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
import numpy as np
from numpy import vdot, asarray
from numpy.linalg import norm, solve

class NewtonRaphson(object):
	"""Damped Newton with lazy Jacobian reuse for the (extended) continuation
	system.

	:param func: residue callable R(x).
	:param jacobian: Jacobian callable J(x).
	:param maximum_iterations: hard iteration cap (failure when hit).
	:param absolute_tolerance: exit when ||R|| drops below this.
	:param jacobian_update_frequency: recompute the Jacobian only every k-th
		iteration (modified Newton) ...
	:param jacobian_reuse_delta_threshold: ... but refresh immediately while
		||delta|| >= threshold * ||x||, i.e. reuse only near convergence.
	:param column_scale: right preconditioner D -- solve (J D) y = R and take
		delta = D y. This improves coordinate conditioning while keeping the
		unknowns and residuals in physical coordinates. None = plain solve.
	:param merit_scale: optional (n+1, 1) coordinate scale of an extended
		system. The line search then divides the first n residual rows (in
		state units) by their scale, so they weigh like the dimensionless last
		row. Convergence is still tested on the unscaled residual. None = plain norm.
	"""
	def __init__(self, func, jacobian, maximum_iterations: int, absolute_tolerance: float,
				 jacobian_update_frequency: int = 3,
				 jacobian_reuse_delta_threshold: float = 1e-3, column_scale=None,
				 merit_scale=None):
		self.compute_residue = func
		self.compute_jacobian = jacobian
		self.maximum_iterations = maximum_iterations
		self.absolute_tolerance = absolute_tolerance
		self.jacobian_update_frequency = jacobian_update_frequency
		self.jacobian_reuse_delta_threshold = jacobian_reuse_delta_threshold

		self.column_scale = None if column_scale is None else asarray(column_scale).reshape(-1, 1)
		self.merit_scale = None if merit_scale is None else asarray(merit_scale).reshape(-1, 1)

	def merit(self, residue) -> float:
		"""Line-search merit: plain norm, or state rows divided by merit_scale."""
		if self.merit_scale is None or len(residue) != len(self.merit_scale):
			return norm(residue)
		scaled = np.array(residue, copy=True)
		scaled[:-1] /= self.merit_scale[:-1]
		return norm(scaled)

	def is_converged(self) -> bool:
		return norm(self.residue) < self.absolute_tolerance
	
	def update_jacobian(self, iteration: int):
		if self._should_update_jacobian(iteration):
			self.jacobian = self.compute_jacobian(self.x)
			self._last_jacobian_update_iter = iteration
	
	def _should_update_jacobian(self, iteration: int) -> bool:
     
		if (iteration - self._last_jacobian_update_iter) >= self.jacobian_update_frequency:
			return True
		
		return norm(self.delta) >= self.jacobian_reuse_delta_threshold * norm(self.x)
	
	def compute_increment(self):
		if self.column_scale is None:
			self.delta = solve(self.jacobian, self.residue)
		else:
			y = solve(self.jacobian * self.column_scale.T, self.residue) # (J*D) y = R
			self.delta = self.column_scale * y							# delta = D*y
	
	def update_solution(self):
		"""Backtracking line search: halve the step until the merit decreases;
		after max_backtracks halvings without improvement, take the full step."""
		alpha = 1.0
		norm_old = self.merit(self.residue)
		max_backtracks = 10
		for _ in range(max_backtracks):
			x_trial = self.x - alpha * self.delta
			try:
				r_trial = self.compute_residue(x_trial)
				if self.merit(r_trial) < norm_old:
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
		
		for iteration in range(self.maximum_iterations):
			self.residue = self.compute_residue(self.x)

			if not np.isfinite(norm(self.residue)):
				# Overflow -> Inf -> (rfft) -> NaN contamination. Every exit test
				# below compares with '<', which NaN/Inf can never satisfy, so the
				# remaining iteration budget would be burned on NaN arithmetic.
				if iteration == 0:
					self.jacobian = self.compute_jacobian(self.x)
				print(f"Newton-Raphson: non-finite residue at iteration {iteration}, aborting corrector step")
				return self.get_failed_result(return_jacobian)
			
			if self.is_converged():
				return self.get_converged_result(iteration, return_jacobian)
			
			self.update_jacobian(iteration)
			self.compute_increment()
			self.update_solution()
		
		print(f"Newton-Raphson: maximum iterations reached ({self.maximum_iterations})")
		return self.get_failed_result(return_jacobian)

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
	             last_solution=None, step_size=None, dscale = None, **_):
		self.predictor_vector = predictor_vector
		self.predicted_solution = predicted_solution
		self.last_solution = last_solution
		self.step_size = step_size
		self.dinv2 = None if dscale is None else (1.0 / asarray(dscale).reshape(-1, 1)) ** 2

	def compute_parameterization(**kwargs):
		pass

	def compute_jacobian_parameterization(**kwargs):
		pass

class OrthogonalParameterization(CorrectorParameterization):
	"""
	Linear constraint: correction lies in the hyperplane through `predicted_solution`
	orthogonal to `predictor_vector`. Robust through turning points where omega
	is non-monotonic in arc length; quadratic-free, so Newton converges fast.

	g(x) = t^T D^{-2} (x - predicted_solution), D = diag(dscale).
	"""
	def compute_parameterization(self, point, *args):
		delta = point - self.predicted_solution
		if self.dinv2 is None:
			return vdot(self.predictor_vector, delta)
		# orthogonality in the scaled metric: t^T D^{-2} (x - x_pred) = 0
		return float((self.predictor_vector * self.dinv2 * delta).sum())

	def compute_jacobian_parameterization(self, *args):
		if self.dinv2 is None:
			return self.predictor_vector.T
		return (self.dinv2 * self.predictor_vector).T

class ArcLengthParameterization(CorrectorParameterization):
	"""
	Keller pseudo-arclength: corrected solution lies on the sphere of radius
	`step_size` around `last_solution`. Quadratic in x — Newton may need more
	iterations at sharp bends.

	g(x) = ||(x - last_solution) / dscale||^2 - step_size^2
	"""
	def compute_parameterization(self, point, *args):
		delta = point - self.last_solution
		if self.dinv2 is None:
			return vdot(delta, delta) - self.step_size**2
		return float((delta*self.dinv2*delta).sum()) - self.step_size**2

	def compute_jacobian_parameterization(self, point, *args):
		delta = point - self.last_solution
		if self.dinv2 is None:
			return 2 * delta.T
		return 2 * (self.dinv2 * delta).T
