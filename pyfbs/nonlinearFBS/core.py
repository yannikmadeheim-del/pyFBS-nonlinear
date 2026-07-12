# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
import numpy as np
from numpy import array, vstack, hstack, asarray, sqrt
from numpy.linalg import norm
from time import time

from .numerical_continuation.corrector_step import *
from .numerical_continuation.predictor_step import *
from .frequency_domain import *

class SolutionSet(object):
	"""Converged branch points as index-aligned lists: interface Fourier
	coefficients, angular frequency omega [rad/s], Newton iteration count and
	the continuation step length that produced each point."""
	def __init__(self, solution: FourierOmegaPoint, iterations: int, step_length: float):
		self.fourier = [solution.fourier]
		self.omega = [solution.omega]
		self.iterations = [iterations]
		self.step_length = [step_length]
		
	def append(self, solution: FourierOmegaPoint, iterations: int, step_length: float):
		self.fourier.append(solution.fourier)
		self.omega.append(solution.omega)
		self.iterations.append(iterations)
		self.step_length.append(step_length)
   
	def __len__(self):
		return len(self.omega)


class HarmonicBalanceMethod:
	"""Multi-harmonic balance driver with predictor-corrector continuation.

	Wires a frequency-domain problem (freq_domain_ode, e.g. FBSProblem) to a
	Newton corrector, a corrector parameterization (orthogonal / arc-length),
	a tangent predictor and a step-length adaptation.  The constructor
	(re)sizes the shared Fourier / JacobianFourier class tables via
	update_dependencies.  Dscale (from the problem's omega_ref) only
	conditions the omega column of the extended system -- it does not change
	the branch itself.
	"""
	def __init__(self, harmonics: np.ndarray,
				freq_domain_ode,
				corrector_solver = NewtonRaphson,
				corrector_parameterization: CorrectorParameterization = OrthogonalParameterization,
				predictor: Predictor = TangentPredictorOne,
				step_length_adaptation: StepLengthAdaptation = ExponentialAdaptation):

		ode = freq_domain_ode.ode
		HarmonicBalanceMethod.update_dependencies(harmonics, ode.sample_number)

		self.freq_domain_ode = freq_domain_ode


		self.solver = corrector_solver
		self.corrector_parameterization = corrector_parameterization
		self.predictor = predictor
   
		self.step_length_adaptation = step_length_adaptation

		self.reference_force_level = norm(self.freq_domain_ode.external_term.coefficients)

		self.Dscale = getattr(freq_domain_ode, "Dscale", np.ones(self.freq_domain_ode.real_dimension + 1))

	@staticmethod
	def update_dependencies(harmonics: np.ndarray, sample_number: int):
		Fourier.update_class_variables(harmonics, sample_number)
		JacobianFourier.update_class_variables()

	def solve_fixed_frequency(self, initial_guess: FourierOmegaPoint, **solver_kwargs):
		"""Newton-solve R(x) = 0 at fixed omega (no continuation).

		:returns: (solution, iterations, success, extended_jacobian) with
			extended_jacobian = [dR/dx | dR/domega], reused as the first
			predictor input.
		"""
		solution, iterations, success, jacobian = self.solver(
			func = self.freq_domain_ode.compute_residue_RI, 
			jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI, 
			**solver_kwargs
		).solve(initial_guess, return_jacobian=True)
		
		derivative_omega = self.freq_domain_ode.compute_derivative_wrt_omega_RI(solution)
		
		return \
			solution, \
			iterations, \
			success, \
    		hstack((jacobian, derivative_omega))
		

	def extended_residue(self, x: FourierOmegaPoint):
		"""Residue stacked with the scalar parameterization row that closes the
		(N+1)-unknown continuation system."""
		residue = self.freq_domain_ode.compute_residue_RI(x)
		parameterization = self.parameterization.compute_parameterization(asarray(x))
		return vstack((residue, parameterization))

	def extended_jacobian(self, x: FourierOmegaPoint):
		"""[dR/dx | dR/domega] stacked with the parameterization's Jacobian row."""
		jacobian = self.freq_domain_ode.compute_jacobian_of_residue_RI(x)
		derivative_omega = self.freq_domain_ode.compute_derivative_wrt_omega_RI(x)
		parameterization = self.parameterization.compute_jacobian_parameterization(asarray(x))
		return vstack((hstack((jacobian, derivative_omega)), parameterization))

	def solve_and_continue(
		self, 
		maximum_number_of_solutions, 
		angular_frequency_range, 
		solver_kwargs: dict, 
		step_length_adaptation_kwargs: dict,
		predictor_kwargs: dict = {},
    	initial_guess: FourierOmegaPoint = None, 
		initial_reference_direction: FourierOmegaPoint = None, 
		jacobian_update_frequency: int = 3,
		jacobian_reuse_delta_threshold: float = 1e-3,
		maximum_predictor_corrector_loops_per_solution: int = 10,
		verbose: bool = True
	) -> SolutionSet:
		"""Trace a branch across angular_frequency_range by predictor-corrector
		continuation, appending every converged point to a SolutionSet.

		Stops when the branch leaves the frequency window, the corrector fails
		with the step length locked at its minimum, the predictor fails, or
		maximum_number_of_solutions is reached.

		:param angular_frequency_range: [w_a, w_b] in rad/s (sorted in place).
		:param solver_kwargs: NewtonRaphson settings (maximum_iterations,
			absolute_tolerance, optional relative_tolerance /
			stagnation_tolerance).  absolute_tolerance is rescaled by
			sqrt(2)/Nt so it acts as a time-domain-equivalent tolerance.
		:param step_length_adaptation_kwargs: base, initial/maximum/
			minimum_step_length, goal_number_of_iterations.
		:param predictor_kwargs: forwarded to the predictor (e.g. rcond).
		:param initial_guess: starting FourierOmegaPoint; None = zero amplitude
			at w_a.
		:param initial_reference_direction: orients the first tangent, e.g.
			omega=-1.0 to sweep downward.
		:param jacobian_update_frequency: corrector recomputes the Jacobian
			only every k-th iteration (see NewtonRaphson).
		:param jacobian_reuse_delta_threshold: corrector refreshes the Jacobian
			anyway while the step is large relative to x (see NewtonRaphson).
		:param maximum_predictor_corrector_loops_per_solution: step-halving
			retries per branch point before terminating the continuation.
		:returns: SolutionSet (possibly partial on early termination).
		"""

		t0 = time()
    
		angular_frequency_range.sort()

		solver_kwargs = dict(solver_kwargs)  # avoid mutating the caller's dict
		solver_kwargs["absolute_tolerance"] *= sqrt(2) / Fourier.number_of_time_samples
   
		solver = self.solver(
			func = self.extended_residue, 
			jacobian = self.extended_jacobian, 
			**solver_kwargs,
			jacobian_update_frequency = jacobian_update_frequency,
			jacobian_reuse_delta_threshold = jacobian_reuse_delta_threshold,
			column_scale = self.Dscale
		)

		step_length_adaptation = self.step_length_adaptation(**step_length_adaptation_kwargs)
   
		if initial_guess is None:
			initial_guess = self.zero_initialization(omega=angular_frequency_range[0])
		
		if initial_reference_direction is not None:
			reference_direction = asarray(initial_reference_direction)
		else: 
			reference_direction = asarray(self.zero_initialization(omega=1.0))

		solution, iterations, success, jacobian = self.solve_fixed_frequency(initial_guess, **solver_kwargs)
		solution_set = SolutionSet(solution, iterations, step_length_adaptation.step_length)

		if not success:
			print("\nTerminate: solver failure at initial solution (empty solution set)")
			return solution_set

		for solution_number in range(1, maximum_number_of_solutions):
			
			previous_solution: FourierOmegaPoint = solution
   
			if self.predictor.autonomous:
				phase_shift_direction = previous_solution.adimensional_time_derivative_RI()
				predictor_kwargs["remove_direction"] = phase_shift_direction / norm(phase_shift_direction)
    
			predictor_vector: np.ndarray = self.predictor.compute_predictor_vector(
				jacobian = jacobian[:self.freq_domain_ode.real_dimension],
				reference_direction = reference_direction,
				dscale = self.Dscale,
    			**predictor_kwargs,
          	)
   
			if predictor_vector is None:
				print(f"\nTerminate: predictor failure after {solution_number} solutions")
				print(f"Current omega: {previous_solution.omega}")
				print("Total solving time:", time()-t0, "seconds")
				return solution_set

			count_min_step_length = 1 if step_length_adaptation.step_length == step_length_adaptation.min_step_length else 0
   
			for __ in range(maximum_predictor_corrector_loops_per_solution):
				
				predicted_solution: FourierOmegaPoint = previous_solution + predictor_vector * step_length_adaptation.step_length

				self.parameterization = self.corrector_parameterization(
					predictor_vector=predictor_vector,
					predicted_solution=asarray(predicted_solution),
					last_solution=asarray(previous_solution),
					step_size=step_length_adaptation.step_length,
					dscale = self.Dscale
				)

				solution, iterations, success, jacobian = solver.solve(predicted_solution, return_jacobian=True)
				count_min_step_length += step_length_adaptation.update_step_length(iterations) # do this check before
    
				if success: break
				if count_min_step_length > 1:
					print(f"\nTerminate: solver failure with step size locked at minimum, after {solution_number} solutions")
					print(f"Current omega: {predicted_solution.omega}, step length: {step_length_adaptation.step_length}")
					print("Total solving time:", time()-t0, "seconds")
					return solution_set
    
			else:
				print(f"\nTerminate: solver failure after {solution_number} solutions")
				print(f"Current omega: {predicted_solution.omega}, step length: {step_length_adaptation.step_length}")
				print("Total solving time:", time()-t0, "seconds")
				return solution_set

			solution_set.append(solution, iterations, step_length_adaptation.step_length)

			sweep_upward = initial_reference_direction is None or initial_reference_direction.omega > 0
			if sweep_upward:
				progress = (solution.omega-angular_frequency_range[0])/(angular_frequency_range[-1]-angular_frequency_range[0])
			else:
				progress = (angular_frequency_range[-1]-solution.omega)/(angular_frequency_range[-1]-angular_frequency_range[0])

			if verbose:
				print("progress {:.3f} %".format(100*progress), f"\tsolution points: {solution_number}/{maximum_number_of_solutions}", f"\titerations {iterations}", "\tΔω {:.2e}".format(predictor_vector[-1,0]), end="\r")

			if  not (angular_frequency_range[0] <= solution.omega <= angular_frequency_range[-1]):
				print(f"\nTerminate: outside frequency range after {solution_number+1} solutions")
				print("Total solving time:", time()-t0, "seconds")
				return solution_set

			reference_direction = asarray(solution - previous_solution)

		print("\nTerminate: maximum number of solutions reached")
		print(f"Current omega: {predicted_solution.omega}")
		print("Total solving time:", time()-t0, "seconds")
		return solution_set

	def zero_initialization(self, omega):
		return FourierOmegaPoint.zero_amplitude(dimension=self.freq_domain_ode.d_int, omega=omega)
