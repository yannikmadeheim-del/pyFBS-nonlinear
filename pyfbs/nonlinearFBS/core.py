# Ported into pyFBS from pyhbm (branch vandcard_DLFT, commit 462a081).
# Original Fourier/HBM machinery by Tiago Martins; see https://github.com/tiagomrns/pyhbm.
#%%
import numpy as np
from numpy import array, vstack, hstack, asarray, sqrt
from numpy.linalg import norm
from time import time

from .numerical_continuation.corrector_step import *
from .numerical_continuation.predictor_step import *
from .frequency_domain import *

class SolutionSet(object):
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

	@staticmethod
	def update_dependencies(harmonics: np.ndarray, sample_number: int):
		Fourier.update_class_variables(harmonics, sample_number)
		JacobianFourier.update_class_variables()

	def solve_fixed_frequency(self, initial_guess: FourierOmegaPoint, **solver_kwargs):
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
		residue = self.freq_domain_ode.compute_residue_RI(x)
		parameterization = self.parameterization.compute_parameterization(asarray(x))
		return vstack((residue, parameterization))

	def extended_jacobian(self, x: FourierOmegaPoint):
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

			progress = max(\
        			(solution.omega-angular_frequency_range[0])/(angular_frequency_range[-1]-angular_frequency_range[0]), \
				solution_number/maximum_number_of_solutions)

			if verbose:
				print("progress {:.3f} %".format(100*progress), f"\titerations {iterations}", "\tΔω {:.2e}".format(predictor_vector[-1,0]), end="\r")

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
