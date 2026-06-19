# %%
from pathlib import Path

from dynamical_system import *
from pyfbs.nonlinearFBS import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from time import time

c1, c2, k1, k2, k3, beta, alpha, P = 0.09, 0.09, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0
harmonics = [1, 3, 5, 7, 9]

SAVE_PNG = False   # set True to write the figure to disk

# --- System instances ---
fbs_numerical    = System2DoF_FBS(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)
fbs_experimental = System2DoF_FBS_experimental(c1=c1, c2=c2, k1=k1, k2=k2, k3=k3, beta=beta, alpha=alpha, P=P)

# Must be called before any FrequencyDomain* objects are created
HarmonicBalanceMethod.update_dependencies(harmonics, fbs_numerical.polynomial_degree)

# --- Shared solver settings ---
solver_kwargs = {"maximum_iterations": 200, "absolute_tolerance": fbs_numerical.P * 1e-6}
step_kwargs_FBS   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 1.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
step_kwargs   = {"base": 2, "initial_step_length": 0.01, "maximum_step_length": 1.0,
                 "minimum_step_length": 5e-6, "goal_number_of_iterations": 3}
angular_frequency_range = [0.0, 5.0]
initial_omega = 5.0

# FBS unknowns: u_rel (dimension = 1)
initial_guess_fbs = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=initial_omega)
initial_dir_fbs   = FourierOmegaPoint.zero_amplitude(dimension=fbs_numerical.dimension, omega=-1.0)

# --- FBS numerical ---
t0 = time()
fbs_num_ode    = FBSProblem(
    fbs_numerical,
    NumericalFRF(fbs_numerical.mass_matrix, fbs_numerical.damping_matrix, fbs_numerical.stiffness_matrix),
    AFT())
fbs_num_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_num_ode)
solution_fbs_num = fbs_num_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs_FBS,
)
time_fbs_num = time() - t0
print(f"Time FBS numerical:    {time_fbs_num:.3f} s")

# --- FBS experimental ---
t1 = time()
fbs_exp_ode    = FBSProblem(
    fbs_experimental,
    ExperimentalFRF(fbs_experimental.omega_frf, fbs_experimental.Y_frf, fd_step=1e-6),
    AFT())
fbs_exp_solver = HarmonicBalanceMethod(harmonics=harmonics, freq_domain_ode=fbs_exp_ode)
solution_fbs_exp = fbs_exp_solver.solve_and_continue(
    initial_guess=initial_guess_fbs,
    initial_reference_direction=initial_dir_fbs,
    maximum_number_of_solutions=3500,
    angular_frequency_range=angular_frequency_range,
    solver_kwargs=solver_kwargs,
    step_length_adaptation_kwargs=step_kwargs_FBS,
)
time_fbs_exp = time() - t1
print(f"Time FBS experimental: {time_fbs_exp:.3f} s")

# --- Post-processing: recover full DOF response from FBS ---
# compute_full_response returns Fourier with coefficients shape (Nh, total_dimension, 1)
# DOF 0 = q1 (excited mass), DOF 1 = q2
full_resp_fbs_num = [fbs_num_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_num.fourier, solution_fbs_num.omega)]
full_resp_fbs_exp = [fbs_exp_ode.compute_full_response(f, w)
                     for f, w in zip(solution_fbs_exp.fourier, solution_fbs_exp.omega)]

# --- Helpers ---
def norm_dof(fourier_list, dof):
    return np.array([np.linalg.norm(f.coefficients[:, dof, 0]) for f in fourier_list])

# --- Plot setup ---
solvers = [
    (solution_fbs_num.omega, None,                 full_resp_fbs_num, f'FBS numerical ({time_fbs_num:.2f} s)',         'C1', '--', [0, 1]),
    (solution_fbs_exp.omega, None,                 full_resp_fbs_exp, f'FBS experimental ({time_fbs_exp:.2f} s)',      'C2', ':',  [0, 1]),
]

dof_labels = ['DOF 1 (excited)', 'DOF 2']

fig = plt.figure(figsize=(14, 13))
gs = fig.add_gridspec(4, 6, hspace=0.55, wspace=0.35)
fig.suptitle('FBS vs Reference — Nonlinear 2-DOF Duffing')

# --- Row 0: comparison per DOF (each spanning 3 columns) ---
for col, dof_label in enumerate(dof_labels):
    ax = fig.add_subplot(gs[0, col * 3: col * 3 + 3])
    for omegas, fourier_list, full_list, label, color, ls, dofs in solvers:
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[col]), label=label, color=color, linestyle=ls)
    ax.set_xlabel('ω')
    ax.set_ylabel('||Q||')
    ax.set_title(dof_label)
    ax.legend(fontsize=7)

# --- Rows 1–2: individual subplots per solver (3 cols), one row per DOF ---
for j, dof_label in enumerate(dof_labels):
    for i, (omegas, fourier_list, full_list, label, color, ls, dofs) in enumerate(solvers):
        ax = fig.add_subplot(gs[1 + j, i * 2: i * 2 + 2])
        data = fourier_list if full_list is None else full_list
        ax.plot(omegas, norm_dof(data, dofs[j]), color=color, linestyle=ls)
        ax.set_xlabel('ω')
        ax.set_ylabel('||Q||')
        ax.set_title(f'{label}\n{dof_label}', fontsize=8)

if SAVE_PNG:
    out = Path(__file__).parent / "duffing_FBS_2DoF_frc.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved: {out}")

plt.show()
