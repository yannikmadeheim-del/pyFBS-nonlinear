"""
Subframe + ControlArm, coupled by two rubber bushings.

Each bushing is condensed to one virtual point with 6 DoF (VPT), so the nonlinear
coupling acts on 12 interface DoFs:

    q_rel = Bc u = [VP1 | VP2]_Subframe - [VP1 | VP2]_ControlArm

Units as in the FE model: mm, N (stiffness in N/mm and N*mm/rad).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System

HERE = Path(__file__).resolve().parent
DATA = HERE / "Subframe_Control_Arm"
fem = DATA / "FEM"
xlsx = DATA / "Measurements" / "measurements.xlsx"

N_IF = 12            # 2 bushings x 6 DoF

# free-free substructures: f_start > 0 skips the rigid-body poles;
# f_end must cover max(harmonics) * f_hi
no_modes = 100
modal_damping = 0.01
f_start, f_end, f_resolution = 0.1, 10000.0, 0.1

# --- Substructures ---------------------------------------------------------------
MK_SF = pyfbs.mck.Model.from_ansys(str(fem / "Subframe.rst"), str(fem / "Subframe.full"),
                                   no_modes=no_modes, mesh_scale=1)
MK_CA = pyfbs.mck.Model.from_ansys(str(fem / "ControlArm.rst"), str(fem / "ControlArm.full"),
                                   no_modes=no_modes, mesh_scale=1)

chn_SF = pd.read_excel(xlsx, sheet_name="Channels_Subframe")
imp_SF = pd.read_excel(xlsx, sheet_name="Impacts_Subframe")
chn_CA = pd.read_excel(xlsx, sheet_name="Channels_ControlArm")
imp_CA = pd.read_excel(xlsx, sheet_name="Impacts_ControlArm")
df_vp    = pd.read_excel(xlsx, sheet_name="VP_Channels")
df_vpref = pd.read_excel(xlsx, sheet_name="VP_RefChannels")

chn_SF = MK_SF.update_locations_df(chn_SF)          # snap to FE nodes
imp_SF = MK_SF.update_locations_df(imp_SF)
chn_CA = MK_CA.update_locations_df(chn_CA)
imp_CA = MK_CA.update_locations_df(imp_CA)

MK_SF.frf_synth(chn_SF, imp_SF, f_start=f_start, f_end=f_end,
                f_resolution=f_resolution, modal_damping=modal_damping,
                static_correction=False)
MK_CA.frf_synth(chn_CA, imp_CA, f_start=f_start, f_end=f_end,
                f_resolution=f_resolution, modal_damping=modal_damping,
                static_correction=False)

# --- Virtual points --------------------------------------------------------------
vpt_SF = pyfbs.interface.VPT(chn_SF, imp_SF, df_vp, df_vpref)
vpt_CA = pyfbs.interface.VPT(chn_CA, imp_CA, df_vp, df_vpref)
vpt_SF.apply_vpt(MK_SF.freq, MK_SF.frf)
vpt_CA.apply_vpt(MK_CA.freq, MK_CA.frf)

freq = MK_SF.freq
omega = 2 * np.pi * freq

# channel and impact grids are not collocated, so Y is rectangular:
# rows = channels (outputs), columns = impacts (inputs)
n_out_SF, n_in_SF = vpt_SF.frf.shape[1], vpt_SF.frf.shape[2]     # 15, 12
n_out_CA, n_in_CA = vpt_CA.frf.shape[1], vpt_CA.frf.shape[2]     # 12, 13
N_out = n_out_SF + n_out_CA                                       # 27
N_in  = n_in_SF + n_in_CA                                         # 25

Y = np.zeros((len(freq), N_out, N_in), dtype=complex)
Y[:, :n_out_SF, :n_in_SF] = vpt_SF.frf
Y[:, n_out_SF:, n_in_SF:] = vpt_CA.frf

# --- Coupling --------------------------------------------------------------------
# VPT DoFs per substructure, sorted by grouping: [VP1 (0:6) | VP2 (6:12) | pass-through (12:)]
# Bc (compatibility) acts on the channels, Bf (equilibrium) on the impacts
Bc = np.zeros((N_IF, N_out))
Bc[0:6,  0:6]                        =  np.eye(6)
Bc[0:6,  n_out_SF + 0:n_out_SF + 6]  = -np.eye(6)
Bc[6:12, 6:12]                       =  np.eye(6)
Bc[6:12, n_out_SF + 6:n_out_SF + 12] = -np.eye(6)

Bf = np.zeros((N_IF, N_in))
Bf[0:6,  0:6]                      =  np.eye(6)
Bf[0:6,  n_in_SF + 0:n_in_SF + 6]  = -np.eye(6)
Bf[6:12, 6:12]                     =  np.eye(6)
Bf[6:12, n_in_SF + 6:n_in_SF + 12] = -np.eye(6)

out_full = 12                    # output sensor: Subframe, group 10, x
inp_full = n_in_SF + 12          # excitation: ControlArm, group 20

M_TO_MM = 1e3


def bushing_parameters_from_si(k_trans, k_rot, c_trans, c_rot,
                               alpha_trans=0.0, alpha_rot=0.0, beta_trans=0.0, beta_rot=0.0):
    """Bushing parameters in SI -> per-DoF vectors (12,) in model units (mm, N).

    k [N/m, N*m/rad], c [N*s/m, N*m*s/rad], alpha [N/m^3, N*m/rad^3], beta [N*s^3/m^3, N*m*s^3/rad^3].
    """
    def per_dof(trans, rot):
        return np.array([trans] * 3 + [rot] * 3 + [trans] * 3 + [rot] * 3, dtype=float)

    return dict(k=per_dof(k_trans / M_TO_MM, k_rot * M_TO_MM),
                c=per_dof(c_trans / M_TO_MM, c_rot * M_TO_MM),
                alpha=per_dof(alpha_trans / M_TO_MM**3, alpha_rot * M_TO_MM),
                beta=per_dof(beta_trans / M_TO_MM**3, beta_rot * M_TO_MM))


class Subframe_ControlArm(FBS_System):
    """Diagonal bushing law on all 12 interface DoFs:

        f = k u + c v + alpha u^3 + beta v^3

    k, c, alpha, beta: scalar or (12,), ordered [bushing 1 ux uy uz rx ry rz, bushing 2 ...].
    """
    is_real_valued = True

    def __init__(self, k, c, alpha=0.0, beta=0.0, F0=1.0, sample_number=256, omega_ref=1.0):
        self.B_coupling = Bc
        self.B_coupling_f = Bf
        self.sample_number = sample_number
        self.omega_ref = omega_ref      # free-free: not the first eigenfrequency (~0 Hz)
        self.out_full, self.inp_full = out_full, inp_full
        self.F0 = F0
        self.k = np.broadcast_to(np.asarray(k, float), (N_IF,)).copy()
        self.c = np.broadcast_to(np.asarray(c, float), (N_IF,)).copy()
        self.alpha = np.broadcast_to(np.asarray(alpha, float), (N_IF,)).copy()
        self.beta = np.broadcast_to(np.asarray(beta, float), (N_IF,)).copy()

    def external_term(self, tau):
        f = np.zeros((len(tau), N_in, 1))      # forces live in the input space
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # u_rel_dot is already the physical velocity
    def interface_force(self, u_rel, u_rel_dot, tau):
        u, v = u_rel[:, :, 0], u_rel_dot[:, :, 0]
        f = np.zeros((len(tau), N_IF, 1))
        f[:, :, 0] = self.k * u + self.c * v + self.alpha * u ** 3 + self.beta * v ** 3
        return f

    def jacobian_interface_force(self, u_rel, u_rel_dot, tau):
        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, np.arange(N_IF), np.arange(N_IF)] = self.k + 3 * self.alpha * u_rel[:, :, 0] ** 2
        return J

    def jacobian_interface_force_qdot(self, u_rel, u_rel_dot, tau):
        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, np.arange(N_IF), np.arange(N_IF)] = self.c + 3 * self.beta * u_rel_dot[:, :, 0] ** 2
        return J
