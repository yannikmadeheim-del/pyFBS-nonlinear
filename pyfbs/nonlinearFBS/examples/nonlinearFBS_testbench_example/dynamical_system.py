"""
pyFBS lab testbench A + B, coupled on the 6-DoF virtual-point gap

    q_rel = Bc u = VP_A - VP_B

by one polynomial joint law on all six DoFs:

    f = k q_rel + alpha q_rel^3 + c qdot_rel + beta qdot_rel^3

Solved with DLFTContactAFT. L_N and L_F are empty, so the law is the f^nl,A block.
Units: m, N, rad, s.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System

# --- Data ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

cwd = os.getcwd()
os.chdir(HERE)                          # download_lab_testbench writes to the cwd
try:
    pyfbs.io.download_lab_testbench()   # skips files that are already there
finally:
    os.chdir(cwd)

LAB = HERE / "lab_testbench"
fem = LAB / "FEM"
xlsx = LAB / "Measurements" / "coupling_example.xlsx"

# --- Substructures ------------------------------------------------------------
N_IF = 6                 # [ux, uy, uz, rx, ry, rz]
no_modes = 100
modal_damping = 0.005

MK_A = pyfbs.mck.Model.from_ansys(str(fem / "A.rst"), str(fem / "A.full"), no_modes=no_modes,
                                  allow_pickle=True, recalculate=False, mesh_scale=1)
MK_B = pyfbs.mck.Model.from_ansys(str(fem / "B.rst"), str(fem / "B.full"), no_modes=no_modes,
                                  allow_pickle=True, recalculate=False, mesh_scale=1)

df_chn_A = MK_A.update_locations_df(pd.read_excel(xlsx, sheet_name="Channels_A"), scale=1)
df_imp_A = MK_A.update_locations_df(pd.read_excel(xlsx, sheet_name="Impacts_A"), scale=1)
df_chn_B = MK_B.update_locations_df(pd.read_excel(xlsx, sheet_name="Channels_B"), scale=1)
df_imp_B = MK_B.update_locations_df(pd.read_excel(xlsx, sheet_name="Impacts_B"), scale=1)
df_vp    = pd.read_excel(xlsx, sheet_name="VP_Channels")
df_vpref = pd.read_excel(xlsx, sheet_name="VP_RefChannels")

# --- Virtual points -----------------------------------------------------------
# Tu/Tf are built in the constructor; the modal provider needs nothing else.
vpt_A = pyfbs.interface.VPT(df_chn_A, df_imp_A, df_vp, df_vpref)
vpt_B = pyfbs.interface.VPT(df_chn_B, df_imp_B, df_vp, df_vpref)

nA, nB = vpt_A.tu.shape[0], vpt_B.tu.shape[0]
N = nA + nB

# --- Experimental FRF (optional) ----------------------------------------------
# The spline is evaluated at h*omega, so the grid must reach max(harmonics)*f_hi;
# np.arange excludes the end point, hence + f_resolution.
# f_resolution = 0.1
# f_end = 7 * 500.0 * 1.02 + f_resolution
# MK_A.frf_synth(df_chn_A, df_imp_A, f_start=0, f_end=f_end,
#                f_resolution=f_resolution, modal_damping=modal_damping)
# MK_B.frf_synth(df_chn_B, df_imp_B, f_start=0, f_end=f_end,
#                f_resolution=f_resolution, modal_damping=modal_damping)
# vpt_A.apply_vpt(MK_A.freq, MK_A.frf)
# vpt_B.apply_vpt(MK_B.freq, MK_B.frf)
#
# omega = 2 * np.pi * MK_A.freq
# Y = np.zeros((len(omega), N, N), dtype=complex)
# Y[:, :nA, :nA] = vpt_A.frf
# Y[:, nA:, nA:] = vpt_B.frf

# --- Coupling -----------------------------------------------------------------
# VP DoFs are sorted by grouping (1 < 10 < 100):
# [A ref | A interface (6) | B interface (6) | B ref]
Bc = np.zeros((N_IF, N))
Bc[np.arange(N_IF), np.arange(nA - N_IF, nA)] = 1.0
Bc[np.arange(N_IF), np.arange(nA, nA + N_IF)] = -1.0

out_full = 0                 # first A reference DoF
inp_full = nA + N_IF         # first B reference DoF

# DLFTContactAFT localisations: no normal and no friction DoFs
L_N = np.zeros((0, N_IF))
L_F = np.zeros((0, N_IF))


class TestbenchJoint(FBS_System):
    """f = k q + alpha q^3 + c qdot + beta qdot^3, diagonal on all six VP DoFs."""
    is_real_valued = True

    def __init__(self, k, alpha, c, beta, F0=1.0, sample_number=256):
        self.B_coupling = Bc
        self.k, self.alpha, self.c, self.beta = k, alpha, c, beta
        self.F0 = F0
        self.sample_number = sample_number
        self.out_full, self.inp_full = out_full, inp_full

    def external_term(self, tau):
        f = np.zeros((len(tau), N, 1))
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # --- f^nl,A on the full interface; u_rel_dot is the physical velocity ---
    def interface_force(self, u_rel, u_rel_dot, tau):
        return (self.k * u_rel + self.alpha * u_rel ** 3
                + self.c * u_rel_dot + self.beta * u_rel_dot ** 3)

    def jacobian_interface_force(self, u_rel, u_rel_dot, tau):
        return (self.k + 3 * self.alpha * u_rel ** 2) * np.eye(N_IF)

    def jacobian_interface_force_qdot(self, u_rel, u_rel_dot, tau):
        return (self.c + 3 * self.beta * u_rel_dot ** 2) * np.eye(N_IF)

    # --- f^F on the n_F = 0 friction DoFs ---
    def friction_force(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return np.zeros((len(tau), 0, 1))

    def jacobian_friction_force(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return np.zeros((len(tau), 0, 0))

    def jacobian_friction_force_qdot(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return np.zeros((len(tau), 0, 0))

    def jacobian_friction_force_normal(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return np.zeros((len(tau), 0, 0))
