"""
Subframe + ControlArm, coupled by two rubber bushings whose dynamic stiffness comes
from a measured stiffness map k*_j(f, a_j) per DoF.

Each bushing is condensed to one virtual point with 6 DoF (VPT), so the map acts on
12 interface DoFs:

    q_rel = Bc u = [VP1 | VP2]_Subframe - [VP1 | VP2]_ControlArm

FE model and measurement setup are shared with ../Industrial_Application.
Units as in the FE model: mm, N, rad (stiffness in N/mm and N*mm/rad).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_SystemComplexStiffnessExperimental

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Industrial_Application" / "Subframe_Control_Arm"
fem = DATA / "FEM"
xlsx = DATA / "Measurements" / "measurements.xlsx"

N_IF = 12            # 2 bushings x 6 DoF

# free-free substructures: f_start > 0 skips the rigid-body poles;
# only the first harmonic is retained, so f_end must cover 1.02 * f_hi
no_modes = 100
modal_damping = 0.01
f_start, f_end, f_resolution = 0.1, 2000.0, 1.0

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


# --- Bushing stiffness maps ------------------------------------------------------
def stiffness_map(bushing_1, bushing_2):
    """Stack the maps of both bushings into one table on the 12 interface DoFs.

    Each bushing is a dict with DoF order [ux, uy, uz, rx, ry, rz] in the VP frame:

    - ``freq``: measurement frequencies, shape (P,) [Hz]
    - ``amplitude``: amplitude levels of each DoF, shape (M, 6) [mm or rad]
    - ``K``: complex dynamic stiffness k' + i k'', shape (P, M, 6) [N/mm or N*mm/rad]

    Both bushings must share the frequency grid and the number of amplitude levels M;
    the level values themselves may differ per DoF.

    :param bushing_1: map of bushing 1 (interface DoFs 0:6)
    :param bushing_2: map of bushing 2 (interface DoFs 6:12)
    :return: omega_grid (P,) [rad/s], amplitude_grid (M, 12), K_table (P, M, 12)
    """
    for name, bushing in (("bushing 1", bushing_1), ("bushing 2", bushing_2)):
        if any(np.size(bushing[key]) == 0 for key in ("freq", "amplitude", "K")):
            raise ValueError(f"the stiffness map of {name} is still empty - "
                             f"fill in freq, amplitude and K in main.py")

    freq_1, freq_2 = np.asarray(bushing_1["freq"], float), np.asarray(bushing_2["freq"], float)
    if freq_1.shape != freq_2.shape or not np.allclose(freq_1, freq_2):
        raise ValueError("both bushings must share one frequency grid")
    amplitude_1, amplitude_2 = np.asarray(bushing_1["amplitude"]), np.asarray(bushing_2["amplitude"])
    if amplitude_1.shape[0] != amplitude_2.shape[0]:
        raise ValueError("both bushings must have the same number of amplitude levels")

    omega_grid = 2 * np.pi * freq_1
    amplitude_grid = np.hstack((amplitude_1, amplitude_2))
    K_table = np.concatenate((bushing_1["K"], bushing_2["K"]), axis=2)
    return omega_grid, amplitude_grid, K_table


class Subframe_ControlArmStiffnessMap(FBS_SystemComplexStiffnessExperimental):
    """Both bushings as a tabulated complex stiffness k*_j(omega, a_j), diagonal on all
    12 interface DoFs (see stiffness_map for the table layout)."""

    def __init__(self, omega_grid, amplitude_grid, K_table, F0=1.0, sample_number=256, omega_ref=1.0):
        super().__init__(omega_grid, amplitude_grid, K_table)   # stores, checks, builds the splines
        self.B_coupling = Bc
        self.B_coupling_f = Bf
        self.sample_number = sample_number
        self.omega_ref = omega_ref      # free-free: not the first eigenfrequency (~0 Hz)
        self.out_full, self.inp_full = out_full, inp_full
        self.F0 = F0

    def external_term(self, tau):
        f = np.zeros((len(tau), N_in, 1))      # forces live in the input space
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f
