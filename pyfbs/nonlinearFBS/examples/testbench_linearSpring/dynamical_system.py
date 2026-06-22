import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System
N_IF = 6   # virtual-point interface DoFs: [ux, uy, uz, rx, ry, rz]

def build_testbench_data(k_trans=1.0e6, k_rot=1.0e3, c_trans=0.0, c_rot=0.0,
                         f_resolution=1.0):
    """
    :param f_resolution: FRF frequency resolution Delta f [Hz], passed to frf_synth
        and preserved by the VPT (which re-projects per frequency, never resamples).
        Smaller = denser sampling: density = 1 / f_resolution samples per Hz. Default
        1.0 (the pyFBS default); use e.g. 0.1 to resolve the lightly-damped resonances
        (Q ~ 167 -> ~1.4 Hz wide) that 1 Hz sampling renders as 1-point spikes.
    :returns: dict with
        freq      (N_freq,)       measurement frequencies [Hz]
        omega     (N_freq,)       angular frequencies 2*pi*freq [rad/s]
        Y         (N_freq, N, N)  uncoupled admittance diag(Y_A, Y_B) [m/N]
        nA, nB, N                 A / B / total DoF counts
        Bc        (6, N)          signed-Boolean VP coupling (A_if - B_if)
        k_diag, c_diag (6,)       per-DoF spring stiffness / damping
        out_full  int             response DoF (first A reference DoF = 0)
        inp_full  int             force DoF    (first B reference DoF = nA + 6)
    """
    pyfbs.io.download_lab_testbench()
    pos_xlsx = r"./lab_testbench/Measurements/coupling_example.xlsx"

    df_chn_A = pd.read_excel(pos_xlsx, sheet_name="Channels_A")
    df_imp_A = pd.read_excel(pos_xlsx, sheet_name="Impacts_A")
    df_chn_B = pd.read_excel(pos_xlsx, sheet_name="Channels_B")
    df_imp_B = pd.read_excel(pos_xlsx, sheet_name="Impacts_B")
    df_vp    = pd.read_excel(pos_xlsx, sheet_name="VP_Channels")
    df_vpref = pd.read_excel(pos_xlsx, sheet_name="VP_RefChannels")

    MK_A = pyfbs.mck.Model.from_ansys(r"./lab_testbench/FEM/A.rst", r"./lab_testbench/FEM/A.full",
                                      no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)
    MK_B = pyfbs.mck.Model.from_ansys(r"./lab_testbench/FEM/B.rst", r"./lab_testbench/FEM/B.full",
                                      no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)

    df_chn_A = MK_A.update_locations_df(df_chn_A, scale=1)
    df_imp_A = MK_A.update_locations_df(df_imp_A, scale=1)
    df_chn_B = MK_B.update_locations_df(df_chn_B, scale=1)
    df_imp_B = MK_B.update_locations_df(df_imp_B, scale=1)

    MK_A.frf_synth(df_chn_A, df_imp_A, f_start=0, f_resolution=f_resolution, modal_damping=0.003)
    MK_B.frf_synth(df_chn_B, df_imp_B, f_start=0, f_resolution=f_resolution, modal_damping=0.003)

    freq  = MK_A.freq
    omega = 2 * np.pi * freq

    vpt_A = pyfbs.interface.VPT(df_chn_A, df_imp_A, df_vp, df_vpref)
    vpt_B = pyfbs.interface.VPT(df_chn_B, df_imp_B, df_vp, df_vpref)
    vpt_A.apply_vpt(MK_A.freq, MK_A.frf)
    vpt_B.apply_vpt(MK_B.freq, MK_B.frf)

    Y_A = vpt_A.frf   # (N_freq, nA, nA): A reference DoFs first, 6 VP interface DoFs last
    Y_B = vpt_B.frf   # (N_freq, nB, nB): 6 VP interface DoFs first, B reference DoFs after

    nA, nB = Y_A.shape[1], Y_B.shape[1]
    N = nA + nB
    Y = np.zeros((len(freq), N, N), dtype=complex)
    Y[:, :nA, :nA] = Y_A
    Y[:, nA:, nA:] = Y_B

    # signed-Boolean coupling of the 6 VP DoFs (A interface = last 6 of A,
    # B interface = first 6 of B) -- identical to the reference's Bc.
    A_if = np.arange(nA - N_IF, nA)
    B_if = np.arange(nA, nA + N_IF)
    Bc = np.zeros((N_IF, N))
    Bc[np.arange(N_IF), A_if] = 1.0
    Bc[np.arange(N_IF), B_if] = -1.0

    k_diag = np.array([k_trans, k_trans, k_trans, k_rot, k_rot, k_rot])
    c_diag = np.array([c_trans, c_trans, c_trans, c_rot, c_rot, c_rot])

    # response (out) and force (inp) physical DoFs in FULL block-diag indexing.
    # Layout: [A ref | A interface(6) | B interface(6) | B ref]; the reference plot's
    # reduced out=0 / inp=nA-6 map to full 0 and nA+6 once the 12 interface DoFs drop.
    out_full = 0
    inp_full = nA + N_IF

    return dict(freq=freq, omega=omega, Y=Y, Y_A=Y_A, Y_B=Y_B,
                nA=nA, nB=nB, N=N, Bc=Bc, k_diag=k_diag, c_diag=c_diag,
                out_full=out_full, inp_full=inp_full)


class TestbenchLinearSpring(FBS_System):
    """
    :param data:          output of :func:`build_testbench_data`
    :param F0:            harmonic force amplitude [N] (1.0 -> the response curve IS
                          the receptance |Y_coupled[out, inp]|).
    :param sample_number: AFT time samples (a linear force needs very few).
    """
    is_real_valued = True

    def __init__(self, data, F0=1.0, sample_number=256):
        self.B_coupling    = data["Bc"]                 # (6, N)
        self.K_spring      = np.diag(data["k_diag"])    # (6, 6)
        self.F0            = F0
        self.sample_number = sample_number
        self.out_full      = data["out_full"]
        self.inp_full      = data["inp_full"]

    def external_term(self, tau):
        N = self.B_coupling.shape[1]                    # total DoFs (no total_dimension needed)
        f = np.zeros((len(tau), N, 1))
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # --- linear bushing spring on the 6-DoF VP gap x_r = B u --------------------
    def interface_force(self, u_rel, udot_rel, tau):
        # K_spring (6,6) broadcasts over the leading time axis of u_rel (Nt,6,1)
        return self.K_spring @ u_rel                     # (Nt, 6, 1)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        n_int = self.B_coupling.shape[0]                 # 6 (interface dim, from B)
        return np.broadcast_to(self.K_spring, (len(tau), n_int, n_int))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        n_int = self.B_coupling.shape[0]
        return np.zeros((len(tau), n_int, n_int))
