import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System

N_IF = 6   # virtual-point interface DoFs: [ux, uy, uz, rx, ry, rz]


def build_testbench_data(k_trans=1.0e6, k_rot=1.0e3,
                         alpha_trans=1.0e6, alpha_rot=0.0,
                         beta_trans=0.0, beta_rot=0.0,
                         f_resolution=1.0, modal_damping=0.003):
    """
    Build the pyFBS lab-testbench FBS data for a CUBIC interface spring with
    optional CUBIC (nonlinear) damping.

    Identical data pipeline to the linear-spring example (ANSYS modes -> VPT ->
    block-diagonal admittance Y = diag(Y_A, Y_B)); the change is that each
    interface DoF carries a cubic hardening coefficient ``alpha`` in addition to
    its linear stiffness ``k``, plus a cubic damping coefficient ``beta`` acting
    on the relative velocity, giving the bushing law

        f = k (x + alpha x^3) + beta * xdot^3.

    :param k_trans/k_rot:     linear stiffness [N/m] / [Nm/rad] per interface DoF.
    :param alpha_trans/alpha_rot: cubic stiffness coefficient [1/m^2] / [1/rad^2];
        the cubic force k*alpha*x^3 equals the linear force k*x at |x| = 1/sqrt(alpha),
        so alpha sets the gap amplitude at which the nonlinearity becomes ~100%.
    :param beta_trans/beta_rot: cubic damping coefficient [N s^3/m^3] /
        [Nm s^3/rad^3]; the damping force beta*xdot^3 grows with the cube of the
        relative interface velocity (0 -> no nonlinear damping).
    :param f_resolution: FRF frequency resolution Delta f [Hz] (smaller = denser).
    :param modal_damping: modal damping ratio used in the synthesis.
    :returns: dict with freq/omega/Y, DoF counts, the signed-Boolean coupling Bc,
        the per-DoF k_diag/alpha_diag/beta_diag, the out/inp DoFs, and the modal +
        VPT objects for the numerical ModalVPFRF pipeline (see the linear example).
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

    MK_A.frf_synth(df_chn_A, df_imp_A, f_start=0, f_end=8*1000, f_resolution=f_resolution, modal_damping=modal_damping)
    MK_B.frf_synth(df_chn_B, df_imp_B, f_start=0, f_end=8*1000, f_resolution=f_resolution, modal_damping=modal_damping)

    freq  = MK_A.freq
    omega = 2 * np.pi * freq

    vpt_A = pyfbs.interface.VPT(df_chn_A, df_imp_A, df_vp, df_vpref)
    vpt_B = pyfbs.interface.VPT(df_chn_B, df_imp_B, df_vp, df_vpref)
    vpt_A.apply_vpt(MK_A.freq, MK_A.frf)
    vpt_B.apply_vpt(MK_B.freq, MK_B.frf)

    Y_A = vpt_A.frf   # (N_freq, nA, nA): A reference DoFs first, 6 VP interface DoFs last
    Y_B = vpt_B.frf   # (N_freq, nB, nB): 6 VP interface DoFs first, B reference DoFs after

    nA, nB = Y_A.shape[1], Y_B.shape[1]
    print(nA, nB)
    N = nA + nB
    Y = np.zeros((len(freq), N, N), dtype=complex)
    Y[:, :nA, :nA] = Y_A
    Y[:, nA:, nA:] = Y_B

    # signed-Boolean coupling of the 6 VP DoFs (A interface = last 6 of A,
    # B interface = first 6 of B).
    A_if = np.arange(nA - N_IF, nA)
    B_if = np.arange(nA, nA + N_IF)
    Bc = np.zeros((N_IF, N))
    Bc[np.arange(N_IF), A_if] = 1.0
    Bc[np.arange(N_IF), B_if] = -1.0

    k_diag     = np.array([k_trans, k_trans, k_trans, k_rot, k_rot, k_rot])
    alpha_diag = np.array([alpha_trans, alpha_trans, alpha_trans, alpha_rot, alpha_rot, alpha_rot])
    beta_diag  = np.array([beta_trans, beta_trans, beta_trans, beta_rot, beta_rot, beta_rot])

    # response (out) and force (inp) physical DoFs in FULL block-diag indexing.
    # Layout: [A ref | A interface(6) | B interface(6) | B ref].
    out_full = 0
    inp_full = nA + N_IF

    return dict(freq=freq, omega=omega, Y=Y, Y_A=Y_A, Y_B=Y_B,
                nA=nA, nB=nB, N=N, Bc=Bc, k_diag=k_diag, alpha_diag=alpha_diag,
                beta_diag=beta_diag,
                out_full=out_full, inp_full=inp_full,
                # modal + VPT objects for the numerical ModalVPFRF pipeline
                MK_A=MK_A, MK_B=MK_B, vpt_A=vpt_A, vpt_B=vpt_B,
                df_chn_A=df_chn_A, df_imp_A=df_imp_A,
                df_chn_B=df_chn_B, df_imp_B=df_imp_B,
                modal_damping=modal_damping)


class TestbenchCubicSpring(FBS_System):
    """
    Cubic (hardening) bushing with cubic (nonlinear) damping on the 6-DoF VP
    interface gap  x_r = B u  and its relative velocity  xdot_r = B udot:

        f_nl = K_spring (x_r + alpha * x_r^3) + beta * xdot_r^3   (elementwise per DoF)

    Same testbench, same FBS structure and the same AFT + arc-length HBM solver as
    the linear-spring example; the differences are the per-DoF cubic stiffness term
    ``alpha`` (bends the resonance) and the per-DoF cubic damping term ``beta``
    (limits the peak amplitude); both couple the harmonics, so the result is a true
    nonlinear forced-response curve, not a receptance.

    :param data:          output of :func:`build_testbench_data`.
    :param F0:            harmonic force amplitude [N] (drives the gap amplitude, and
                          hence how strongly the cubic terms act -- tune with alpha/beta).
    :param sample_number: AFT time samples (must resolve the highest harmonic).
    """
    is_real_valued = True

    def __init__(self, data, F0=1.0, sample_number=256):
        self.B_coupling    = data["Bc"]                  # (6, N)
        self.k_diag        = data["k_diag"]              # (6,)
        self.alpha_diag    = data["alpha_diag"]          # (6,)
        self.beta_diag     = data["beta_diag"]           # (6,)
        self.F0            = F0
        self.sample_number = sample_number
        self.out_full      = data["out_full"]
        self.inp_full      = data["inp_full"]

    def external_term(self, tau):
        N = self.B_coupling.shape[1]                     # total DoFs
        f = np.zeros((len(tau), N, 1))
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # --- cubic bushing (spring + damper) on the 6-DoF VP gap x_r = B u ----------
    def interface_force(self, u_rel, udot_rel, tau):
        # f_i = k_i (x_i + alpha_i x_i^3) + beta_i xdot_i^3; the per-DoF coefficients
        # (6,) broadcast over the leading time axis of u_rel/udot_rel (Nt, 6, 1).
        # udot_rel is the PHYSICAL relative velocity (omega * dx/dtau).
        k     = self.k_diag[None, :, None]
        alpha = self.alpha_diag[None, :, None]
        beta  = self.beta_diag[None, :, None]
        return k * u_rel + alpha * u_rel ** 3 + beta * udot_rel ** 3   # (Nt, 6, 1)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        # df_i/dx_i = k_i (1 + 3 alpha_i x_i^2); the damping term has no displacement
        # dependence. Each DoF depends only on its own gap -> diagonal (Nt, 6, 6).
        n_int = self.B_coupling.shape[0]                 # 6
        diag  = self.k_diag[None, :] + 3.0 * self.alpha_diag[None, :] * u_rel[:, :, 0] ** 2
        J = np.zeros((len(tau), n_int, n_int))
        d = np.arange(n_int)
        J[:, d, d] = diag
        return J

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        # df_i/dxdot_i = 3 beta_i xdot_i^2; each DoF depends only on its own relative
        # velocity -> diagonal (Nt, 6, 6). Zero when all beta = 0 (pure spring).
        n_int = self.B_coupling.shape[0]                 # 6
        diag  = 3.0 * self.beta_diag[None, :] * udot_rel[:, :, 0] ** 2
        J = np.zeros((len(tau), n_int, n_int))
        d = np.arange(n_int)
        J[:, d, d] = diag
        return J
