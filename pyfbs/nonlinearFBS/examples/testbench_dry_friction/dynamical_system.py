import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System

N_IF = 6   # virtual-point interface DoFs: [ux, uy, uz, rx, ry, rz]


def build_testbench_data(mu_trans=0.3, k_trans=1.0e6, alpha=1.0e3,
                         G=0.01, k_rot=1.0e6,
                         f_resolution=1.0, modal_damping=0.003):
    """
    Build the pyFBS lab-testbench FBS data for a DRY-FRICTION joint: regularized
    Coulomb friction in the x-y plane and around z, penalty springs on the
    remaining gap DoFs.

    Identical data pipeline to the spring examples (ANSYS modes -> VPT ->
    block-diagonal admittance Y = diag(Y_A, Y_B)); only the joint law differs.
    At the joint B clamps A's plate: the contact faces are normal to z, so slip
    happens in the x-y plane (translation) and about z (spin). On the VP
    interface gap x_r = B u this gives, per DoF,

        f_T   = 2 mu N tanh(alpha ||v_T||) v_T/||v_T||   (ux, uy -- friction)
        f_z   = k_trans z_r                              (uz -- penalty spring)
        M_rxy = k_rot theta_r                            (rx, ry -- penalty springs)
        M_rz  = 2 mu N G tanh(alpha G thetadot_rz)       (rz -- torsional friction)

    with v_T the PHYSICAL relative velocity of the tangential gap DoFs and N
    the bolt clamping force (slip force 2*mu_trans*N: both clamp faces carry
    mu_trans*N; N is passed to :class:`TestbenchDryFriction`, independent of
    the excitation amplitude F0). Every DoF's joint term is decoupled from the
    others.

    :param mu_trans: friction coefficient [-]; the tangential force saturates
        at the slip force 2*mu_trans*N once the interface slides.
    :param k_trans: penalty stiffness [N/m] tying the normal (z) interface gap.
    :param alpha: tanh regularization sharpness [s/m]; near sticking the
        friction acts like a viscous damper c_eff = 2*mu_trans*N*alpha and it
        saturates to ideal Coulomb for ||v_T|| >> 1/alpha.
    :param G: geometry factor [m] of the torsional friction = effective radius
        of the clamped contact ring: G*thetadot_rz is the sliding speed fed to
        the same tanh regularization, and the spin moment saturates at the
        slip moment 2*mu_trans*N*G.
    :param k_rot: rotational penalty stiffness [Nm/rad] tying the tilt gap
        DoFs rx, ry (the clamped faces resist tilting; scale ~ k_trans*G^2).
    :param f_resolution: FRF frequency resolution Delta f [Hz] (smaller = denser).
    :param modal_damping: modal damping ratio used in the synthesis.
    :returns: dict with freq/omega/Y, DoF counts, the signed-Boolean coupling Bc,
        the joint parameters mu_trans/k_trans/alpha/G/k_rot, the out/inp DoFs,
        and the modal + VPT objects for the numerical ModalVPFRF pipeline.
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

    # response (out) and force (inp) physical DoFs in FULL block-diag indexing.
    # Layout: [A ref | A interface(6) | B interface(6) | B ref].
    out_full = 0
    inp_full = nA + N_IF

    return dict(freq=freq, omega=omega, Y=Y, Y_A=Y_A, Y_B=Y_B,
                nA=nA, nB=nB, N=N, Bc=Bc,
                mu_trans=mu_trans, k_trans=k_trans, alpha=alpha,
                G=G, k_rot=k_rot,
                out_full=out_full, inp_full=inp_full,
                # modal + VPT objects for the numerical ModalVPFRF pipeline
                MK_A=MK_A, MK_B=MK_B, vpt_A=vpt_A, vpt_B=vpt_B,
                df_chn_A=df_chn_A, df_imp_A=df_imp_A,
                df_chn_B=df_chn_B, df_imp_B=df_imp_B,
                modal_damping=modal_damping)


class TestbenchDryFriction(FBS_System):
    """
    Dry-friction joint on the 6-DoF VP interface gap x_r = B u, mirroring the
    physical joint (B clamps A's plate, contact normal along z, slip in the
    x-y plane and spin about z):

        f_T   = 2 mu N tanh(alpha ||v_T||) v_T/||v_T||,  v_T = [xdot_r, ydot_r]
        f_z   = k_trans z_r
        M_rxy = k_rot theta_r                            (tilt penalty, rx/ry)
        M_rz  = 2 mu N G tanh(alpha G thetadot_rz)       (torsional friction)

    Every DoF's term is decoupled from the others: the slip force 2*mu_trans*N
    and slip moment 2*mu_trans*N*G are constant (no dependence on the spring
    forces) and the springs carry no velocity term. G is the effective contact
    radius, so G*thetadot_rz is the sliding speed of the spin -- it enters the
    SAME tanh regularization as the translational slip. Same testbench, FBS
    structure and AFT + arc-length HBM solver as the spring examples. Near
    sticking the friction terms act like viscous couplers (c_eff =
    2*mu_trans*N*alpha resp. 2*mu_trans*N*alpha*G^2 for the spin); once the
    transmitted force/moment reaches the slip value the interface slides and
    it saturates. The friction terms are odd in the velocities, so only odd
    harmonics are generated.

    :param data:          output of :func:`build_testbench_data`.
    :param F0:            harmonic excitation amplitude [N].
    :param N:             bolt clamping force [N], independent of F0; each
                          clamp face carries mu_trans*N, so the slip force is
                          2*mu_trans*N.
    :param sample_number: AFT time samples (must resolve the highest harmonic).
    """
    is_real_valued = True

    def __init__(self, data, F0=1.0, N=200.0, sample_number=1024):
        self.B_coupling    = data["Bc"]                  # (6, n_dof)
        self.mu_trans      = data["mu_trans"]            # [-]
        self.k_trans       = data["k_trans"]             # [N/m]
        self.alpha         = data["alpha"]               # [s/m]
        self.G             = data["G"]                   # [m]
        self.k_rot         = data["k_rot"]               # [Nm/rad]
        self.F0            = F0
        self.N             = N
        self.sample_number = sample_number
        self.out_full      = data["out_full"]
        self.inp_full      = data["inp_full"]

    def external_term(self, tau):
        n_dof = self.B_coupling.shape[1]                 # total DoFs
        f = np.zeros((len(tau), n_dof, 1))
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # --- Coulomb friction (x-y, spin about z) + penalty springs (z, rx, ry) -----
    def _tang(self, vT):
        """s = tanh(alpha*g)/g (limit alpha at g = 0) and g = ||v_T||."""
        g = np.sqrt((vT ** 2).sum(axis=1))
        s = np.where(g > 1e-12, np.tanh(self.alpha * g) / np.maximum(g, 1e-30), self.alpha)
        return s, g

    def interface_force(self, u_rel, udot_rel, tau):
        # f_T = 2*mu*N*tanh(alpha*||v_T||)*v_T/||v_T|| on the x-y gap velocity,
        # f_z = k_trans * z gap, M_rx/ry = k_rot * tilt gap, and the torsional
        # Coulomb moment M_rz = 2*mu*N*G*tanh(alpha*G*rzdot): G turns the spin
        # rate into the sliding speed at the contact radius and the friction
        # force back into a moment. udot_rel is the PHYSICAL relative velocity
        # (omega * dx/dtau).
        vT = udot_rel[:, :2, 0]                          # (Nt, 2)
        s, _ = self._tang(vT)
        f = np.zeros((len(tau), N_IF, 1))
        f[:, :2, 0] = (2.0 * self.mu_trans * self.N * s)[:, None] * vT
        f[:, 2, 0]  = self.k_trans * u_rel[:, 2, 0]
        f[:, 3, 0]  = self.k_rot * u_rel[:, 3, 0]
        f[:, 4, 0]  = self.k_rot * u_rel[:, 4, 0]
        f[:, 5, 0]  = (2.0 * self.mu_trans * self.N * self.G
                       * np.tanh(self.alpha * self.G * udot_rel[:, 5, 0]))
        return f

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        # only the springs depend on displacement: df_z/dz = k_trans and
        # dM_rx/drx = dM_ry/dry = k_rot. The slip force/moment amplitudes are
        # constant, so the friction terms contribute nothing here.
        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, 2, 2] = self.k_trans
        J[:, 3, 3] = self.k_rot
        J[:, 4, 4] = self.k_rot
        return J

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        # df_T/dv_T = 2 mu N [ alpha(1-t^2) vhat vhat^T + (t/g)(I - vhat vhat^T) ]:
        # tanh slope along the sliding direction, direction-turning term across
        # it; g->0 limit = 2 mu N alpha I (viscous stick limit). The torsional
        # friction is 1-D: dM_rz/drzdot = 2 mu N alpha G^2 (1 - t_rz^2). The
        # springs have no velocity term.
        vT = udot_rel[:, :2, 0]                          # (Nt, 2)
        g  = np.sqrt((vT ** 2).sum(axis=1))
        gs   = np.maximum(g, 1e-30)
        t    = np.tanh(self.alpha * g)
        tog  = np.where(g > 1e-12, t / gs, self.alpha)                       # t/g -> alpha
        coef = np.where(g > 1e-12, self.alpha * (1 - t ** 2), self.alpha)    # -> alpha
        vhat = vT / gs[:, None]                                              # 0 at g = 0
        vv = vhat[:, :, None] * vhat[:, None, :]
        I2 = np.eye(2)[None]
        JTT = (2.0 * self.mu_trans * self.N) * (coef[:, None, None] * vv
                                                + tog[:, None, None] * (I2 - vv))
        J = np.zeros((len(tau), N_IF, N_IF))
        J[:, :2, :2] = JTT
        t_rz = np.tanh(self.alpha * self.G * udot_rel[:, 5, 0])
        J[:, 5, 5] = (2.0 * self.mu_trans * self.N * self.alpha * self.G ** 2
                      * (1.0 - t_rz ** 2))
        return J
