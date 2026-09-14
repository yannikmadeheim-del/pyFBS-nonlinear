"""
Composable joint models on the pyFBS lab testbench, for the nonlinearFBS solver.

Two parts:

  * :func:`build_testbench_data` -- the (joint-independent) FBS data pipeline:
    ANSYS modes -> VPT -> block-diagonal admittance Y = diag(Y_A, Y_B) plus the
    signed-Boolean coupling Bc of the six virtual-point interface DoFs. This is
    the expensive step and knows nothing about the joint.

  * A small library of JOINT ELEMENTS (linear / cubic / dry friction) that are
    each assigned to a named subset of the interface DoFs
    ("ux", "uy", "uz", "rx", "ry", "rz") and summed into one interface force by
    :class:`TestbenchJoint`. That is what makes joint models comparable: a cubic
    AND a linear spring on uz, friction on ux-uy and linear springs on the
    rotations are just four entries in one list.

Usage::

    data   = build_testbench_data(f_resolution=0.1, modal_damping=0.005)
    joints = make_joints([dict(type="linear", k=1e6, c=0.5, dofs=("ux", "uy", "uz")),
                          dict(type="cubic",  alpha=1e8, dofs=("uz",))])
    system = TestbenchJoint(data, joints, F0=200.0, sample_number=256)

The solver wiring (update_dependencies -> FBSProblem -> HarmonicBalanceMethod)
lives in main.py.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import pyfbs
from pyfbs.nonlinearFBS import FBS_System
from pyfbs.nonlinearFBS.examples.testbench_joint_comparison import measurements

HERE = Path(__file__).resolve().parent

N_IF = 6                                                # virtual-point interface DoFs
VP_DOFS = ("ux", "uy", "uz", "rx", "ry", "rz")          # their names, in row order
ALL_DOFS = VP_DOFS                                      # default assignment of an element

# The lab testbench data is ~280 MB. The constant points at the copy that already
# sits next to the cubic-spring example so it is not duplicated a fifth time;
# every FE/xlsx path below is built from it, never from the current directory.
DATA_DIR = HERE.parent / "testbench_cubicSpring" / "lab_testbench"


def _resolve_data_dir():
    """DATA_DIR if it is there, otherwise download the testbench into this folder.

    ``download_lab_testbench()`` always writes ``./lab_testbench`` relative to the
    current working directory, so the call is bracketed by a chdir.
    """
    if DATA_DIR.exists():
        return DATA_DIR
    cwd = os.getcwd()
    os.chdir(HERE)
    try:
        pyfbs.io.download_lab_testbench()
    finally:
        os.chdir(cwd)
    return HERE / "lab_testbench"


def build_testbench_data(f_resolution=1.0, modal_damping=0.003, f_end=20000.0,
                         limit_modes=None, no_modes=100,
                         workbook=measurements.DEFAULT, static_correction=True):
    """
    Build the joint-INDEPENDENT FBS data of the pyFBS lab testbench.

    ANSYS modes (A.rst/A.full, B.rst/B.full) -> synthesized channel FRFs -> VPT
    per substructure -> block-diagonal uncoupled admittance Y = diag(Y_A, Y_B),
    plus the signed-Boolean matrix Bc that forms the six-DoF virtual-point gap
    x_r = Bc u = VP_A - VP_B. The joint law is applied on x_r afterwards and is
    therefore not part of this function.

    :param f_resolution: FRF grid resolution Delta f [Hz]. Only the
        ExperimentalFRF provider interpolates on this grid; ModalVPFRF
        synthesizes at the exact n*omega and ignores it.
    :param modal_damping: modal damping ratio used in the synthesis.
    :param f_end: end frequency [Hz] of the synthesized grid. It must cover the
        highest harmonic the continuation will ask for (max(harmonics) * f_hi)
        when the experimental provider is used.
    :param limit_modes: number of free-interface modes per substructure kept in
        the mode superposition that builds Y (None = all computed ones). The
        ModalVPFRF provider does its own truncation and takes the same number in
        main.build_data_and_provider.
    :param no_modes: size of the eigensolve. Kept constant across a mode-count
        study: the .full.pkl cache is keyed on it, so a varying no_modes would
        re-solve the 20370-DoF eigenproblem for every run.
    :param workbook: measurement table to read, named by its file name without
        the extension. ``measurements.available()`` lists the choices.
    :param static_correction: add the residual-flexibility term so the truncated
        modal receptance is statically exact, which is what lets a low mode count
        match the pyhbm Craig-Bampton reference. It only reaches the solver
        through the ExperimentalFRF provider; ModalVPFRF synthesizes its own
        admittance and ignores it.
    :returns: dict with freq/omega/Y, the DoF counts nA/nB/N, the coupling Bc,
        the out/inp DoFs and the modal + VPT objects the ModalVPFRF pipeline needs.
    """
    root = _resolve_data_dir()
    # The FE data is shared with the cubic-spring example, the workbooks are
    # NOT: the copies in measurements/ have collocated interface rows --
    # Channels_<X> and Impacts_<X> both hold the union of the two sheets'
    # Grouping==10 rows, so the VPT sees the same interface DoF set as the pyhbm
    # RBE_rigid / RBE_average condensations it is compared against. The two
    # sheets list that set in a different row order (channels+impacts against
    # impacts+channels), so Tf equals Tu.T only up to that permutation. Both
    # pipelines apply the permutation consistently, which is what keeps the
    # coupled virtual-point admittance reciprocal.
    pos_xlsx = str(measurements.resolve(workbook))

    df_chn_A = pd.read_excel(pos_xlsx, sheet_name="Channels_A")
    df_imp_A = pd.read_excel(pos_xlsx, sheet_name="Impacts_A")
    df_chn_B = pd.read_excel(pos_xlsx, sheet_name="Channels_B")
    df_imp_B = pd.read_excel(pos_xlsx, sheet_name="Impacts_B")
    df_vp    = pd.read_excel(pos_xlsx, sheet_name="VP_Channels")
    df_vpref = pd.read_excel(pos_xlsx, sheet_name="VP_RefChannels")

    fem = root / "FEM"
    # pyFBS truncates a too-large limit_modes against the modes that EXIST and
    # says nothing, so asking for more than were computed would silently give a
    # different curve than the one the run is labelled with.
    n_modes = no_modes if limit_modes is None else max(no_modes, int(limit_modes))
    MK_A = pyfbs.mck.Model.from_ansys(str(fem / "A.rst"), str(fem / "A.full"),
                                      no_modes=n_modes, allow_pickle=True, recalculate=False, mesh_scale=1)
    MK_B = pyfbs.mck.Model.from_ansys(str(fem / "B.rst"), str(fem / "B.full"),
                                      no_modes=n_modes, allow_pickle=True, recalculate=False, mesh_scale=1)

    df_chn_A = MK_A.update_locations_df(df_chn_A, scale=1)
    df_imp_A = MK_A.update_locations_df(df_imp_A, scale=1)
    df_chn_B = MK_B.update_locations_df(df_chn_B, scale=1)
    df_imp_B = MK_B.update_locations_df(df_imp_B, scale=1)

    MK_A.frf_synth(df_chn_A, df_imp_A, f_start=0, f_end=f_end,
                   f_resolution=f_resolution, modal_damping=modal_damping,
                   limit_modes=limit_modes, static_correction=static_correction)
    MK_B.frf_synth(df_chn_B, df_imp_B, f_start=0, f_end=f_end,
                   f_resolution=f_resolution, modal_damping=modal_damping,
                   limit_modes=limit_modes, static_correction=static_correction)

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

    print(f"testbench data: nA={nA}, nB={nB}, {len(freq)} FRF lines up to "
          f"{f_end:g} Hz, {n_modes} modes computed, "
          f"{limit_modes if limit_modes is not None else n_modes} used in Y")
    return dict(freq=freq, omega=omega, Y=Y, Y_A=Y_A, Y_B=Y_B,
                nA=nA, nB=nB, N=N, Bc=Bc,
                out_full=out_full, inp_full=inp_full,
                # modal + VPT objects for the numerical ModalVPFRF pipeline
                MK_A=MK_A, MK_B=MK_B, vpt_A=vpt_A, vpt_B=vpt_B,
                df_chn_A=df_chn_A, df_imp_A=df_imp_A,
                df_chn_B=df_chn_B, df_imp_B=df_imp_B,
                modal_damping=modal_damping)


# ---------------------------------------------------------------------------
# Joint elements
#
# Every element acts on a named subset of the six interface DoFs and returns a
# CONTRIBUTION of the shapes the FBS_System interface expects; TestbenchJoint
# sums them. Rows outside the element's DoFs stay zero, so elements never
# overwrite each other and any number of them may share a DoF.
#
# All coefficients are SCALARS. Different values on different DoFs are expressed
# by a second entry of the same type rather than by per-DoF vectors or
# *_trans/*_rot shorthands: study.py can then read "a list value means sweep
# this parameter" without having to guess whether a list is a sweep or a
# per-DoF vector.
# ---------------------------------------------------------------------------

def resolve_dofs(dofs):
    """Interface-DoF indices for a sequence of names out of :data:`VP_DOFS`."""
    idx = []
    for name in dofs:
        if name not in VP_DOFS:
            raise ValueError(f"unknown interface DoF {name!r}; valid names are "
                             f"{', '.join(VP_DOFS)}")
        idx.append(VP_DOFS.index(name))
    return np.array(idx, dtype=int)


class JointElement:
    """
    Base class: one joint contribution on the DoFs named in ``dofs``.

    Subclasses implement, all on the physical interface gap ``u_rel`` (Nt, 6, 1)
    and its physical relative velocity ``udot_rel`` (Nt, 6, 1):

        force(u_rel, udot_rel)    -> (Nt, 6, 1)
        jac_u(u_rel, udot_rel)    -> (Nt, 6, 6)   df/du_rel
        jac_udot(u_rel, udot_rel) -> (Nt, 6, 6)   df/dudot_rel
    """

    def __init__(self, dofs):
        self.dofs = tuple(dofs)
        self.idx = resolve_dofs(self.dofs)

    def constrained_dofs(self):
        """DoF indices this element actually ties (stiffness or friction).
        Used only for the ill-conditioning warning in :class:`TestbenchJoint`."""
        return ()

    def _zeros_f(self, u_rel):
        return np.zeros((u_rel.shape[0], N_IF, 1))

    def _zeros_J(self, u_rel):
        return np.zeros((u_rel.shape[0], N_IF, N_IF))


class LinearJoint(JointElement):
    """
    Linear spring + viscous damper, decoupled per DoF:  f = k*u + c*v.

    :param k: stiffness [N/m] resp. [Nm/rad].
    :param c: viscous damping [N s/m] resp. [Nm s/rad]. The joint
        (rigid-body-vs-spring) modes carry no substructure modal damping, so
        only this damper limits their resonances.
    """

    def __init__(self, k, c=0.0, dofs=ALL_DOFS):
        super().__init__(dofs)
        self.k = float(k)
        self.c = float(c)

    def constrained_dofs(self):
        return tuple(self.idx) if self.k != 0.0 else ()

    def force(self, u_rel, udot_rel):
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (self.k * u_rel[:, self.idx, 0]
                             + self.c * udot_rel[:, self.idx, 0])
        return f

    def jac_u(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = self.k
        return J

    def jac_udot(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = self.c
        return J


class CubicJoint(JointElement):
    """
    Cubic hardening spring with optional cubic damping:  f = alpha*u^3 + beta*v^3.

    :param alpha: cubic stiffness [N/m^3] resp. [Nm/rad^3]. Combined with a
        linear spring k on the same DoF, the cubic force equals the linear one
        at |u| = sqrt(k/alpha), so alpha sets the gap amplitude at which the
        nonlinearity takes over.
    :param beta: cubic damping [N s^3/m^3] resp. [Nm s^3/rad^3]; grows with the
        cube of the relative velocity and clips the hardening peak.
    """

    def __init__(self, alpha, beta=0.0, dofs=ALL_DOFS):
        super().__init__(dofs)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def constrained_dofs(self):
        return tuple(self.idx) if self.alpha != 0.0 else ()

    def force(self, u_rel, udot_rel):
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (self.alpha * u_rel[:, self.idx, 0] ** 3
                             + self.beta * udot_rel[:, self.idx, 0] ** 3)
        return f

    def jac_u(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = 3.0 * self.alpha * u_rel[:, self.idx, 0] ** 2
        return J

    def jac_udot(self, u_rel, udot_rel):
        J = self._zeros_J(u_rel)
        J[:, self.idx, self.idx] = 3.0 * self.beta * udot_rel[:, self.idx, 0] ** 2
        return J


class FrictionJoint(JointElement):
    """
    tanh-regularized isotropic Coulomb friction on a TANGENTIAL PAIR of DoFs:

        f_T = 2 mu N tanh(alpha_reg ||v_T||) v_T / ||v_T||

    with v_T the PHYSICAL relative velocity of the two DoFs in ``dofs``. Both
    clamp faces carry mu*N, hence the slip force 2*mu*N. Near sticking the law
    acts like a viscous coupler (c_eff = 2*mu*N*alpha_reg); once the transmitted
    force reaches the slip value the interface slides and the force saturates.
    Being odd in the velocity, it generates only odd harmonics. The force
    depends on the velocity alone, so ``jac_u`` is zero -- the normal and
    rotational springs of a real bolted joint are separate ``linear`` entries on
    their own DoFs, not part of this element.

    :param mu: friction coefficient [-].
    :param N: clamping force [N] (e.g. bolt preload), independent of the
        excitation amplitude F0.
    :param alpha_reg: tanh regularization sharpness [s/m]; ideal Coulomb is
        approached for ||v_T|| >> 1/alpha_reg.
    :param dofs: the tangential pair the isotropic law couples (exactly two).
    :param spin_dof: rotational DoF carrying the torsional friction
        M = 2 mu N G tanh(alpha_reg G rdot). Optional, but it and ``G`` are two
        halves of one term: give both to enable it, or neither for pure
        translational friction.
    :param G: effective contact radius [m] of the torsional friction: G*rdot is
        the sliding speed fed to the same regularization and the slip moment is
        2*mu*N*G.
    """

    def __init__(self, mu, N, alpha_reg, dofs=("ux", "uy"), spin_dof=None, G=0.0):
        super().__init__(dofs)
        assert len(self.idx) == 2, (
            f"FrictionJoint acts on a tangential PAIR of DoFs, got {self.dofs}")
        self.mu = float(mu)
        self.N = float(N)
        self.alpha_reg = float(alpha_reg)
        self.G = float(G)
        self.spin_dof = spin_dof
        # Half a torsional specification would silently do nothing, so it is
        # rejected instead: either both halves or neither.
        if (spin_dof is None) != (self.G == 0.0):
            raise ValueError(
                f"FrictionJoint: the torsional term needs spin_dof AND a "
                f"non-zero G, got spin_dof={spin_dof!r}, G={self.G!r}. Pass "
                f"both to enable it, or neither for pure translational friction.")
        self.spin = None if spin_dof is None else int(resolve_dofs((spin_dof,))[0])

    @property
    def _spin_active(self):
        # the constructor rejects half a specification, so spin_dof alone decides
        return self.spin is not None

    def constrained_dofs(self):
        return tuple(self.idx) + ((self.spin,) if self._spin_active else ())

    def force(self, u_rel, udot_rel):
        vT = udot_rel[:, self.idx, 0]                                # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))
        # s = tanh(alpha_reg*g)/g, continued by its limit alpha_reg at g = 0
        s = np.where(g > 1e-12,
                     np.tanh(self.alpha_reg * g) / np.maximum(g, 1e-30),
                     self.alpha_reg)
        f = self._zeros_f(u_rel)
        f[:, self.idx, 0] = (2.0 * self.mu * self.N * s)[:, None] * vT
        if self._spin_active:
            f[:, self.spin, 0] = (2.0 * self.mu * self.N * self.G
                                  * np.tanh(self.alpha_reg * self.G
                                            * udot_rel[:, self.spin, 0]))
        return f

    def jac_u(self, u_rel, udot_rel):
        # the slip force amplitude is constant -- no displacement dependence.
        return self._zeros_J(u_rel)

    def jac_udot(self, u_rel, udot_rel):
        # df_T/dv_T = 2 mu N [ alpha(1-t^2) vhat vhat^T + (t/g)(I - vhat vhat^T) ]:
        # tanh slope along the sliding direction, direction-turning term across
        # it; both terms tend to alpha_reg as g -> 0 (the viscous stick limit).
        vT = udot_rel[:, self.idx, 0]                                # (Nt, 2)
        g = np.sqrt((vT ** 2).sum(axis=1))
        gs = np.maximum(g, 1e-30)
        t = np.tanh(self.alpha_reg * g)
        tog = np.where(g > 1e-12, t / gs, self.alpha_reg)
        coef = np.where(g > 1e-12, self.alpha_reg * (1 - t ** 2), self.alpha_reg)
        vhat = vT / gs[:, None]                                      # 0 at g = 0
        vv = vhat[:, :, None] * vhat[:, None, :]
        I2 = np.eye(2)[None]
        JTT = (2.0 * self.mu * self.N) * (coef[:, None, None] * vv
                                          + tog[:, None, None] * (I2 - vv))
        J = self._zeros_J(u_rel)
        J[:, self.idx[:, None], self.idx[None, :]] = JTT
        if self._spin_active:
            t_s = np.tanh(self.alpha_reg * self.G * udot_rel[:, self.spin, 0])
            J[:, self.spin, self.spin] = (2.0 * self.mu * self.N * self.alpha_reg
                                          * self.G ** 2 * (1.0 - t_s ** 2))
        return J


JOINT_TYPES = {"linear": LinearJoint, "cubic": CubicJoint, "friction": FrictionJoint}


def make_joints(specs):
    """Build the element list from a list of dicts, each with a ``"type"`` key
    naming an entry of :data:`JOINT_TYPES`; the remaining keys are the element's
    constructor arguments."""
    joints = []
    for spec in specs:
        spec = dict(spec)
        kind = spec.pop("type", None)
        if kind not in JOINT_TYPES:
            raise ValueError(f"unknown joint type {kind!r}; valid types are "
                             f"{', '.join(JOINT_TYPES)}")
        joints.append(JOINT_TYPES[kind](**spec))
    return joints


class TestbenchJoint(FBS_System):
    """
    Testbench A + B coupled by a joint assembled from :class:`JointElement`
    contributions on the six-DoF virtual-point gap x_r = Bc u:

        f_joint(x_r, xdot_r) = sum_e f_e(x_r, xdot_r)

    Each element writes only its own DoF rows, so the elements are independent
    and freely combinable. The FBS structure and the AFT + arc-length HBM solver
    are the same as in the single-law examples.

    :param data:          output of :func:`build_testbench_data`.
    :param joints:        list of :class:`JointElement` (see :func:`make_joints`).
    :param F0:            harmonic excitation amplitude [N] at the input DoF.
    :param sample_number: AFT time samples (must resolve the highest harmonic).
    """
    is_real_valued = True

    def __init__(self, data, joints, F0=1.0, sample_number=256):
        self.B_coupling    = data["Bc"]                  # (6, N)
        self.joints        = list(joints)
        self.F0            = F0
        self.sample_number = sample_number
        self.out_full      = data["out_full"]
        self.inp_full      = data["inp_full"]

        # A DoF with neither stiffness nor friction leaves A and B unconnected
        # there: the coupled residual has (almost) no restoring term on that gap
        # component and the Newton system becomes ill-conditioned.
        held = set()
        for element in self.joints:
            held.update(int(i) for i in element.constrained_dofs())
        loose = [VP_DOFS[i] for i in range(N_IF) if i not in held]
        if loose:
            print(f"warning: interface DoF(s) {', '.join(loose)} carry neither "
                  f"stiffness nor friction -- the coupled problem is "
                  f"unconstrained there and likely ill-conditioned")

    def external_term(self, tau):
        n_dof = self.B_coupling.shape[1]                 # total DoFs
        f = np.zeros((len(tau), n_dof, 1))
        f[:, self.inp_full, 0] = self.F0 * np.cos(tau)
        return f

    # --- assembled joint on the 6-DoF VP gap; udot_rel is the PHYSICAL relative
    #     velocity (the framework has already multiplied by omega) --------------
    def interface_force(self, u_rel, udot_rel, tau):
        f = np.zeros((len(tau), N_IF, 1))
        for element in self.joints:
            f += element.force(u_rel, udot_rel)
        return f

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        J = np.zeros((len(tau), N_IF, N_IF))
        for element in self.joints:
            J += element.jac_u(u_rel, udot_rel)
        return J

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        J = np.zeros((len(tau), N_IF, N_IF))
        for element in self.joints:
            J += element.jac_udot(u_rel, udot_rel)
        return J
