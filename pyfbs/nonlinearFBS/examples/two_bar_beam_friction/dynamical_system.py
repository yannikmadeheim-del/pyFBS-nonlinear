"""
Two bar+beam elements connected by normal contact and dry friction (sketch: system.png).

Two identical cantilever bar+beam elements (Martins, case study 6.4), each one
substructure with three tip DoFs:

    element 1 tip:  q1 (axial), q2 (transverse), q3 (slope)
    element 2 tip:  q4 (axial), q5 (transverse), q6 (slope)

They are uncoupled in the linear system and coupled only at the tips:

    x_N = q5 - q2      normal, contact when x_N > eps  (gap function h = eps - x_N)
    x_T = q1 - q4      tangential, dry friction

Normal contact is rigid (DLFT); friction is f_T = mu f_N tanh(alpha1 d/dt x_T).
Damping C = beta K. Forcing f_ext = [P1 cos, -P2 sin - P5, 0, 0, P5, 0]^T with the
static clamping force P5.
"""

from dataclasses import dataclass

import numpy as np
from numpy import zeros, sin, cos, tanh
from scipy.linalg import eigh

from pyfbs.nonlinearFBS import FBS_System

Q1, Q2, Q3, Q4, Q5, Q6 = range(6)

# interface ordering [normal ; tangential]
N_IF = 2
L_N = np.array([[1.0, 0.0]])
L_F = np.array([[0.0, 1.0]])


@dataclass
class BarBeamParams:
    l:      float = 1.0       # element length
    EA:     float = 1.0 / 3   # axial rigidity
    EI:     float = 1.0 / 3   # bending rigidity
    lam:    float = 1.0       # mass per unit length
    beta:   float = 0.05      # stiffness-proportional damping C = beta K
    eps:    float = 0.01      # initial normal gap h0
    alpha1: float = 150.0     # friction regularization
    mu:     float = 0.1       # friction coefficient
    P1:     float = 0.02      # axial excitation amplitude (q1)
    P2:     float = 0.02      # transverse excitation amplitude (q2)
    P5:     float = 0.05      # static clamping force


def element_matrices(p):
    """Tip block of one cantilever, DoFs [axial, transverse, slope]."""
    l, EA, EI, lam = p.l, p.EA, p.EI, p.lam
    Ke = np.array([
        [EA / l,        0.0,             0.0          ],
        [0.0,      12.0 * EI / l**3, -6.0 * EI / l**2 ],
        [0.0,      -6.0 * EI / l**2,  4.0 * EI / l    ],
    ])
    Me = (lam * l / 210.0) * np.array([
        [70.0,   0.0,        0.0       ],
        [0.0,   78.0,      -11.0 * l   ],
        [0.0,  -11.0 * l,    2.0 * l**2],
    ])
    return Ke, Me


def assemble(p):
    """Block-diagonal M, C, K of both substructures."""
    Ke, Me = element_matrices(p)
    M = zeros((6, 6)); K = zeros((6, 6))
    M[:3, :3] = Me; M[3:, 3:] = Me
    K[:3, :3] = Ke; K[3:, 3:] = Ke
    return M, p.beta * K, K


def modal_model(p):
    """Mass-normalized modes (complete basis), eigenfrequencies and zeta_r = beta Omega_r / 2."""
    M, _, K = assemble(p)
    eigenvalues, Phi = eigh(K, M)
    Omega = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return Omega, Phi, 0.5 * p.beta * Omega


def coupling_matrix():
    """x_rel = B q, ordered [normal ; tangential]."""
    B = zeros((N_IF, 6))
    B[0, Q5] = +1.0; B[0, Q2] = -1.0            # x_N = q5 - q2
    B[1, Q1] = +1.0; B[1, Q4] = -1.0            # x_T = q1 - q4
    return B


class BarBeamDLFT(FBS_System):
    """Rigid DLFT normal contact + regularized friction.

    Use with DLFTContactAFT(L_N, L_F, epsilon, g_zero=eps).
    """
    is_real_valued = True

    def __init__(self, params=None, sample_number=256):
        self.p = params or BarBeamParams()
        self.sample_number = sample_number
        self.omega_ref = 1.0                         # first undamped resonance
        self.B_coupling = coupling_matrix()
        self.mass_matrix, self.damping_matrix, self.stiffness_matrix = assemble(self.p)
        self.angular_eig_freq, self.eig_vec, self.zeta = modal_model(self.p)

    def external_term(self, tau):
        f = zeros((len(tau), 6, 1))
        f[:, Q1, 0] = self.p.P1 * cos(tau)                    # axial -> sliding
        f[:, Q2, 0] = -self.p.P2 * sin(tau) - self.p.P5       # transverse + clamping
        f[:, Q5, 0] = self.p.P5                               # clamping
        return f

    # --- no additional law on the full interface ---
    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, N_IF))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, N_IF))

    # --- friction f_T = mu f_N tanh(alpha1 v_T), driven by the DLFT normal force ---
    def friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        p = self.p
        return p.mu * f_normal * tanh(p.alpha1 * udot_F_rel)

    def jacobian_friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        return zeros((len(tau), 1, 1))

    def jacobian_friction_force_qdot(self, u_F_rel, udot_F_rel, f_normal, tau):
        p = self.p
        th = tanh(p.alpha1 * udot_F_rel[:, :, 0])
        return (p.mu * f_normal[:, :, 0] * p.alpha1 * (1.0 - th * th))[:, :, None]

    def jacobian_friction_force_normal(self, u_F_rel, udot_F_rel, f_normal, tau):
        p = self.p
        return p.mu * tanh(p.alpha1 * udot_F_rel)
