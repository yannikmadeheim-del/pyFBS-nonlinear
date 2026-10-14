"""
Two rods under unilateral contact (sketch: system.png).

Two identical clamped-free axial FE rods (Vadcard, Table 1) face each other across an
initial gap g0. Rod A is driven at its tip by F0 cos(omega t); rod B is passive. Both tip
displacements are positive toward the contact (the sketch shows q^II positive to the
right, i.e. u_B = -q^II), so the relative approach is

    x_r = u_A + u_B,        B = [+1 (tip A), +1 (tip B)],

and the rigid contact constraint reads x_r <= g0. Damping: modal ratio xi for every mode.
Units: m, N, s.
"""

from dataclasses import dataclass

import numpy as np
from numpy import zeros, cos
from scipy.linalg import eigh

from pyfbs.nonlinearFBS import FBS_System

# DLFTContactAFT localisation: x_r is the normal DoF, no friction DoFs
L_N = np.ones((1, 1))
L_F = np.zeros((0, 1))


@dataclass
class RodParams:
    n_elem: int = 20           # bar elements per rod
    L:      float = 0.13       # rod length [m]
    E:      float = 210e9      # Young's modulus [Pa]
    rho:    float = 7800.0     # density [kg/m^3]
    A:      float = 15.6e-4    # cross section [m^2]
    xi:     float = 7.5e-3     # modal damping ratio
    F0:     float = 25e3       # forcing amplitude at tip A [N]
    sample_number: int = 1000  # time samples per period

    @property
    def k_rod(self):
        """Static axial stiffness E A / L [N/m]."""
        return self.E * self.A / self.L


def assemble_rod(p):
    """M, K of one clamped-free rod (clamped node removed); the tip is the last DoF."""
    n, l = p.n_elem, p.L / p.n_elem
    Me = (p.rho * p.A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Ke = (p.E * p.A / l) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    M = zeros((n + 1, n + 1)); K = zeros((n + 1, n + 1))
    for e in range(n):
        M[e:e + 2, e:e + 2] += Me
        K[e:e + 2, e:e + 2] += Ke
    return M[1:, 1:], K[1:, 1:]


class TwoRodVibroImpact(FBS_System):
    """Two identical rods, rigid unilateral contact on x_r (DLFT part of DLFTContactAFT)."""
    is_real_valued = True

    def __init__(self, params=None):
        p = params or RodParams()
        M_rod, K_rod = assemble_rod(p)
        n = p.n_elem
        M = zeros((2 * n, 2 * n)); M[:n, :n] = M_rod; M[n:, n:] = M_rod
        K = zeros((2 * n, 2 * n)); K[:n, :n] = K_rod; K[n:, n:] = K_rod

        # modal data of both rods for NumericalFRF.from_modal (Phi^T M Phi = I)
        w2, Phi = eigh(K, M)
        self.angular_eig_freq = np.sqrt(np.clip(w2, 0.0, None))
        self.eig_vec = Phi
        self.zeta = p.xi
        self.omega_ref = self.angular_eig_freq[0]        # first axial mode omega_1

        self.total_dimension = 2 * n
        self.tipA_idx, self.tipB_idx = n - 1, 2 * n - 1
        self.B_coupling = zeros((1, 2 * n))
        self.B_coupling[0, [self.tipA_idx, self.tipB_idx]] = 1.0

        self.k_rod = p.k_rod
        self.F0 = p.F0
        self.sample_number = p.sample_number

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.tipA_idx, 0] = self.F0 * cos(tau)
        return f

    # --- no additional law on the interface: DLFT supplies the contact force ---
    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), 1, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), 1, 1))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), 1, 1))

    # --- no friction DoFs (n_F = 0) ---
    def friction_force(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return zeros((len(tau), 0, 1))

    def jacobian_friction_force(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return zeros((len(tau), 0, 0))

    def jacobian_friction_force_qdot(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return zeros((len(tau), 0, 0))

    def jacobian_friction_force_normal(self, u_F_rel, u_F_rel_dot, f_normal, tau):
        return zeros((len(tau), 0, 1))
