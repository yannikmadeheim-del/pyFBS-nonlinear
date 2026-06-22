"""System definition for the rod vibro-impact example.

Clamped-free FE axial rod whose free end B impacts a FLEXIBLE obstacle (a grounded
linear spring k_obs), coupled to the rod through Frequency-Based Substructuring.
See ``main.py`` for the full model description (Vadcard, Batailly & Thouverez,
J. Sound Vib. 531, 2022, Fig. 17 + Table 1).
"""
import numpy as np
from numpy import zeros, cos
from scipy.linalg import eigh

from pyfbs.nonlinearFBS import FBS_System


class RodVibroImpactFlexible(FBS_System):
    """Clamped-free axial bar whose free end B impacts a FLEXIBLE wall.

    The wall is a grounded spring k_obs appended as one extra DOF (the obstacle
    node u_w).  The assembled system is block-diagonal -- rod and obstacle are
    linearly UNcoupled; they interact only through the unilateral DLFT contact on
    the relative interface DOF  x_r = u_B - u_w.  Hence:

        M = blkdiag(M_rod, 0),  C = blkdiag(C_rod, 0),  K = blkdiag(K_rod, k_obs)
        B_coupling = [ ... +1 (rod tip) ... -1 (obstacle node) ]   ->  x_r = u_B - u_w

    The obstacle stiffness is k_obs = k_rel * k_rod with k_rod = E*A/L.
    """
    is_real_valued = True

    def __init__(self, n_elem=20, L=0.13, E=210e9, rho=7800.0, A=15.6e-4,
                 F0=1.0e4, xi=7.5e-3, k_rel=1.0, poly_deg=33):
        l = L / n_elem
        n = n_elem                       # rod free DOF count (node 0 clamped)

        # --- assemble global rod M, K, then drop the clamped DOF (node 0) ---
        Me = (rho * A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
        Ke = (E * A / l)         * np.array([[1.0, -1.0], [-1.0, 1.0]])
        M_rod = zeros((n + 1, n + 1))
        K_rod = zeros((n + 1, n + 1))
        for e in range(n_elem):
            M_rod[e:e + 2, e:e + 2] += Me
            K_rod[e:e + 2, e:e + 2] += Ke
        M_rod = M_rod[1:, 1:]            # remove clamped node 0
        K_rod = K_rod[1:, 1:]

        # --- modal damping: C = M Phi diag(2 xi w_i) Phi^T M  (Phi mass-normalized) ---
        w2, Phi = eigh(K_rod, M_rod)     # Phi^T M Phi = I
        omega_modes = np.sqrt(np.clip(w2, 0.0, None))
        C_rod = M_rod @ Phi @ np.diag(2.0 * xi * omega_modes) @ Phi.T @ M_rod

        # --- flexible wall: grounded spring k_obs = k_rel * k_rod ---
        # The rigid wall is the limit k_rel -> infinity; a finite k_rel is a finite
        # obstacle stiffness.  k_rel must be > 0 (k_rel = 0 would mean no obstacle).
        if k_rel <= 0.0:
            raise ValueError("k_rel must be > 0 (k_obs = k_rel * k_rod).")
        k_rod = E * A / L                # rod static axial stiffness  [N/m]
        k_obs = k_rel * k_rod
        self.k_rod = k_rod
        self.k_ref = k_rod               # alias kept for back-compat
        self.k_obs = k_obs
        self.k_rel = k_rel

        # --- augment with the obstacle node (zero mass, zero damping, k_obs to ground) ---
        d_tot = n + 1                    # rod DOFs (n) + obstacle node (1)
        M = zeros((d_tot, d_tot)); M[:n, :n] = M_rod
        C = zeros((d_tot, d_tot)); C[:n, :n] = C_rod
        K = zeros((d_tot, d_tot)); K[:n, :n] = K_rod
        K[n, n] = k_obs                  # obstacle spring to ground

        # --- nondimensionalize frequency by omega_1 of the ROD: sweep w_hat = w/w_1.
        # With M' = w1^2 M, C' = w1 C, K' = K the FRF in w_hat equals the physical
        # FRF in w (the constant obstacle block k_obs is frequency-independent, so
        # it is unaffected), so the arc-length stepper sees an O(1) frequency axis. ---
        omega_ref = omega_modes[0]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * M
        self.damping_matrix   = omega_ref      * C
        self.stiffness_matrix = K

        # interface = relative DOF x_r = u_B - u_w  (rod tip minus obstacle node)
        self.rod_tip_idx  = n - 1
        self.obstacle_idx = n
        B = zeros((1, d_tot))
        B[0, self.rod_tip_idx]  = +1.0
        B[0, self.obstacle_idx] = -1.0
        self.B_coupling = B

        self.dimension        = self.B_coupling.shape[0]   # n_int: one contact DOF
        self.total_dimension  = d_tot
        self.polynomial_degree = poly_deg
        self.F0 = F0
        self.omega_modes = omega_modes

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)   # forcing at the free end B
        return f

    # DLFT supplies the contact force; AFT stubs kept for interface completeness.
    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), self.dimension, self.dimension))
