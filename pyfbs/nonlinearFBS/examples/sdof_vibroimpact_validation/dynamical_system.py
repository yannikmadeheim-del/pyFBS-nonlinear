"""System definition for the SDOF vibro-impact validation example.

Single-DOF mass-spring-damper against a "flexible" wall, written in FBS form so the
unilateral contact is enforced by DLFTContact on the interface DOF.  See main.py.
"""
import numpy as np

from pyfbs.nonlinearFBS import FBS_System


class SDOFVibroImpact(FBS_System):
    """
    SDOF mass-spring-damper-wall in FBS form.

        m q'' + c q' + k q = F0 cos(tau)
        subject to  q <= g0  (rigid wall, enforced by DLFTContact)
    """
    is_real_valued = True

    def __init__(self, m=1.0, c=0.01, k=1.0, F0=0.02, poly_deg=100, k_rel=100):
        self.mass_matrix       = np.array([[m, 0],[0, 0]])
        self.damping_matrix    = np.array([[c, 0],[0, 0]])
        self.stiffness_matrix  = np.array([[k, 0],[0, k_rel*k]])
        self.B_coupling        = np.array([[1.0, -1.0]])      # 1 interface DOF
        self.total_dimension   = 2
        self.dimension         = 1                       # n_int
        self.polynomial_degree = poly_deg
        self.F0 = F0

    def external_term(self, tau):
        f = np.zeros((len(tau), self.total_dimension, 1))
        f[:, 0, 0] = self.F0 * np.cos(tau)
        return f

    # DLFT computes contact internally; stubs kept so AFT could be plugged in.
    def interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, 1))
    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return np.zeros((len(tau), self.dimension, self.dimension))
