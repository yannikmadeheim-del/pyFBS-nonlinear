"""Dynamical-system definitions for the TWO-ROD vibro-impact example.

Two clamped-free axial FE rods face each other across a gap g0:

    Rod A (left)   -- clamped to the LEFT wall, free tip A pointing right; this
                      rod is harmonically driven at its tip (F0 cos(tau)).
    Rod B (right)  -- clamped to the RIGHT wall, free tip B pointing left; an
                      IDENTICAL but PASSIVE rod that rings only through contact.

There is no obstacle spring: the right-hand "wall" of the single-rod example is
replaced by a real, compliant FE rod, so the only nonlinearity is the unilateral
contact between the two tips.

Coordinate convention (both tips positive TOWARD the contact)
------------------------------------------------------------
u_A > 0 moves tip A rightward (toward B); u_B > 0 moves tip B leftward (toward A).
Both rods therefore reuse the orientation-agnostic bar matrices unchanged.  The
relative interface DOF (the total approach of the two tips) is

    x_r = u_A + u_B,        B = [ +1 (tip A), +1 (tip B) ],

the initial tip-to-tip gap is g0, penetration occurs when x_r > g0, and the
no-penetration constraint is x_r <= g0 -- the SAME algebraic form as the single-
rod example, so DLFTContact / AFT / the runner / the plots carry over unchanged.

Two models of the SAME physics:

    TwoRodVibroImpact    -- DLFT, rigid unilateral tip-to-tip contact (x_r <= g0).
                            No penalty spring.  Both tips are compliant (finite
                            k_rod), so the interface admittance is well conditioned
                            and this converges cold; the dense run is the reference.
    TwoRodPenaltyContact -- AFT, a tanh-REGULARIZED one-sided penalty spring k_c
                            between the tips.  k_c = k_rel * k_rod; alpha is the
                            regularization sharpness (alpha->inf => hard, nonsmooth
                            contact == "without regularization").

Material / geometry (Vadcard Table 1) live in :class:`RodParams`; both rods are
identical.
"""
from dataclasses import dataclass

import numpy as np
from numpy import zeros, cos
from scipy.linalg import eigh

from pyfbs.nonlinearFBS import FBS_System


# ============================ rod assembly ==================================

@dataclass
class RodParams:
    """Clamped-free axial bar parameters (Vadcard Table 1 defaults)."""
    n_elem: int = 10          # number of two-node bar elements (per rod)
    L:      float = 0.13      # rod length [m]
    E:      float = 210e9     # Young's modulus [Pa]
    rho:    float = 7800.0    # density [kg/m^3]
    A:      float = 15.6e-4   # cross section [m^2]   (15.6 cm^2)
    xi:     float = 7.5e-3    # modal damping ratio
    F0:     float = 25e3      # harmonic forcing at the driven tip A [N]
    sample_number : int = 600        # AFT/aliasing polynomial degree

    @property
    def k_rod(self) -> float:
        """Static axial stiffness E*A/L [N/m] -- the contact-stiffness reference."""
        return self.E * self.A / self.L


def assemble_rod(p: RodParams):
    """Assemble ONE clamped-free rod and return its FE operators.

    :returns: dict with
        M_rod, K_rod, C_rod   -- (n, n) rod matrices (clamped node dropped),
        omega_modes           -- (n,) undamped natural frequencies [rad/s],
        omega_ref             -- omega_modes[0], the nondimensionalization base,
        n                     -- free DOF count (= n_elem).
    """
    n = p.n_elem
    l = p.L / n
    Me = (p.rho * p.A * l / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    Ke = (p.E * p.A / l)         * np.array([[1.0, -1.0], [-1.0, 1.0]])

    M_rod = zeros((n + 1, n + 1))
    K_rod = zeros((n + 1, n + 1))
    for e in range(n):
        M_rod[e:e + 2, e:e + 2] += Me
        K_rod[e:e + 2, e:e + 2] += Ke
    M_rod = M_rod[1:, 1:]            # drop clamped node 0
    K_rod = K_rod[1:, 1:]

    # undamped modes -> omega_ref for the nondimensionalization.  No C matrix is
    # built: proportional damping enters later as the modal ratio zeta = xi.
    w2, Phi = eigh(K_rod, M_rod)
    omega_modes = np.sqrt(np.clip(w2, 0.0, None))

    return dict(M_rod=M_rod, K_rod=K_rod,
                omega_modes=omega_modes, omega_ref=omega_modes[0], n=n)


def assemble_two_rods(p: RodParams, p_B: RodParams = None):
    """Block-diagonal assembly of two clamped-free rods (A driven, B passive).

    Rod A always uses ``p`` (the Vadcard rod of the single-rod example).  Rod B
    defaults to an identical copy but can be ANY other clamped-free rod via
    ``p_B`` -- e.g. a different length L_B to raise/lower its static stiffness
    k_B = E*A/L_B.

    DOF layout:  indices [0 .. nA)        -> rod A,  index nA-1       = tip A,
                 indices [nA .. nA+nB)    -> rod B,  index nA+nB-1    = tip B.

    omega_ref is ROD A's first axial mode, so the nondimensional sweep window
    omega/omega_1 stays the same as the single-rod study no matter what rod B is.
    """
    rA = assemble_rod(p)
    rB = rA if p_B is None else assemble_rod(p_B)
    nA, nB = rA["n"], rB["n"]
    d = nA + nB
    M = zeros((d, d)); M[:nA, :nA] = rA["M_rod"]; M[nA:, nA:] = rB["M_rod"]
    K = zeros((d, d)); K[:nA, :nA] = rA["K_rod"]; K[nA:, nA:] = rB["K_rod"]
    return dict(M=M, K=K, n=nA, d=d,
                tipA=nA - 1, tipB=nA + nB - 1,
                omega_ref=rA["omega_ref"], omega_modes=rA["omega_modes"])


def _zero_interface(dimension):
    """interface_force stub (Nt, d, 1) for DLFT systems where the method supplies F."""
    def _f(self, u_rel, udot_rel, tau):
        return zeros((len(tau), dimension, 1))
    return _f


# ============================ DLFT: rigid tip-to-tip contact ================

class TwoRodVibroImpact(FBS_System):
    """Two identical rods coupled by FBS; rigid unilateral DLFT contact on x_r.

    M = blkdiag(M_A, M_B),  C = blkdiag(C_A, C_B),  K = blkdiag(K_A, K_B)
    B = [ +1 (tip A) , +1 (tip B) ]   ->  x_r = u_A + u_B   (total approach)

    DLFTContact enforces x_r <= g0 exactly (no penalty spring).  Because both tips
    are compliant FE rods (not a rigid wall) the contact is numerically benign and
    converges from a cold linear start; the dense run is the ground-truth NFRC.
    """
    is_real_valued = True

    def __init__(self, params: RodParams = None, params_B: RodParams = None):
        p = params or RodParams()
        a = assemble_two_rods(p, params_B)

        self.k_rod = self.k_ref = p.k_rod            # rod A (epsilon/penalty scale)
        self.kB_rod = (params_B or p).k_rod          # rod B static stiffness E*A/L_B

        omega_ref = a["omega_ref"]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * a["M"]
        self.stiffness_matrix = a["K"]

        # modal data of the (nondimensional) coupled system -> NumericalFRF.from_modal.
        # proportional damping: every mode gets the same ratio zeta = xi (no C matrix).
        w2, Phi = eigh(self.stiffness_matrix, self.mass_matrix)   # Phi^T M Phi = I
        self.angular_eig_freq = np.sqrt(np.clip(w2, 0.0, None))   # Omega_hat (first = 1)
        self.eig_vec          = Phi
        self.zeta             = p.xi

        self.rod_tip_idx = a["tipA"]          # driven tip A (forcing + plot signal)
        self.tipB_idx    = a["tipB"]
        B = zeros((1, a["d"]))
        B[0, a["tipA"]] = +1.0
        B[0, a["tipB"]] = +1.0
        self.B_coupling = B

        self.dimension         = 1
        self.total_dimension   = a["d"]
        self.sample_number = RodParams.sample_number
        self.F0 = p.F0
        self.omega_modes = a["omega_modes"]

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)
        return f

    interface_force               = _zero_interface(1)
    jacobian_interface_force      = lambda self, u, ud, tau: zeros((len(tau), 1, 1))
    jacobian_interface_force_qdot = lambda self, u, ud, tau: zeros((len(tau), 1, 1))


# ============================ AFT: regularized penalty contact ==============

def _smoothed_ramp_1(d, alpha, k):
    """tanh-regularized ramp:  f(d) = k * 0.5*d*(1+tanh(alpha*d)) -> k*max(0,d)."""
    if not np.isfinite(alpha):
        r  = np.where(d > 0.0, k * d, 0.0)
        dr = np.where(d > 0.0, k,     0.0)
        return r, dr
    t  = np.tanh(alpha * d)
    H  = 0.5 * (1.0 + t)
    r  = k * (d * H)
    dr = k * (H + 0.5 * alpha * d * (1.0 - t * t))
    return r, dr


def _smoothed_ramp_2(d, alpha, k):
    """softplus-regularized ramp:  f(d) = k * ln(1+exp(alpha*d))/alpha -> k*max(0,d)."""
    if not np.isfinite(alpha):
        r  = np.where(d > 0.0, k * d, 0.0)
        dr = np.where(d > 0.0, k,     0.0)
        return r, dr
    r = k * np.logaddexp(0.0, alpha * d) / alpha  # = k*ln(1+exp(alpha*d))/alpha, overflow-safe
    dr = k * (1.0 / (1.0 + np.exp(-alpha * d)))  # = k*sigmoid(alpha*d)
    return r, dr


class TwoRodPenaltyContact(FBS_System):
    """Two identical rods with a tanh-REGULARIZED one-sided penalty spring (AFT).

    Same block-diagonal assembly and coupling B = [+1 (tip A), +1 (tip B)] as
    :class:`TwoRodVibroImpact`, but the contact is a smooth penalty force on the
    relative approach x_r = u_A + u_B:

        f_c(x_r) = k_c * r(x_r - g0),   r = tanh-smoothed ramp ~ max(0, .)

    Positive f_c (compression) is applied as -B^T f_c, i.e. it pushes BOTH tips
    back toward their walls -- the same sign convention as the DLFT multiplier, so
    the two methods are directly comparable.  k_c = k_rel * k_rod is the interface
    penalty stiffness; alpha is the regularization sharpness (alpha->inf => rigid,
    nonsmooth == "without regularization").  Velocity-independent.
    """
    is_real_valued = True

    def __init__(self, params: RodParams = None, params_B: RodParams = None,
                 k_rel: float = 1.0, g_zero: float = 0.0, alpha: float = 1.0e4):
        p = params or RodParams()
        if k_rel <= 0.0:
            raise ValueError("k_rel must be > 0 (k_c = k_rel * k_rod).")
        a = assemble_two_rods(p, params_B)

        self.k_rod = self.k_ref = p.k_rod            # rod A (penalty scale)
        self.kB_rod = (params_B or p).k_rod          # rod B static stiffness E*A/L_B
        self.k_c    = k_rel * p.k_rod
        self.k_obs  = self.k_c               # alias: comparison scripts read k_obs
        self.k_rel  = k_rel
        self.g_zero = g_zero
        self.alpha  = alpha

        omega_ref = a["omega_ref"]
        self.omega_ref        = omega_ref
        self.mass_matrix      = omega_ref ** 2 * a["M"]
        self.stiffness_matrix = a["K"]

        # modal data of the (nondimensional) coupled system -> NumericalFRF.from_modal.
        # proportional damping: every mode gets the same ratio zeta = xi (no C matrix).
        w2, Phi = eigh(self.stiffness_matrix, self.mass_matrix)   # Phi^T M Phi = I
        self.angular_eig_freq = np.sqrt(np.clip(w2, 0.0, None))   # Omega_hat (first = 1)
        self.eig_vec          = Phi
        self.zeta             = p.xi

        self.rod_tip_idx = a["tipA"]
        self.tipB_idx    = a["tipB"]
        B = zeros((1, a["d"]))
        B[0, a["tipA"]] = +1.0
        B[0, a["tipB"]] = +1.0
        self.B_coupling = B

        self.dimension         = 1
        self.total_dimension   = a["d"]
        self.sample_number = RodParams.sample_number
        self.F0 = p.F0
        self.omega_modes = a["omega_modes"]

    def external_term(self, tau):
        f = zeros((len(tau), self.total_dimension, 1))
        f[:, self.rod_tip_idx, 0] = self.F0 * cos(tau)
        return f

    # --- AFT penalty contact on the relative interface DOF x_r = u_A + u_B ---
    def interface_force(self, u_rel, udot_rel, tau):
        d = u_rel[:, :, 0] - self.g_zero               # (Nt, 1)
        r, _ = _smoothed_ramp_1(d, self.alpha, self.k_c)
        return r[:, :, None]            # (Nt, 1, 1)

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        d = u_rel[:, :, 0] - self.g_zero
        _, dr = _smoothed_ramp_1(d, self.alpha, self.k_c)
        return dr[:, :, None]           # (Nt, 1, 1)

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), 1, 1))
