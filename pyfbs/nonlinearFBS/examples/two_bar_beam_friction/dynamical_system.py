"""Two bar+beam elements connected by a frictional contact element.

Reproduction of Tiago Martins' MSc thesis case study 6.4 ("Two bar+beam elements
with frictional contact", figure 6.13): two identical cantilever bar+beam
elements, each clamped at one node and carrying three tip DOFs

    element 1 tip :  q1 (axial / longitudinal),  q2 (transverse),  q3 (slope)
    element 2 tip :  q4 (axial / longitudinal),  q5 (transverse),  q6 (slope)

EACH ELEMENT IS ONE SUBSTRUCTURE. They are uncoupled in the LINEAR system
(block-diagonal M, K), and coupled only through a single nonlinear contact
element between the tips, carrying two nonlinearities:

    * NORMAL (vertical) -- unilateral contact / contact-separation between q2 and
      q5 across a small vertical gap eps.
    * TANGENTIAL (horizontal) -- dry friction along the axial direction, driven
      by the relative tangential velocity d/dt(q1 - q4) and bounded by mu*N.

Two interface relative coordinates, ordered [normal ; tangential]::

    x_N = q5 - q2            (normal,     contact when x_N > eps)
    x_T = q1 - q4            (tangential, sliding)

so that  B^T lambda  with  lambda = [N, f_T]  reproduces the thesis nonlinear
force  f_nl = [f_T, -N, 0, -f_T, N, 0]^T  exactly.

The thesis measures the normal gap as q_perp = eps + q2 - q5 = eps - x_N, so
penetration (q_perp < 0) is x_N > eps. That is ALREADY pyFBS's DLFTContact sign
convention -- gap positive in penetration, contact when L_N q_rel > g_zero -- so
both systems below share it with g_zero = eps.

Two models of the SAME physics are provided:

    BarBeamAFT   -- the thesis MHBM model, both nonlinearities regularized:
        normal      N   = softplus(k*alpha2*(x_N - eps)) / alpha2      (eq. 6.15)
        tangential  f_T = mu * N * tanh(alpha1 * d/dt x_T)             (eq. 6.11)
      Solve with AFT().

    BarBeamDLFT  -- normal contact by DLFT (RIGID: no k, no alpha2), friction by
      the IDENTICAL regularized tanh law, driven by the DLFT normal force.
      Solve with DLFTContactAFT(L_N, L_F, epsilon, g_zero=eps).

Numerical values (thesis):  l = 1, EA = 1/3, EI = 1/3, lambda = 1;
contact  alpha2 = 2, k = 500, eps = 0.01, alpha1 = 150, mu = 0.1;
forcing  P1 = 0.1 (axial, harmonic), P5 = 0.4 (transverse, harmonic);
damping  C = 0.05 * K  (first undamped resonance at omega = 1).

DEVIATION FROM THE THESIS: the thesis clamps the tips together with a STATIC
P5, which leaves the normal direction statically determinate -- axial and
bending are uncoupled in the linear system, the only harmonic force is axial,
and nothing can modulate the contact, so N comes out exactly constant (every
harmonic h >= 1 of the contact force is identically zero). Here P5 is applied
HARMONICALLY on q2 instead, which drives x_N directly and makes the contact open
and close within the period.
"""
from dataclasses import dataclass, asdict

import numpy as np
from numpy import zeros, sin, cos, tanh, exp, log1p
from scipy.linalg import eigh
from scipy.special import expit

from pyfbs.nonlinearFBS import FBS_System


# DOF ordering of the assembled 6-DOF system (figure 6.13).
Q1, Q2, Q3, Q4, Q5, Q6 = 0, 1, 2, 3, 4, 5
DOF_LABELS = ("q1", "q2", "q3", "q4", "q5", "q6")

# Interface ordering [normal ; tangential] and the localisation matrices that
# split it for DLFTContactAFT. They must be orthonormal and mutually orthogonal.
N_IF = 2
IF_LABELS = ("x_rel_N", "x_rel_T")
L_N = np.array([[1.0, 0.0]])
L_F = np.array([[0.0, 1.0]])


# ============================ FE assembly ==================================

@dataclass
class BarBeamParams:
    """Bar+beam element + contact parameters (thesis case study 6.4 defaults)."""
    l:      float = 1.0       # element length
    EA:     float = 1.0 / 3   # axial rigidity  E*A
    EI:     float = 1.0 / 3   # bending rigidity E*I
    lam:    float = 1.0       # linear density  lambda

    # contact / friction
    k:      float = 500.0     # normal contact (surface) stiffness   [AFT only]
    alpha2: float = 2.0       # normal softplus regularization       [AFT only]
    eps:    float = 0.01      # vertical gap between the tips
    alpha1: float = 150.0     # tangential tanh regularization (stick slope)
    mu:     float = 0.1       # Coulomb friction coefficient

    # forcing / damping
    P1:     float = 0.1       # amplitude of the axial harmonic excitation on q1
    # Amplitude of the transverse harmonic excitation on q2, which drives the
    # normal contact in and out of touch. NOT the thesis' 0.4: that value is a
    # STATIC clamping force, and driving it harmonically instead puts the free
    # normal motion ~48 gaps deep into the contact, an impacting regime that 16
    # harmonics cannot resolve and neither method converges on. 0.02 leaves the
    # free motion at ~2.4 gaps, so the contact opens and closes every period.
    P2:     float = 0.1
    P5:     float = 0.02
    beta:   float = 0.05      # stiffness-proportional damping  C = beta * K

    def to_dict(self):
        """Plain-JSON view, for the result CSV header."""
        return asdict(self)


def element_matrices(p: BarBeamParams):
    """Elementary 3x3 stiffness and mass matrices (thesis eq. 6.12).

    DOF order per element: [axial, transverse, slope]. The clamped node's three
    DOFs are already removed, so this is the tip block of one cantilever.
    """
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


def assemble(p: BarBeamParams):
    """Assemble the global 6x6 block-diagonal M, C, K (thesis eq. 6.13).

    The two substructures do not share DOFs, so the linear system is uncoupled:
        K = blkdiag(Ke, Ke),  M = blkdiag(Me, Me),  C = beta * K.
    """
    Ke, Me = element_matrices(p)
    M = zeros((6, 6)); K = zeros((6, 6))
    M[:3, :3] = Me; M[3:, 3:] = Me
    K[:3, :3] = Ke; K[3:, 3:] = Ke
    C = p.beta * K
    return M, C, K


def modal_model(p: BarBeamParams):
    """Real modal basis of the block-diagonal linear system.

    Returns (angular_eig_freq, eig_vec, modal_damping) ready for
    ``NumericalFRF.from_modal``. ``eigh(K, M)`` mass-normalizes the modes
    (Phi^T M Phi = I), which is what the modal-sum receptance assumes.

    Six DOFs give six modes, i.e. a COMPLETE basis: the modal FRF is exact here
    and mode truncation is not a variable of this example.

    Damping is stiffness-proportional, C = beta*K, so mode r has
    2 zeta_r Omega_r = beta Omega_r^2, i.e. a PER-MODE ratio zeta_r =
    beta*Omega_r/2 -- not a constant.
    """
    M, _, K = assemble(p)
    eigenvalues, Phi = eigh(K, M)
    Omega = np.sqrt(np.clip(eigenvalues, 0.0, None))
    zeta = 0.5 * p.beta * Omega
    return Omega, Phi, zeta


def coupling_matrix():
    """Signed-Boolean B with x_rel = B q, ordered [normal ; tangential]."""
    B = zeros((N_IF, 6))
    B[0, Q5] = +1.0; B[0, Q2] = -1.0            # normal      x_N = q5 - q2
    B[1, Q1] = +1.0; B[1, Q4] = -1.0            # tangential  x_T = q1 - q4
    return B


def static_contact_state(p: BarBeamParams, rigid: bool = False):
    """Static (harmonic-0) contact state under the clamping force alone.

    Bending and axial dynamics are uncoupled in the linear system and the
    clamping force is constant, so the normal contact is essentially static.
    With equal substructures the tips deflect equally and oppositely,
    q2 = -q5 = N - P5, hence x_N = q5 - q2 = 2*compliance*(P5 - N).

    :param rigid: True -> the DLFT limit, zero penetration (x_N = eps exactly);
        False -> the AFT softplus law, solved by bisection on N.
    :returns: (x_N, N) -- the static normal gap coordinate and compression force.

    Used only to seed the continuation's h=0 coefficient; a rough value is fine.
    """
    # tip transverse compliance of one cantilever under a tip force: l^3/(3 EI)
    compliance = p.l ** 3 / (3.0 * p.EI)
    if rigid:
        x_N = p.eps
        N = p.P5 - 0.5 * p.eps / compliance
        return x_N, N

    # N_law(gap_of(N)) - N is strictly decreasing in N, so plain bisection works
    def gap_of(N):
        return 2.0 * compliance * (p.P5 - N)
    lo, hi = 0.0, 2.0 * p.P5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        N_law, _ = _softplus_normal(np.array([gap_of(mid)]), p)
        if N_law[0] > mid:
            lo = mid
        else:
            hi = mid
    N = 0.5 * (lo + hi)
    return gap_of(N), N


# ============================ force laws ===================================

def _softplus_normal(xN, p: BarBeamParams):
    """Regularized normal compression force and its slope (thesis eq. 6.15).

        N      = softplus(c) / alpha2,   c = k*alpha2*(xN - eps)
        dN/dxN = k * sigmoid(c)

    softplus(c) = log(1 + exp(c)), evaluated so that large c cannot overflow.
    """
    c = p.k * p.alpha2 * (xN - p.eps)
    # stable softplus: log1p(exp(-|c|)) + max(c, 0)
    softplus = log1p(exp(-np.abs(c))) + np.maximum(c, 0.0)
    N        = softplus / p.alpha2
    # expit is the overflow-safe sigmoid: plain 1/(1+exp(-c)) overflows for c << 0,
    # which large alpha2 (a sharp regularization) reaches easily
    dN_dxN   = p.k * expit(c)
    return N, dN_dxN


# ============================ common base ==================================

class _BarBeamBase(FBS_System):
    """Shared assembly, coupling, forcing and reference frequency.

    The thesis works in coordinates where the first undamped (axial) resonance
    sits at omega = 1, so omega_ref = 1 and the matrices are used directly.
    """
    is_real_valued = True

    def __init__(self, params: BarBeamParams = None, sample_number: int = 256):
        self.p = params or BarBeamParams()
        self.sample_number = sample_number
        self.omega_ref = 1.0                         # first undamped resonance
        self.B_coupling = coupling_matrix()          # (2, 6)

        M, C, K = assemble(self.p)
        self.mass_matrix, self.damping_matrix, self.stiffness_matrix = M, C, K
        self.angular_eig_freq, self.eig_vec, self.zeta = modal_model(self.p)

    def external_term(self, tau):
        """f_ext = [P1 cos(tau), P5 cos(tau), 0, 0, 0, 0]^T.

        BOTH components are harmonic, at the fundamental. P1 drives the AXIAL
        direction on q1, i.e. the tangential (sliding) coordinate; P5 drives the
        TRANSVERSE direction on q2 and so modulates the normal coordinate
        x_N = q5 - q2 directly.

        There is no static clamping preload any more: instead of resting under a
        constant compression, the contact now OPENS AND CLOSES within every
        period. Harmonic 0 is still mandatory -- the contact rectifies, so the
        contact force and the response both keep a non-zero mean.
        """
        f = zeros((len(tau), 6, 1))
        f[:, Q1, 0] = self.p.P1 * cos(tau)          # axial -> tangential sliding
        f[:, Q2, 0] = - self.p.P2 * sin(tau) - self.p.P5         # transverse -> normal contact
        f[:, Q5, 0] = self.p.P5
        return f


# ============================ AFT (thesis regularization) ==================

class BarBeamAFT(_BarBeamBase):
    """Both nonlinearities regularized -- the thesis MHBM model. Use with AFT().

    Interface force, ordered [normal ; tangential]::

        lambda_N = N   = softplus(k*alpha2*(x_N - eps)) / alpha2
        lambda_T = f_T = mu * N * tanh(alpha1 * d/dt x_T)

    The normal force depends on the normal displacement only; the tangential
    force depends on the normal displacement (through N) and on the tangential
    velocity.
    """

    def interface_force(self, u_rel, udot_rel, tau):
        p = self.p
        xN = u_rel[:, 0, 0]                          # x_N = q5 - q2
        vT = udot_rel[:, 1, 0]                       # d/dt x_T = q1dot - q4dot
        N, _ = _softplus_normal(xN, p)
        f = zeros((len(tau), N_IF, 1))
        f[:, 0, 0] = N
        f[:, 1, 0] = p.mu * N * tanh(p.alpha1 * vT)
        return f

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        """d lambda / d x_rel  (displacement)."""
        p = self.p
        xN = u_rel[:, 0, 0]
        vT = udot_rel[:, 1, 0]
        _, dN = _softplus_normal(xN, p)
        J = zeros((len(tau), N_IF, N_IF))
        J[:, 0, 0] = dN                                    # dN/dx_N
        J[:, 1, 0] = p.mu * tanh(p.alpha1 * vT) * dN       # df_T/dx_N (through N)
        return J

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        """d lambda / d (d/dt x_rel)  (velocity)."""
        p = self.p
        xN = u_rel[:, 0, 0]
        vT = udot_rel[:, 1, 0]
        N, _ = _softplus_normal(xN, p)
        th = tanh(p.alpha1 * vT)
        J = zeros((len(tau), N_IF, N_IF))
        J[:, 1, 1] = p.mu * N * p.alpha1 * (1.0 - th * th)   # df_T/dv_T
        return J


# ============================ DLFT contact + AFT friction ==================

class BarBeamDLFT(_BarBeamBase):
    """Rigid DLFT normal contact + the SAME regularized tanh friction.

    Use with ``DLFTContactAFT(L_N, L_F, epsilon, g_zero=eps)``.

    There is no additional AFT law on the full interface, so the three
    ``interface_force*`` hooks return zeros -- anything they returned would enter
    the DLFT normal prediction. The friction law is identical to
    :class:`BarBeamAFT`'s, with the regularized softplus N replaced by the DLFT
    normal force ``f_normal``; k and alpha2 play no role at all.
    """

    # --- f^nl,A on the full interface: none ---

    def interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, 1))

    def jacobian_interface_force(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, N_IF))

    def jacobian_interface_force_qdot(self, u_rel, udot_rel, tau):
        return zeros((len(tau), N_IF, N_IF))

    # --- f^F on the friction DOFs (n_F = n_N = 1), bounded by the DLFT normal ---

    def friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        """f_T = mu * f_normal * tanh(alpha1 * v_T).   -> (Nt, n_F, 1)"""
        p = self.p
        return p.mu * f_normal * tanh(p.alpha1 * udot_F_rel)

    def jacobian_friction_force(self, u_F_rel, udot_F_rel, f_normal, tau):
        """df_T/dx_T = 0 -- the law has no tangential-displacement dependence."""
        return zeros((len(tau), 1, 1))

    def jacobian_friction_force_qdot(self, u_F_rel, udot_F_rel, f_normal, tau):
        """df_T/dv_T = mu * f_normal * alpha1 * (1 - tanh^2)."""
        p = self.p
        th = tanh(p.alpha1 * udot_F_rel[:, :, 0])
        slope = p.mu * f_normal[:, :, 0] * p.alpha1 * (1.0 - th * th)
        return slope[:, :, None]                     # (Nt, 1) -> (Nt, 1, 1)

    def jacobian_friction_force_normal(self, u_F_rel, udot_F_rel, f_normal, tau):
        """df_T/df^N = mu * tanh(alpha1 * v_T).   -> (Nt, n_F, n_N)"""
        p = self.p
        return p.mu * tanh(p.alpha1 * udot_F_rel)


SYSTEM_TYPES = {"aft": BarBeamAFT, "dlft_aft": BarBeamDLFT}
