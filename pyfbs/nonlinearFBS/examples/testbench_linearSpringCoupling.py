"""
Linear-spring (bushing) FBS coupling of the pyFBS lab testbench.

Substructures A and B are coupled through a linear spring acting on all six
virtual-point interface DoFs (3 translations + 3 rotations) instead of being
rigidly connected. This is the "bushing as compatibility relaxation" formulation:
the interface compatibility  B u = 0  is relaxed to  B u = Gamma * lambda, where
Gamma is the dynamic flexibility of the spring. The assembled admittance is then

    Y_assembled = Y - Y B^T (B Y B^T + Gamma)^-1 B Y                     (Eq. 17)

For Gamma -> 0 (infinitely stiff spring) this reduces to the standard rigid
LM-FBS coupling  Y - Y B^T (B Y B^T)^-1 B Y.

pyFBS has no dedicated coupling routine (the official 07_FBS_coupling tutorial
writes the LM-FBS algebra inline as well); the internal helper used here is
pyfbs.tpinv (batched truncated pseudo-inverse) for the interface inversion. The
result is plotted with matplotlib because every pyfbs.display plotter is Altair
based (renders to HTML/browser only) and cannot open a native Python window.

Run this script from  pyfbs/nonlinearFBS/examples/  (the lab_testbench data lives
there; download_lab_testbench() is a no-op if the files are already present).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyfbs

# ---------------------------------------------------------------------------
# Linear spring parameters (one spring per interface DoF) -- tune these.
#   k_trans : translational stiffness [N/m]    for ux, uy, uz
#   k_rot   : rotational    stiffness [Nm/rad]  for rx, ry, rz
#   c_trans : translational viscous damping [Ns/m]    (0.0 = pure spring)
#   c_rot   : rotational    viscous damping [Nms/rad]
# ---------------------------------------------------------------------------
k_trans, k_rot = 1.0e6, 1.0e3
c_trans, c_rot = 0.0, 0.0

# ---------------------------------------------------------------------------
# 1) Data: pyFBS lab testbench.
# ---------------------------------------------------------------------------
pyfbs.io.download_lab_testbench()
pos_xlsx = r"./lab_testbench/Measurements/coupling_example.xlsx"

df_chn_A = pd.read_excel(pos_xlsx, sheet_name="Channels_A")
df_imp_A = pd.read_excel(pos_xlsx, sheet_name="Impacts_A")
df_chn_B = pd.read_excel(pos_xlsx, sheet_name="Channels_B")
df_imp_B = pd.read_excel(pos_xlsx, sheet_name="Impacts_B")
df_vp    = pd.read_excel(pos_xlsx, sheet_name="VP_Channels")
df_vpref = pd.read_excel(pos_xlsx, sheet_name="VP_RefChannels")

# ---------------------------------------------------------------------------
# 2) Numerical models -> synthesized receptance FRFs.
# ---------------------------------------------------------------------------
MK_A = pyfbs.mck.Model.from_ansys(r"./lab_testbench/FEM/A.rst", r"./lab_testbench/FEM/A.full",
                                  no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)
MK_B = pyfbs.mck.Model.from_ansys(r"./lab_testbench/FEM/B.rst", r"./lab_testbench/FEM/B.full",
                                  no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)

# snap channel/impact locations to the nearest FE node (numerical case)
df_chn_A = MK_A.update_locations_df(df_chn_A, scale=1)
df_imp_A = MK_A.update_locations_df(df_imp_A, scale=1)
df_chn_B = MK_B.update_locations_df(df_chn_B, scale=1)
df_imp_B = MK_B.update_locations_df(df_imp_B, scale=1)

MK_A.frf_synth(df_chn_A, df_imp_A, f_start=0, modal_damping=0.003)  # receptance (default)
MK_B.frf_synth(df_chn_B, df_imp_B, f_start=0, modal_damping=0.003)

freq  = MK_A.freq
omega = 2 * np.pi * freq

# ---------------------------------------------------------------------------
# 3) Virtual point transformation -> collocated 6-DoF interface admittances.
# ---------------------------------------------------------------------------
vpt_A = pyfbs.interface.VPT(df_chn_A, df_imp_A, df_vp, df_vpref)
vpt_B = pyfbs.interface.VPT(df_chn_B, df_imp_B, df_vp, df_vpref)
vpt_A.apply_vpt(MK_A.freq, MK_A.frf)
vpt_B.apply_vpt(MK_B.freq, MK_B.frf)

Y_A = vpt_A.frf   # (N_freq, nA, nA): A reference DoFs first, 6 VP interface DoFs last
Y_B = vpt_B.frf   # (N_freq, nB, nB): 6 VP interface DoFs first, B reference DoFs after

# ---------------------------------------------------------------------------
# 4) Uncoupled block-diagonal admittance  Y = diag(Y_A, Y_B).
# ---------------------------------------------------------------------------
nA, nB = Y_A.shape[1], Y_B.shape[1]
N = nA + nB
Y = np.zeros((len(freq), N, N), dtype=complex)
Y[:, :nA, :nA] = Y_A
Y[:, nA:, nA:] = Y_B

# ---------------------------------------------------------------------------
# 5) Signed Boolean matrix B coupling the 6 VP interface DoFs of A to those of B.
#    Interface DoFs are the last 6 columns of A and the first 6 columns of B.
# ---------------------------------------------------------------------------
n_if = 6
A_if = np.arange(nA - n_if, nA)   # A interface DoFs (global indices)
B_if = np.arange(nA, nA + n_if)   # B interface DoFs (global indices)

Bc = np.zeros((n_if, N))
Bc[np.arange(n_if), A_if] = 1.0
Bc[np.arange(n_if), B_if] = -1.0

# ---------------------------------------------------------------------------
# 6) Spring dynamic flexibility Gamma(omega) on the 6 interface DoFs.
#    VP order is [ux, uy, uz, rx, ry, rz]; Gamma is diagonal and (through the
#    damping term) frequency dependent:  Gamma_ii = 1 / (k_i + i*omega*c_i).
# ---------------------------------------------------------------------------
k_diag = np.array([k_trans, k_trans, k_trans, k_rot, k_rot, k_rot])
c_diag = np.array([c_trans, c_trans, c_trans, c_rot, c_rot, c_rot])

Z_spring = k_diag[None, :] + 1j * omega[:, None] * c_diag[None, :]   # (N_freq, 6)
Gamma = np.zeros((len(freq), n_if, n_if), dtype=complex)
d = np.arange(n_if)
Gamma[:, d, d] = 1.0 / Z_spring

# ---------------------------------------------------------------------------
# 7) Assembled admittance (LM-FBS dual form; algebra is inline as in tutorial 07,
#    but the interface inverse uses pyFBS's batched truncated pseudo-inverse).
#    Spring (Eq. 17):  Y - Y B^T (B Y B^T + Gamma)^-1 B Y
#    Rigid (limit) :   Y - Y B^T (B Y B^T)^-1        B Y
#    (2-D Bc broadcasts over the leading frequency axis of the (N_freq,N,N) stack.)
# ---------------------------------------------------------------------------
BY   = Bc @ Y        # (N_freq, 6, N)
YBt  = Y @ Bc.T      # (N_freq, N, 6)
Yint = Bc @ YBt      # (N_freq, 6, 6)  == B Y B^T

# trunc=0 -> full pseudo-inverse (no truncation). tpinv's documented trunc=None
# "no truncation" path is broken upstream (passes None into a subtraction), so
# trunc=0 is the correct no-truncation setting.
Y_spring = Y - YBt @ pyfbs.tpinv(Yint + Gamma, trunc=0) @ BY
Y_rigid  = Y - YBt @ pyfbs.tpinv(Yint,         trunc=0) @ BY

# ---------------------------------------------------------------------------
# 8) Drop the 12 interface DoFs, keep physical reference DoFs, and compare the
#    rigid and spring coupling in a native matplotlib window (magnitude + phase).
# ---------------------------------------------------------------------------
ref = np.r_[np.arange(0, nA - n_if), np.arange(nA + n_if, N)]  # A ref + B ref
Y_spring_ref = Y_spring[:, ref][:, :, ref]
Y_rigid_ref  = Y_rigid[:,  ref][:, :, ref]

out = 0             # response DoF on substructure A
inp = nA - n_if     # force DoF on substructure B (first B reference DoF) -> cross-interface FRF

fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
for label, Yc in (("rigid coupling", Y_rigid_ref), ("spring coupling", Y_spring_ref)):
    ax_mag.semilogy(freq, np.abs(Yc[:, out, inp]), label=label)
    ax_ph.plot(freq, np.angle(Yc[:, out, inp]), label=label)
ax_mag.set_ylabel("Receptance |Y|  [m/N]")
ax_mag.set_title("Linear-spring vs rigid FBS coupling of testbench A + B")
ax_mag.grid(True, which="both", alpha=0.3)
ax_mag.legend()
ax_ph.set_ylabel("Phase [rad]")
ax_ph.set_xlabel("Frequency [Hz]")
ax_ph.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()