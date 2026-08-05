from pyfbs.mck import Model

n_freqs = 60

MK_A = Model.from_ansys(r"./lab_testbench/FEM/A.rst", r"./lab_testbench/FEM/A.full",
                                  no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)
MK_B = Model.from_ansys(r"./lab_testbench/FEM/B.rst", r"./lab_testbench/FEM/B.full",
                                  no_modes=100, allow_pickle=True, recalculate=False, mesh_scale=1)

eig_freq_A = MK_A.eig_freq                   # eigenfrequencies in Hz      (np.array)
angular_eig_freq_A = MK_A.angular_eig_freq   # eigenfrequencies in rad/s
eig_val_A = MK_A.eig_val                     # eigenvalues (ω²)
eig_vec_A = MK_A.eig_vec                     # mode shapes

eig_freq_B = MK_B.eig_freq                   # eigenfrequencies in Hz      (np.array)
angular_eig_freq_B = MK_B.angular_eig_freq   # eigenfrequencies in rad/s
eig_val_B = MK_B.eig_val                     # eigenvalues (ω²)
eig_vec_B = MK_B.eig_vec

print("Eigenfrequencies of A:")
for i in range(n_freqs):
    print(f"\t{i+1}: {eig_freq_A[i]}")

print("Eigenfrequencies of B:")
for j in range(n_freqs):
    print(f"\t{j+1}: {eig_freq_B[j]}")