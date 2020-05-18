import re

from numpy import ndarray
import matplotlib.pyplot as plt
import math as mt
import cmath as cmt
import numpy as np
import pyansys
import numpy as np
import scipy as sp
from scipy.sparse import linalg,diags
import pandas as pd
import pyansys
from tqdm import tqdm

from scipy.sparse import linalg
import scipy as sp
import numpy as np
from scipy import spatial


class MK_model(object):
    def __init__(self,ress_file,full_file,no_modes = 100,solve = True,modal_damping = 0.003):
        rst = pyansys.read_binary(ress_file)
        self.nodes = rst.geometry["nodes"][:, :3]  # only translational dofs

        full = pyansys.read_binary(full_file)

        self.dof_ref, self.K, self.M = full.load_km(sort=False)  # dof_ref: 0-x 1-y 2-z

        self.K += diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape)

        self.M += sp.sparse.triu(self.M, 1).T
        self.K += sp.sparse.triu(self.K, 1).T

        self.modal_damping = modal_damping

        if solve:
            self.eig_freq, self.eigen_val, self.eigen_vec = self.eig_solve(self.M,self.K,no_modes)




    def eig_solve(self,mass_mat, stiff_mat, no_modes):
        # tolerances and sigma may significantly affect the output!
        eigen_val, eigen_vec = sp.sparse.linalg.eigsh(stiff_mat, k=no_modes, M=mass_mat, sigma=10000, tol=1e-3)

        eigen_val = np.clip(eigen_val, 0, np.max(eigen_val))  # avoiding negative values
        eigen_freq = np.sqrt(eigen_val)  #/(2*np.pi)
        return (eigen_freq, eigen_val, eigen_vec)


    def find_nearest_locations(self,dense_mesh_points, sparse_mesh_points):
        """
        This function finds the nearest coordinate locations of sparse mesh in the corresponding dense mesh.
            :param dense_mesh_points: nodal coordinates of dense mesh in 3D space
            :type dense_mesh_points: array
            :param sparse_mesh_points: nodal coordinates of sparse mesh in 3D space
            :type sparse_mesh_points: array
            :param dense_mesh_node_id: nodal coordinates id of sparse mesh
            :type dense_mesh_node_id: array
            :return: Selected nodes by index and by id regarding the dense mesh
            :rtype: (array(int), array(int))

        """
        tree = spatial.KDTree(list(
            zip(dense_mesh_points[:, 0].ravel(), dense_mesh_points[:, 1].ravel(), dense_mesh_points[:, 2].ravel())))
        selected_dense_mesh_node_index = (tree.query(sparse_mesh_points))[1]

        return selected_dense_mesh_node_index

    def FRF_synth(self,df_channel,df_impact,f_start = 1,f_end = 2000, d_points = 2000):
        imp = df_impact[["Position_1", "Position_2", "Position_3"]].to_numpy()
        imp_dir = df_impact[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()

        chn = df_channel[["Position_1", "Position_2", "Position_3"]].to_numpy()
        chn_dir = df_channel[["Direction_1", "Direction_2", "Direction_3"]].to_numpy()

        index_imp = self.find_nearest_locations(self.nodes, imp)
        index_ch = self.find_nearest_locations(self.nodes, chn)

        freq = np.linspace(f_start, f_end, d_points)
        FRF = np.zeros((len(index_ch), len(index_imp), len(freq)), dtype=complex)

        ome = 2 * np.pi * freq
        damp = self.modal_damping

        for j, ind_ch in enumerate(index_ch):
            mask = self.dof_ref[:, 0] == ind_ch + 1
            gg_chn = np.where(mask == True)

            # each direction in channel
            for k, sel2 in enumerate(gg_chn[0]):

                # for loop for each impact
                for i, ind in enumerate(index_imp):
                    mask = self.dof_ref[:, 0] == ind + 1
                    gg = np.where(mask == True)

                    # for loop for each direction in impact
                    for p, sel1 in enumerate(gg[0]):
                        # print("IN:",i,"OUT:",j,"sel1",sel1,"sel2",sel2)

                        # each modeshape
                        for no in range(self.eigen_vec.shape[1]):
                            FRF[j, i] += self.eigen_vec[sel1, no] * chn_dir[j, k] * self.eigen_vec[sel2, no] * imp_dir[
                                i, p] / (-ome ** 2 + 2 * 1j * damp * ome * self.eig_freq[no] + self.eig_freq[no] ** 2)

        FRF *= -(2 * np.pi * freq) ** 2

        self.freq = freq
        self.FRF = FRF



if __name__ == '__main__':
    print("Test: Dog!")


