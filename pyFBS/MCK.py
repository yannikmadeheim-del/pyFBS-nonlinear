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
    def __init__(self,ress_file,full_file,no_modes = 100,solve = True):
        rst = pyansys.read_binary(ress_file)
        self.nodes = rst.geometry["nodes"][:, :3]  # only translational dofs

        full = pyansys.read_binary(full_file)

        self.dof_ref, self.K, self.M = full.load_km(sort=False)  # dof_ref: 0-x 1-y 2-z

        self.K += diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape)

        self.M += sp.sparse.triu(self.M, 1).T
        self.K += sp.sparse.triu(self.K, 1).T

        if solve:
            self.eig_freq, self.eigen_val, self.eigen_vec = self.eig_solve(self.M,self.K,no_modes)


    def eig_solve(self,mass_mat, stiff_mat, no_modes):
        # tolerances and sigma may significantly affect the output!
        eigen_val, eigen_vec = sp.sparse.linalg.eigsh(stiff_mat, k=no_modes, M=mass_mat, sigma=10000, tol=1e-3)

        eigen_val = np.clip(eigen_val, 0, np.max(eigen_val))  # avoiding negative values
        eigen_freq = np.sqrt(eigen_val)  # /(2*np.pi)
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

        # print(selected_dense_mesh_node_index)
        # print(np.unique(selected_dense_mesh_node_index))

        _, idx = np.unique(selected_dense_mesh_node_index, return_index=True)
        return selected_dense_mesh_node_index[np.sort(idx)]

    def mode_superposition(self):
        print("blah")


if __name__ == '__main__':
    print("Test: Dog!")


