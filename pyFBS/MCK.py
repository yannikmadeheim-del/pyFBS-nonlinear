from scipy.sparse import linalg,diags
import pyansys
from numpy.random import randn

import scipy as sp
import numpy as np
from scipy import spatial
from scipy.linalg import block_diag



class MK_model(object):
    """
    Initialization of the finite element model. Mass and stiffness matrices are imported and also nodes, DoFs and complete mesh of finite elements are defined. 
    If parameter ``solve`` is ``True``, also eigenfrequencies and eigenvectors are computed.

    :param ress_file: Path of the .rst file exported from Ansys
    :type ress_file: str
    :param full_file: Path of the .full file exported from Ansys
    :type full_file: str
    :param no_modes: Number of modes to be included in output of the eigenvalue computation.
    :type no_modes: int
    :param solve: If ``False`` just mass and stiffness matrices with corresponding nodes and their DoFs will be imported. If ``True`` also the eigenvalue problem will be solved.
    :type solve: bool
    """
    def __init__(self,ress_file,full_file,no_modes = 100,solve = True):
        rst = pyansys.read_binary(ress_file)
        self.nodes = rst.geometry["nodes"][:, :3]  # only translational dofs
        self.mesh = rst.grid
        self.no_modes = no_modes


        full = pyansys.read_binary(full_file)

        self.dof_ref, self.K, self.M = full.load_km(sort=True)  # dof_ref: 0-x 1-y 2-z

        self.K += diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape)

        self.M += sp.sparse.triu(self.M, 1).T
        self.K += sp.sparse.triu(self.K, 1).T


        if solve:
            self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M,self.K,no_modes)


    @staticmethod
    def eig_solve(mass_mat, stiff_mat, no_modes):
        """
        Description

        :param mass_mat:
        :param stiff_mat:
        :param no_modes:
        :return:
        """
        # tolerances and sigma may significantly affect the output!
        eigen_val, eigen_vec = sp.sparse.linalg.eigsh(stiff_mat, k=no_modes, M=mass_mat, sigma=10000, tol=1e-3)

        eigen_val = np.clip(eigen_val, 0, np.max(eigen_val))  # avoiding negative values
        eigen_freq = np.sqrt(eigen_val)  #/(2*np.pi)
        return (eigen_freq, eigen_val, eigen_vec)


    @staticmethod
    def find_nearest_locations(dense_mesh_points, sparse_mesh_points, dense_mesh_node_id=None):
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
        tree = spatial.KDTree(list(zip(dense_mesh_points[:, 0].ravel(), dense_mesh_points[:, 1].ravel(), dense_mesh_points[:, 2].ravel())))
        selected_dense_mesh_node_index = (tree.query(sparse_mesh_points))[1]

        if not (dense_mesh_node_id is None):
            selected_dense_mesh_node_id = dense_mesh_node_id[selected_dense_mesh_node_index]
            selected_dense_mesh_node_id = list(map(int, selected_dense_mesh_node_id))
            return selected_dense_mesh_node_index, selected_dense_mesh_node_id
        else:
            return selected_dense_mesh_node_index

    @staticmethod
    def data_preparation(df):
        """
        Description

        :param df:
        :return:
        """
        nodes = df[["Position_1", "Position_2", "Position_3"]].values
        directions = df[["Direction_1", "Direction_2", "Direction_3"]].values

        unique_nodes = nodes[np.sort(np.unique(nodes, axis=0, return_index=True)[1])]
        direction_nodes = []
        for node in unique_nodes:
            loc = np.where((nodes == node).all(axis=1))
            direction_nodes.append(directions[loc])
        return unique_nodes, np.asarray(direction_nodes)

    @staticmethod
    def loc_definition(response_point, response_direction, excitation_point, excitation_direction, rotation_included,
                       all_at_once=False):
        """
        Computation of DoF od specific node in specific direction to find location in modal matrix or global receptance matrix.

        :param response_point: number of node where responce is observed
        :type response_point: int or array(int)
        :param response_direction: direction of observed responnce (0-x, 1-y, 2-z)
        :type response_point: int or array(int)
        :param excitation_point: number of node where excitation is performed
        :type excitation_point: int or array(int)
        :param excitation_direction: direction of performed excitation (0-x, 1-y, 2-z)
        :type excitation_direction: int or array(int)
        :param rotation_included: definition of roations inclusion in DoFs in system
        :type rotation_included: bool
        :param all_at_once: compute all location as once, when response_point and excitation_point are arrays
        :type all_at_once: bool
        :return: sel1, sel2
        :rtype: (int, int)
        """
        if rotation_included:
            N_DOFs = 6
        else:
            N_DOFs = 3

        if all_at_once == False:
            sel1 = (response_point - 1) * N_DOFs + response_direction
            sel2 = (excitation_point - 1) * N_DOFs + excitation_direction
            # print(sel1,sel2)
        elif all_at_once == True:
            _sel1 = (response_point - 1) * N_DOFs
            _sel2 = (excitation_point - 1) * N_DOFs
            sel1 = []
            sel2 = []
            for i in response_direction:
                sel1.append(_sel1 + i)
            for i in excitation_direction:
                sel2.append(_sel2 + i)
            sel1 = np.ravel(sel1, 'F')  # combine all together in alternating way
            sel2 = np.ravel(sel2, 'F')  # combine all together in alternating way

        return sel1, sel2

    def FRF_synth(self,df_channel,df_impact,f_start = 0, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance"):
        """
        Description

        :param df_channel:s
        :param df_impact:
        :param f_start:
        :param f_end:
        :param f_resolution:
        :param limit_modes:
        :param modal_damping:
        :param type:
        :return:
        """
        unique_nodes_chn, direction_nodes_chn = self.data_preparation(df_channel)
        unique_nodes_imp, direction_nodes_imp = self.data_preparation(df_impact)

        index_chn = self.find_nearest_locations(self.nodes, unique_nodes_chn)
        index_imp = self.find_nearest_locations(self.nodes, unique_nodes_imp)

        rotation_included = False
        response_points = index_chn + 1
        response_directions = [0, 1, 2]
        excitation_points = index_imp + 1
        excitation_directions = [0, 1, 2]

        if limit_modes == None:
            no_modes = self.no_modes
        else:
            no_modes = limit_modes

        if modal_damping == None:
            damping = np.asarray([0] * no_modes)
        elif type(modal_damping) == float:
            damping = np.asarray([modal_damping] * no_modes)

        loc1, loc2 = self.loc_definition(response_points, response_directions, excitation_points, excitation_directions,
                                    rotation_included, all_at_once=True)

        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * freq
        ome2 = ome ** 2
        _eig_val2 = self.eig_freq ** 2

        m_p_chan = block_diag(*direction_nodes_chn) @ self.eig_vec[loc1, :no_modes]
        m_p_imp = block_diag(*direction_nodes_imp) @ self.eig_vec[loc2, :no_modes]
        m_p = np.einsum('ij,kj->jik', m_p_chan, m_p_imp)
        
        denominator = (_eig_val2[:no_modes, np.newaxis] - ome2) + np.einsum('ij,i->ij',
                                                                            (ome * self.eig_freq[:no_modes, np.newaxis]),
                                                                            (2 * 1j * damping[:no_modes]))
        FRF_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = FRF_matrix

        elif frf_type == "mobility":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, (1j*2*np.pi*freq))

        elif frf_type == "accelerance":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, -(2*np.pi*freq)**2)

        self.FRF = _temp
        self.freq = freq


    def add_noise(self,n1 = 1e-3, n2 = 1e-3, n3 = 8e-4 ,n4 = 7e-4):
        """
        Aditive noise

        :param n1:
        :param n2:
        :param n3:
        :param n4:
        :return:
        """
        self.FRF_noise = np.zeros_like(self.FRF, dtype=complex)

        for i in range(self.FRF.shape[0]):
            for j in range(self.FRF.shape[1]):
                noise = n1 * (randn(len(self.freq))) * np.abs(self.FRF[i, j]) + 1j * n2 * (randn(len(self.freq))) * np.abs(
                    self.FRF[i, j]) + n3 * (randn(len(self.freq))) + 1j * n4 * (randn(len(self.freq)))
                self.FRF_noise[i, j] = self.FRF[i, j] + noise


if __name__ == '__main__':
    print("Test: Dog!")


