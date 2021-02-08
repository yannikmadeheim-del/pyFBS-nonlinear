from scipy.sparse import linalg,diags
import pyansys
from numpy.random import randn
from pyFBS.VPT import VPT

import pandas as pd
import scipy as sp
import numpy as np
from scipy import spatial
from scipy.linalg import block_diag
import pickle
from os import path

class MK_model(object):
    """
    Initialization of the finite element model. Mass and stiffness matrices are imported and also nodes, DoFs and complete mesh of finite elements are defined.
    If parameter ``recalculate`` is ``Ture`` eigenvalues and eigenvectors are calculated. 
    For faster processing by default pickle file is generated where mass and stiffness matrices are stored and also computed eigenvalues, eigenvectors and used number of modes.
    If changes are detected in the mass or stiffness matrix with respect to the stored pickle file, the calculation of eigenvalues and eigenvectors is repeated.
    
    :param rst_file: path of the .rst file exported from Ansys
    :type rst_file: str
    :param full_file: path of the .full file exported from Ansys
    :type full_file: str
    :param no_modes: number of modes to be included in output of the eigenvalue computation
    :type no_modes: int
    :param allow_pickle: if ``True``, pickle file will be generated to store data or will pickle file be used to load data
    :type allow_pickle: bool
    :param recalculate: if ``False`` just mass and stiffness matrices with corresponding nodes and their DoFs will be imported. If ``True`` also the eigenvalue problem will be solved.
    :type recalculate: bool
    :param scale: distance scaling factor
    :type scale: float
    :param read_rst: if ``True`` reads the eigenvalue solution directly from .rst file
    :type read_rst: bool
    """

    def __init__(self, rst_file, full_file, no_modes = 100, allow_pickle = True, recalculate = False,scale = 1000,read_rst = False):
        rst = pyansys.read_binary(rst_file)

        # new version of pyansys
        self.nodes = rst.mesh.nodes*scale  # only translational dofs
        self.mesh = rst.grid
        self.mesh.points *= scale
        self.pts = self.mesh.points.copy()

        self.no_modes = no_modes

        self._all = False

        full = pyansys.read_binary(full_file)
        self.dof_ref, self.K, self.M = full.load_km(sort=True)  # dof_ref: 0-x 1-y 2-z
        if self.dof_ref[0, 0] != 1:
            self.dof_ref[:, 0] = self.dof_ref[:, 0] - (self.dof_ref[0, 0] - 1)

        if np.max(self.dof_ref[:, 1]) == 5:
            self.rotation_included = True
        elif np.max(self.dof_ref[:, 1]) == 2:
            self.rotation_included = False

        # an option to read directly the .rst file
        if read_rst == False:
            #print("evaluating M and K matrices")
            self._K = self.K + diags(np.random.random(self.K.shape[0]) / 1e20, shape=self.K.shape) # avoid error

            self.M += sp.sparse.triu(self.M, 1).T
            self._K += sp.sparse.triu(self._K, 1).T

            p_file = '{}.pkl'.format(full_file)
            # check if there is a .pkl file
            same = False
            if allow_pickle and path.exists(p_file):
                # load the pickle file
                _M,_K,_eig_freq,_eig_val,_eig_vec,_no_modes = pickle.load( open(p_file, "rb" ))
                # check if the solution is the same
                if _K.shape == self.K.shape and _M.shape == self.M.shape:
                    check_mas  = (_K != self.K).nnz == 0
                    check_stif = (_M != self.M).nnz == 0
                else:
                    check_mas = False
                    check_stif = False
                check_no_modes = _no_modes == no_modes
                same = np.all([check_mas,check_stif,check_no_modes])
                if same:
                    self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes = pickle.load(open(p_file, "rb"))
                # solve the problem
            if same == False or recalculate == True:
                self.eig_freq, self.eig_val, self.eig_vec = self.eig_solve(self.M, self._K, no_modes)

                if allow_pickle:
                    pickle.dump([self.M, self.K, self.eig_freq, self.eig_val, self.eig_vec, no_modes],open(p_file, "wb"))
        else:
            # read from pyansys - from rst file
            #print("Reading RST file")
            self.eig_freq, self.eig_val, self.eig_vec = self.get_values_from_rst(rst)


    @staticmethod
    def get_values_from_rst(rst):
        """
        Return eigenvalues and eigenvectors for a given rst file.

        :param rst: rst file
        :rtype: (array(float), array(float), array(float))
        """
        eigen_freq = rst.time_values
        eigen_val = eigen_freq**2
        eigen_vec = []
        for i in range(len(rst.time_values)):
            nnum, disp = rst.nodal_solution(i)
            eigen_vec.append(disp.flatten())

        eigen_vec = np.asarray(eigen_vec).T

        return (eigen_freq, eigen_val, eigen_vec)


    @staticmethod
    def eig_solve(mass_mat, stiff_mat, no_modes):
        """
        Find eigenvalues and eigenvectors for given mass matrix ``mass_mat`` and stiffness matrix ``stiff_mat``.

        :param mass_mat: mass matrix
        :type mass_mat: scipy.sparse
        :param stiff_mat: stiffness matrix
        :type stiff_mat: scipy.sparse
        :param no_modes: number of considered modes
        :type no_modes: int
        :return:
        :rtype: (array(float), array(float), array(float))
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
        :type dense_mesh_points: array(float)
        :param sparse_mesh_points: nodal coordinates of sparse mesh in 3D space
        :type sparse_mesh_points: array(float)
        :param dense_mesh_node_id: nodal coordinates id of sparse mesh
        :type dense_mesh_node_id: array(int)
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
        Returns unique locations of all nodal coordinates in ``df`` and all directions for each node.

        :param df: data frame of locations and corresponding directions 
        :type df: pandas.DataFrame
        :return: unique nodal coordinates and directions for each node
        :rtype: (array(float), array(int))
        """
        nodes = df[["Position_1", "Position_2", "Position_3"]].values.astype(float)
        directions = df[["Direction_1", "Direction_2", "Direction_3"]].values.astype(float)

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
        :param all_at_once: Compute response at all locations - ODS animation of the whole mesh,
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

    def update_locations_df(self,df,scale = 1):
        """
        Update locations in data frame ``df`` to nearest nodal locations of the finite element model.
        Directions remain the same.

        :param df: data frame of locations, for which the nearest locations in the numerical model will be found.
        :type df: pandas.DataFrame
        :return: updated data frame
        :rtype: pandas.DataFrame
        """
        _df = df.copy(deep = True)
        _loc = _df[["Position_1", "Position_2", "Position_3"]].to_numpy()*scale
        _index = self.find_nearest_locations(self.nodes,_loc)
        for i,_ind in enumerate(_index):
            _df.loc[i, ["Position_1", "Position_2", "Position_3"]] = self.nodes[_ind]

        return _df

    def get_modeshape(self,select_mode):
        """
        Return desired mode shape. 

        :param select_mode: order of mode shape, starting from 0
        :type select_mode: int
        :return: selected modes shape
        :rtype: array(float)
        """
        _modeshape = np.zeros_like(self.nodes)
        _mode = self.eig_vec[:, select_mode]
        _dof_ref = self.dof_ref
        if self.rotation_included: # to skip rotational modeshape
            _mode = np.asarray([val for m, val in enumerate(_mode) if m % (3 * 2) < 3])
            _dof_ref = np.asarray([val for m, val in enumerate(_dof_ref) if m % (3 * 2) < 3])
        for ref, mode in zip(_dof_ref, _mode):
            _modeshape[ref[0] - 1, ref[1]] = mode

        return _modeshape


    def FRF_synth(self,df_channel,df_impact,f_start = 1, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance",_all = False):
        """
        Synthetisation of frequency response functions using the mode superposition method.

        :param df_channel: locations and directions of responses where FRFs will be generated
        :type df_channel: pandas.DataFrame
        :param df_impact: locations and directions of impacts where FRFs will be generated
        :type df_impact: pandas.DataFrame
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for FRF synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated FRF type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        :param _all: synthetize response at all nodes - can be usefull ot animate FRFs
        :type _all, optional: boolean
        """
        unique_nodes_chn, direction_nodes_chn = self.data_preparation(df_channel)
        unique_nodes_imp, direction_nodes_imp = self.data_preparation(df_impact)

        index_chn = self.find_nearest_locations(self.nodes, unique_nodes_chn)
        index_imp = self.find_nearest_locations(self.nodes, unique_nodes_imp)
            
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
        else:
            damping = modal_damping

        loc1, loc2 = self.loc_definition(response_points, response_directions, excitation_points, excitation_directions, 
                                         self.rotation_included, all_at_once=True)

        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start+1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)
        
        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * _freq
        ome2 = ome ** 2
        _eig_val2 = self.eig_freq ** 2

        if _all:
            m_p_chan_all = self.eig_vec[:, :no_modes]
            m_p_chan_sensors = block_diag(*direction_nodes_chn) @ self.eig_vec[loc1, :no_modes]
            m_p_chan = np.vstack([m_p_chan_sensors,m_p_chan_all])

        else:
            m_p_chan = block_diag(*direction_nodes_chn) @ self.eig_vec[loc1, :no_modes]

        m_p_imp = block_diag(*direction_nodes_imp) @ self.eig_vec[loc2, :no_modes]
        m_p = np.einsum('ij,kj->jik', m_p_chan, m_p_imp)
        
        denominator = (_eig_val2[:no_modes, np.newaxis] - ome2) + np.einsum('ij,i->ij',(ome * self.eig_freq[:no_modes, np.newaxis]),(2 * 1j * damping[:no_modes]))

        FRF_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = FRF_matrix

        elif frf_type == "mobility":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, (1j*2*np.pi*_freq))

        elif frf_type == "accelerance":
            _temp = np.einsum('ijk,i->ijk', FRF_matrix, -(2*np.pi*_freq)**2)

        self.FRF = _temp
        self.freq = freq


    def full_DoF_FRF_synth(self, df_imp, df_sen,f_start = 1, f_end = 2000, f_resolution= 1, limit_modes = None, modal_damping = None, frf_type = "receptance", _all = False):
    
        imp_coord = np.asarray([df_imp['Position_1'], df_imp['Position_2'], df_imp['Position_3']]).T
        sen_coord = np.asarray([df_sen['Position_1'], df_sen['Position_2'], df_sen['Position_3']]).T
        
        # finding three nearest nodes
        ind_imp = np.zeros_like(imp_coord,dtype=int)
        nodes_copy = np.copy(self.nodes)

        for i in range(ind_imp.shape[1]):
            ind_imp[:,i] = self.find_nearest_locations(nodes_copy, imp_coord)
            nodes_copy[ind_imp[:,i]] = 0 # change nearest node to 0 not to be selected in next loops
        ind_sen = np.zeros_like(sen_coord,dtype=int)

        nodes_copy = np.copy(self.nodes)
        for j in range(ind_sen.shape[1]):
            ind_sen[:,j] = self.find_nearest_locations(nodes_copy, sen_coord)
            nodes_copy[ind_sen[:,j]] = 0 # change nearest node to 0 not to be selected in next loops
            
        #generating data frame for impacts
        df_imp_ = np.zeros((int(3*3*ind_imp.shape[0]),3)) # assume nine nearest impacts for VPT
        for k in range(self.nodes[ind_imp].shape[0]):
            df_imp_[9*k:9+9*k,:] = np.repeat(np.asarray([[self.nodes[ind_imp][k,:]]][0]),3,axis=1) # assume nine nearest impacts for VPT
        df_imp = pd.DataFrame(data=df_imp_,columns=('Position_1','Position_2','Position_3'))
        df_imp['Direction_1'] = np.tile([1,0,0,1,0,0,1,0,0],self.nodes[ind_imp].shape[0])
        df_imp['Direction_2'] = np.tile([0,1,0,0,1,0,0,1,0],self.nodes[ind_imp].shape[0])
        df_imp['Direction_3'] = np.tile([0,0,1,0,0,1,0,0,1],self.nodes[ind_imp].shape[0])
        df_imp['Grouping'] = np.repeat([np.arange(ind_imp.shape[0])], 9)
        df_imp['Quantity'] = np.tile(np.repeat(['Acceleration'], 9), ind_imp.shape[0])
            
        #generating data frame for channels
        df_chn_ = np.zeros((int(3*3*ind_sen.shape[0]),3)) # assume three nearest sensors (9 channels) for VPT
        for l in range(self.nodes[ind_sen].shape[0]):
            df_chn_[9*l:9+9*l,:] = np.repeat(np.asarray([[self.nodes[ind_sen][l,:]]][0]),3,axis=1) # assume three nearest sensors for VPT
        df_chn = pd.DataFrame(data=df_chn_,columns=('Position_1','Position_2','Position_3'))
        df_chn['Direction_1'] = np.tile([1,0,0,1,0,0,1,0,0],self.nodes[ind_sen].shape[0])
        df_chn['Direction_2'] = np.tile([0,1,0,0,1,0,0,1,0],self.nodes[ind_sen].shape[0])
        df_chn['Direction_3'] = np.tile([0,0,1,0,0,1,0,0,1],self.nodes[ind_sen].shape[0])
        df_chn['Grouping'] = np.repeat([np.arange(ind_sen.shape[0])], 9)
        df_chn['Quantity'] = np.tile(np.repeat(['Acceleration'], 9), ind_sen.shape[0])
        
        # generating FRF
        self.FRF_synth(df_chn,df_imp,f_start, f_end, f_resolution, limit_modes, modal_damping, frf_type,_all)
            
        # generating data frame for impact virtual points
        df_vp_imp_ = np.zeros((int(6*imp_coord.shape[0]),3))
        for ii in range(imp_coord.shape[0]):
            df_vp_imp_[6*ii:6+6*ii,:] = np.asarray([imp_coord[ii,:]]*6)
        df_vp_imp = pd.DataFrame(data=df_vp_imp_,columns=('Position_1','Position_2','Position_3'))
        df_vp_imp['Direction_1'] = np.tile([1,0,0,1,0,0],imp_coord.shape[0])
        df_vp_imp['Direction_2'] = np.tile([0,1,0,0,1,0],imp_coord.shape[0])
        df_vp_imp['Direction_3'] = np.tile([0,0,1,0,0,1],imp_coord.shape[0])
        df_vp_imp['Quantity'] = np.tile(np.repeat(['Acceleration', 'Rotational Acceleration'], 3), imp_coord.shape[0])
        #df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], imp_coord.shape[0])
        df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)

        df_vp_imp['Description'] = np.tile(['fx','fy','fz','mx','my','mz'],imp_coord.shape[0])
        
        # generating data frame for channel virtual points
        df_vp_chn_ = np.zeros((int(6*sen_coord.shape[0]),3))
        for jj in range(sen_coord.shape[0]):
            df_vp_chn_[6*jj:6+6*jj,:] = np.asarray([sen_coord[jj,:]]*6)
        df_vp_chn = pd.DataFrame(data=df_vp_chn_,columns=('Position_1','Position_2','Position_3'))
        df_vp_chn['Direction_1'] = np.tile([1,0,0,1,0,0],sen_coord.shape[0])
        df_vp_chn['Direction_2'] = np.tile([0,1,0,0,1,0],sen_coord.shape[0])
        df_vp_chn['Direction_3'] = np.tile([0,0,1,0,0,1],sen_coord.shape[0])
        df_vp_chn['Quantity'] = np.tile(np.repeat(['Acceleration', 'Rotational Acceleration'], 3), sen_coord.shape[0])
        #df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], sen_coord.shape[0])
        df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)
        df_vp_chn['Description'] = np.tile(['ux','uy','uz','tx','ty','tz'],sen_coord.shape[0])
        
        # empty array
        FRF_FDoF = np.zeros((self.FRF.shape[0],ind_sen.shape[0]*6,ind_imp.shape[0]*6),dtype=complex)
        
        # apply VPT
        for res_ in df_chn['Grouping'].unique():
            for exc_ in df_imp['Grouping'].unique():
                # Read impacts and VP impacts
                _df_imp = df_imp[df_imp['Grouping'] == exc_]
                _df_vp_imp = df_vp_imp[df_vp_imp['Grouping'] == exc_]
                # Set impacts Group to match responses Group
                #print("a", _df_imp['Grouping'], _df_vp_imp['Grouping'])
                #_df_imp['Grouping'] = res_
                #_df_vp_imp['Grouping'] = res_
                #print("b",_df_imp['Grouping'], _df_vp_imp['Grouping'])

                # Read responses and VP responses
                _df_chn = df_chn[df_chn['Grouping'] == res_]
                _df_vp_chn = df_vp_chn[df_vp_chn['Grouping'] == res_]
                vpt_ = VPT(_df_chn, _df_imp, _df_vp_chn, _df_vp_imp)
                vpt_.apply_VPT(self.freq, self.FRF[:,9*res_:9*res_+9,9*exc_:9*exc_+9])
                FRF_FDoF[:,6*res_:6*res_+6,6*exc_:6*exc_+6] = vpt_.vptData            
        
        return FRF_FDoF


    def add_noise(self,n1 = 2e-2, n2 = 2e-1, n3 = 2e-1 ,n4 = 5e-2):
        """
        Additive noise to synthesized FRFs by random values as per standard normal distribution with defined scaling factors.

        :param n1: amplitude of real part shift scalied with FRF absolute amplitude
        :type n1: float
        :param n2: amplitude of imag part shift scalied with FRF absolute amplitude
        :type n2: float
        :param n3: amplitude of real part shift
        :type n3: float
        :param n4: amplitude of real part shift
        :type n4: float
        """
        rand1 = n1 * np.random.randn(*self.FRF.shape)
        rand2 = n2 * np.random.randn(*self.FRF.shape) * 1j
        rand3 = n3 * np.random.randn(*self.FRF.shape)
        rand4 = n4 * np.random.randn(*self.FRF.shape) * 1j

        noise = np.einsum("ijk,ijk->ijk", np.abs(self.FRF), rand1) + np.einsum("ijk,ijk->ijk", np.abs(self.FRF), rand2) + rand3 + rand4

        self.FRF_noise = self.FRF + noise
