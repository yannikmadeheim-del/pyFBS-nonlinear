from scipy.sparse import diags
from ansys.mapdl import reader as pymapdl_reader
from ansys.dpf import core as dpf
from ansys.dpf import post
from ansys.dpf.core import vtk_helper
from ..interface import VPT

import json
import pandas as pd
import pyvista as pv
import scipy as sp
import numpy as np
from scipy.linalg import block_diag
import pickle
import os
from os import path
import h5py
import warnings


class Model:
    def __init__(
        self,
        nodes=None,
        pts=None,
        mesh=None,
        dof_ref=None,
        k=None,
        _k=None,
        m=None,
        eig_val=None,
        angular_eig_freq=None,
        eig_vec=None,
        eig_vec_strain=None,
        no_modes=None,
        rotation_included=False,
        damped_solver=False,
        damped_modes=None,
        scale=1,
        units=None,
        _all=False,
    ):
        """
        Initialization of the Model.
        """
        self.nodes = nodes
        self.pts = pts
        self.mesh = mesh
        self.dof_ref = dof_ref
        self.k = k
        self._k = _k
        self.m = m
        self.eig_val = eig_val
        self.angular_eig_freq = angular_eig_freq
        self.eig_vec = eig_vec
        self.eig_vec_strain = eig_vec_strain
        self.no_modes = no_modes
        self.rotation_included = rotation_included
        self.damped_solver = damped_solver
        self.damped_modes = damped_modes
        self.scale = scale
        self.units = units
        self._all = _all

        if not angular_eig_freq is None:
            self.eig_freq = angular_eig_freq / (2 * np.pi)
        else:
            self.eig_freq = None

    @classmethod
    def from_save(cls, directory='./', file_name='file'):
        """
        Load the model from .vtk and .hdf5 files.
        """
        vtk_file = os.path.join(directory, f"{file_name}.vtk")
        hdf5_file = os.path.join(directory, f"{file_name}.hdf5")
        json_file = os.path.join(directory, f"{file_name}.json")
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                units = json.load(f)
        else:
            units = None

        # Load the mesh from the .vtk file
        mesh = pv.read(vtk_file)

        # Load the attributes from the .hdf5 file
        with h5py.File(hdf5_file, 'r') as f:
            data = {key: f[key][()] for key in f.keys()}

        return cls(mesh=mesh, units=units, **data)

    @classmethod
    def from_ansys(
        cls,
        rst_file=None,
        full_file=None,
        manual_mass_matrix=None,
        manual_stiffness_matrix=None,
        no_modes=100,
        allow_pickle=True,
        recalculate=False,
        scale=1,
        read_rst=False,
    ):
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

        nodes = None
        pts = None
        mesh = None
        dof_ref = None
        k = None
        _k = None
        m = None
        eig_val = None
        angular_eig_freq = None
        eig_vec = None
        eig_vec_strain = None
        rotation_included = False
        damped_solver = False
        damped_modes = None
        _all = False
        units = None

        if (
            rst_file and full_file
        ):  # check if rest and full files are defined, that mass and stiffness matrices will be importd from there

            rst = pymapdl_reader.read_binary(rst_file)

            # new version of pyansys
            nodes = rst.mesh.nodes * scale  # only translational dofs
            mesh = rst.grid
            mesh.points *= scale
            pts = mesh.points.copy()
            _all = False

            full = pymapdl_reader.read_binary(full_file)
            dof_ref, k_triu, m_triu = full.load_km(
                sort=True
            )  # dof_ref: 0-x 1-y 2-z
            m = m_triu + sp.sparse.triu(m_triu, 1).T
            k = k_triu + sp.sparse.triu(k_triu, 1).T
            _k = k + diags(
                np.random.random(k.shape[0]) / 1e20, shape=k.shape
            )  # avoid error

            if dof_ref[0, 0] != 1:
                dof_ref[:, 0] = dof_ref[:, 0] - (dof_ref[0, 0] - 1)

            if no_modes > len(dof_ref):
                no_modes = len(dof_ref)

            if np.max(dof_ref[:, 1]) == 5:
                rotation_included = True
            elif np.max(dof_ref[:, 1]) == 2:
                rotation_included = False

            # an option to read directly the .rst file
            if read_rst == False:
                # print("evaluating M and K matrices")
                p_file = '{}.pkl'.format(full_file)
                # check if there is a .pkl file
                same = False
                if allow_pickle and path.exists(p_file):

                    same = cls.pickle_check(p_file, no_modes, m, k)
                    if same:
                        m, k, angular_eig_freq, eig_val, eig_vec, no_modes = (
                            pickle.load(open(p_file, "rb"))
                        )
                    # solve the problem
                else:
                    angular_eig_freq, eig_val, eig_vec = cls.eig_solve(
                        m, _k, no_modes
                    )

                if same == False or recalculate == True:
                    angular_eig_freq, eig_val, eig_vec = cls.eig_solve(
                        m, _k, no_modes
                    )

                    if allow_pickle:
                        pickle.dump(
                            [
                                m,
                                k,
                                angular_eig_freq,
                                eig_val,
                                eig_vec,
                                no_modes,
                            ],
                            open(p_file, "wb"),
                        )
            else:  # read from pyansys - from rst file
                # print("Reading RST file")

                m += sp.sparse.triu(m, 1).T
                k += sp.sparse.triu(k, 1).T

                _k = k + diags(
                    np.random.random(k.shape[0]) / 1e20, shape=k.shape
                )  # avoid error

                angular_eig_freq, eig_val, eig_vec, eig_vec_strain = (
                    cls.get_values_from_rst(rst)
                )
                if (
                    len(angular_eig_freq) >= no_modes
                ):  # truncation of results in .rst file to match the desired number of modes in ``no_modes`` parameter
                    angular_eig_freq = angular_eig_freq[:no_modes]
                    eig_val = eig_val[:no_modes]
                    eig_vec = eig_vec[:, :no_modes]
                    try:
                        eig_vec_strain = eig_vec_strain[:, :no_modes]
                    except:  # if the strain is not included in the .rst file, then the ``self.eig_vec_strain`` is just left as an empty array
                        pass
                else:
                    _warn_changed_no_modes_rst(no_modes, len(angular_eig_freq))
                    no_modes = len(angular_eig_freq)

        elif rst_file is not None and full_file is None:
            # if only the .rst file is defined, then the mass and stiffness matrices are not available
            # the solution to the eigenvalue problem is read from the .rst file
            model = dpf.Model(rst_file)
            simulation = post.load_simulation(rst_file)
            units = simulation.units
            displacement = simulation.displacement(all_sets=True, norm=False)
            if 'complex' in displacement.columns.names:
                damped_solver = True
            else:
                damped_solver = False

            nodes = simulation.mesh.coordinates.array
            eig_vec = []
            sort_ix = np.argsort(displacement.axes[0].node_ids.values)
            if damped_solver:
                nat_freq_imag = (
                    simulation.time_freq_support.time_frequencies.data
                )
                nat_freq_real = (
                    simulation.time_freq_support.complex_frequencies.data
                )
                nat_freq = (
                    (nat_freq_real + 1.0j * nat_freq_imag) * 2 * np.pi
                )  # from Hz to rad/s
                angular_eig_freq = nat_freq
                eig_val = nat_freq
                angular_eig_freq_undamped = np.abs(nat_freq)
                damping_ratio = -nat_freq.real / angular_eig_freq_undamped
                damped_modes = (damping_ratio > 1e-5) & (
                    damping_ratio < 0.999999
                )
                for i in range(1, len(nat_freq) + 1):
                    _eig_vec_real = displacement.select(
                        set_ids=i, complex=0
                    ).array[sort_ix]
                    _eig_vec_imag = displacement.select(
                        set_ids=i, complex=1
                    ).array[sort_ix]
                    eig_vec.append(_eig_vec_real + 1.0j * _eig_vec_imag)
                eig_vec = np.asarray(eig_vec)
                _eig_vec = eig_vec.reshape(eig_vec.shape[0], -1).T
                a_normalized_eig_vec = _eig_vec[:, damped_modes] * (
                    np.e ** (-1.0j * np.pi / 4)
                    / np.sqrt(2 * eig_val[damped_modes].imag)
                )
                _eig_vec[:, damped_modes] = a_normalized_eig_vec
            else:
                nat_freq = (
                    simulation.time_freq_support.time_frequencies.data
                    * 2
                    * np.pi
                )  # from Hz to rad/s
                angular_eig_freq = nat_freq
                eig_val = nat_freq**2
                for i in range(1, len(nat_freq) + 1):
                    eig_vec.append(
                        displacement.select(set_ids=i).array[sort_ix]
                    )
                eig_vec = np.asarray(eig_vec)
                _eig_vec = eig_vec.reshape(eig_vec.shape[0], -1).T
            _dof_ref = np.zeros(
                (int(nodes.shape[0] * nodes.shape[1]), 2), dtype=int
            )
            _dof_ref[:, 0] = np.repeat(np.arange(1, nodes.shape[0] + 1), 3)
            _dof_ref[1::3, 1] = 1
            _dof_ref[2::3, 1] = 2

            nodes = nodes * scale
            mesh = vtk_helper.dpf_mesh_to_vtk_op(model.metadata.meshed_region)
            mesh.points *= scale
            pts = mesh.points.copy()
            eig_vec = _eig_vec
            dof_ref = _dof_ref
            rotation_included = False
            if no_modes > len(dof_ref):
                no_modes = len(dof_ref)
            if no_modes > len(angular_eig_freq):
                _warn_changed_no_modes_rst(no_modes, len(angular_eig_freq))
                no_modes = len(angular_eig_freq)

        elif (manual_mass_matrix is not None) and (
            manual_stiffness_matrix is not None
        ):  # if mass and stiffness matrices are manually defined
            k, m = manual_stiffness_matrix, manual_mass_matrix
            _k = k + diags(
                np.random.random(k.shape[0]) / 1e20, shape=k.shape
            )  # avoid error
            if no_modes > len(k):
                no_modes = len(k)
            else:
                no_modes = no_modes
            _all = False
            rotation_included = False

            p_file = '{}.pkl'.format("mass_stiffness_matrices")
            same = False
            if allow_pickle and path.exists(p_file):
                same = cls.pickle_check(p_file, no_modes, m, k)
                if same:
                    m, k, angular_eig_freq, eig_val, eig_vec, no_modes = (
                        pickle.load(open(p_file, "rb"))
                    )
                # solve the problem
                if same == False or recalculate == True:
                    angular_eig_freq, eig_val, eig_vec = cls.eig_solve(
                        m, _k, no_modes
                    )

                    if allow_pickle:
                        pickle.dump(
                            [
                                m,
                                k,
                                angular_eig_freq,
                                eig_val,
                                eig_vec,
                                no_modes,
                            ],
                            open(p_file, "wb"),
                        )
            else:
                angular_eig_freq, eig_val, eig_vec = cls.eig_solve(
                    m, _k, no_modes
                )

        return cls(
            nodes=nodes,
            pts=pts,
            mesh=mesh,
            dof_ref=dof_ref,
            k=k,
            _k=_k,
            m=m,
            eig_val=eig_val,
            angular_eig_freq=angular_eig_freq,
            eig_vec=eig_vec,
            eig_vec_strain=eig_vec_strain,
            no_modes=no_modes,
            rotation_included=rotation_included,
            damped_solver=damped_solver,
            damped_modes=damped_modes,
            scale=scale,
            units=units,
            _all=_all,
        )

    @property
    def eig_freq(self):
        """
        Getter for eig_freq property, which is the eigenfrequency in Hz.

        :return: eigenfrequency in Hz
        """
        return self._eig_freq

    @property
    def angular_eig_freq(self):
        """
        Getter for eig_freq property, which is the eigenfrequency in rad/s.

        :return: eigenfrequency in Hz
        """
        return self._angular_eig_freq

    @eig_freq.setter
    def eig_freq(self, _eig_freq):
        self._eig_freq = _eig_freq
        if _eig_freq is None:
            self._angular_eig_freq = None
        else:
            self._angular_eig_freq = _eig_freq * (2 * np.pi)

    @angular_eig_freq.setter
    def angular_eig_freq(self, _angular_eig_freq):
        self._angular_eig_freq = _angular_eig_freq
        if _angular_eig_freq is None:
            self._eig_freq = None
        else:
            self._eig_freq = _angular_eig_freq / (2 * np.pi)

    def save(self, directory='./', file_name='file'):
        """
        Save the model mesh to .vtk and other attributes to .hdf5.

        :param directory: directory where the file will be saved
        :type directory: str
        :param file_name: name of the file
        :type file_name: str
        """
        hdf5_filepath = os.path.join(directory, f"{file_name}.hdf5")
        hdf5_file = h5py.File(hdf5_filepath, "w")
        hdf5_file.create_dataset("nodes", data=self.nodes)
        hdf5_file.create_dataset("pts", data=self.pts)
        hdf5_file.create_dataset("eig_vec", data=self.eig_vec)
        hdf5_file.create_dataset("dof_ref", data=self.dof_ref)
        hdf5_file.create_dataset(
            "angular_eig_freq", data=self.angular_eig_freq
        )
        hdf5_file.create_dataset("eig_val", data=self.eig_val)
        hdf5_file.create_dataset("no_modes", data=self.no_modes)
        hdf5_file.create_dataset(
            "rotation_included", data=self.rotation_included
        )
        hdf5_file.create_dataset("scale", data=self.scale)
        if hasattr(self, 'eig_vec_strain'):
            if self.eig_vec_strain is not None:
                hdf5_file.create_dataset(
                    "eig_vec_strain", data=self.eig_vec_strain
                )
        if hasattr(self, 'damped_solver'):
            hdf5_file.create_dataset("damped_solver", data=self.damped_solver)
            if self.damped_solver:
                hdf5_file.create_dataset(
                    "damped_modes", data=self.damped_modes
                )
        hdf5_file.close()

        # save mesh to vtk
        vtk_filepath = os.path.join(directory, f"{file_name}.vtk")
        grid = self.mesh.copy()
        grid.save(vtk_filepath)

        if hasattr(self, 'units'):
            if self.units is not None:
                json_filepath = os.path.join(directory, f"{file_name}.json")
                with open(json_filepath, 'w') as f:
                    json.dump(self.units, f)
                print(
                    f"Model saved to {file_name}.hdf5, {file_name}.vtk, and {file_name}.json"
                )
            else:
                print(f"Model saved to {file_name}.hdf5 and {file_name}.vtk")
        else:
            print(f"Model saved to {file_name}.hdf5 and {file_name}.vtk")

    @staticmethod
    def pickle_check(p_file, no_modes, m, k):
        """The function checks if the defined mass and stiffness matrices are the same as were defined in the saved pickle file.

        :param p_file: name of pickle file
        :type p_file: array
        :param no_modes: number of modes to be included in output of the eigenvalue computation
        :type no_modes: int
        :param m: mass matrix
        :type m: scipy.sparse
        :param k: stiffness matrix
        :type k: scipy.sparse

        :rtype: bool
        """
        _m, _k, _angular_eig_freq, _eig_val, _eig_vec, _no_modes = pickle.load(
            open(p_file, "rb")
        )
        # check if the solution is the same
        if _k.shape == k.shape and _m.shape == m.shape:
            check_mas = (_m != m).nnz == 0
            check_stif = (_k != k).nnz == 0
        else:
            check_mas = False
            check_stif = False
        check_no_modes = _no_modes == no_modes
        same = np.all([check_mas, check_stif, check_no_modes])
        return same

    def manual_mesh_definition(self, grid, dof_ref):
        """
        Definition of mesh and DoFs for manually inputed mass and stiffness matrices.

        :param grid: grid definition in form of pyvista.PolyData
        :type grid: pyvista
        :param dof_ref: definition of DoFs inside ``Model`` in form of 2D
            matrix, dimensions nx2, where n is the dimension of square mass or
            stiffness matrix. The first column represents the index of node
            location, starting with 1, the second column represents direction
            of this DoF: 0-x, 1-y, 2-z.
        :type dof_ref: array
        """
        self.mesh = grid
        self.nodes = grid.points
        self.pts = grid.points.copy()
        self.dof_ref = dof_ref

    @staticmethod
    def get_values_from_rst(rst):
        """
        Return eigenvalues and eigenvectors for a given rst file.

        :param rst: rst file
        :rtype: (array(float), array(float), array(float))
        """
        eigen_freq = rst.time_values * 2 * np.pi  # from Hz to rad/s
        eigen_val = eigen_freq**2
        eigen_vec = []
        eigen_vec_strain = []
        for i in range(len(rst.time_values)):
            nnum, disp = rst.nodal_displacement(i)
            eigen_vec.append(disp.flatten())
            try:
                nnum, strain = rst.nodal_elastic_strain(i)
                eigen_vec_strain.append(strain.flatten())
            except:
                pass

        eigen_vec = np.asarray(eigen_vec).T
        try:
            eigen_vec_strain = np.asarray(eigen_vec_strain).T
        except:
            pass

        return (eigen_freq, eigen_val, eigen_vec, eigen_vec_strain)

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
        try:
            eigen_val, eigen_vec = sp.sparse.linalg.eigsh(
                stiff_mat, k=no_modes, m=mass_mat, sigma=0
            )
        except np.linalg.LinAlgError:
            # sometimes eigenvalue problems can not be solved using sparse configuration, especially for small analytical systems
            eigen_val, eigen_vec = sp.linalg.eig(stiff_mat, mass_mat)

        eigen_val.sort()
        eigen_freq = np.sqrt(np.abs(np.real(eigen_val)))  # /(2*np.pi)
        return (eigen_freq, eigen_val, eigen_vec)

    def find_nearest_locations(self, points, **kwargs):
        """
        This function finds the nearest coordinate locations of defined points
        array in the corresponding Model mesh.

        :param points: nodal coordinates of points in 3D space
        :type points: array(float)
        :return: Selected nodes by index and by id regarding the dense mesh
        :rtype: (array(int), array(int))

        """
        _index = []
        for _loc_i in points:
            _index.append(self.mesh.find_closest_point(_loc_i, **kwargs))
        return np.array(_index)

    @staticmethod
    def data_preparation(df, n_dim=3):
        """
        Returns unique locations of all nodal coordinates in ``df`` and all directions for each node.

        :param df: data frame of locations and corresponding directions
        :type df: pandas.DataFrame
        :param n_dim: number of dimensions in FEM model
        :type n_dim: int
        :return: unique nodal coordinates and directions for each node
        :rtype: (array(float), array(int))
        """
        nodes = df[["Position_1", "Position_2", "Position_3"]].values.astype(
            float
        )
        directions = df[
            ["Direction_1", "Direction_2", "Direction_3"]
        ].values.astype(float)[:, :n_dim]

        unique_nodes = nodes[
            np.sort(np.unique(nodes, axis=0, return_index=True)[1])
        ]
        direction_nodes = []
        for node in unique_nodes:
            loc = np.where((nodes == node).all(axis=1))
            direction_nodes.append(directions[loc])
        return unique_nodes, direction_nodes

    def loc_definition(self, node_index):
        """
        DoF index generation for the node index in the global model.

        :param point_index: response/excitation node index in the global model (starting with 1)
        :type response_point: int or array(int)
        :return: DoF indices corresponding to the input point indices
        :rtype: int
        """
        node_index = np.asarray([node_index]).ravel()
        return np.array(
            [np.argwhere(self.dof_ref[:, 0] == _)[:3] for _ in node_index]
        ).ravel()

    def update_locations_df(self, df, scale=1):
        """
        Update locations in data frame ``df`` to nearest nodal locations of the finite element model.
        Directions remain the same.

        :param df: data frame of locations, for which the nearest locations in the numerical model will be found.
        :type df: pandas.DataFrame
        :return: updated data frame
        :rtype: pandas.DataFrame
        """
        _df = df.copy(deep=True).reset_index(drop=True)
        _loc = (
            _df[["Position_1", "Position_2", "Position_3"]].to_numpy() * scale
        )
        _index = self.find_nearest_locations(_loc)
        for i, _indedex_i in enumerate(_index):
            _df.loc[i, ["Position_1", "Position_2", "Position_3"]] = (
                self.nodes[_indedex_i]
            )
        return _df

    def get_modeshape(self, select_mode):
        """
        Return desired mode shape.

        :param select_mode: order of mode shape, starting from 0
        :type select_mode: int
        :return: selected modes shape
        :rtype: array(float)
        """
        _modeshape = np.zeros_like(self.nodes, dtype=self.eig_vec.dtype)
        _mode = self.eig_vec[:, select_mode]
        _dof_ref = self.dof_ref
        if self.rotation_included:  # to skip rotational modeshape
            _mode = np.asarray(
                [val for m, val in enumerate(_mode) if m % (3 * 2) < 3]
            )
            _dof_ref = np.asarray(
                [val for m, val in enumerate(_dof_ref) if m % (3 * 2) < 3]
            )
        for ref, mode in zip(_dof_ref, _mode):
            _modeshape[ref[0] - 1, ref[1]] = mode

        return _modeshape

    def get_modeshape_strain(self, select_mode, direction="X"):
        """
        Return desired strain mode shape.

        :param select_mode: order of mode shape, starting from 0
        :type select_mode: int
        :return: selected modes shape
        :rtype: array(float)
        """
        STRAIN_DIRECTIONS = ["X", "Y", "Z", "XY", "YZ", "XZ", "EQV"]
        mode_index = STRAIN_DIRECTIONS.index(direction.upper())

        _modeshape = self.eig_vec_strain[
            mode_index :: len(STRAIN_DIRECTIONS), select_mode
        ]

        return _modeshape

    def transform_modal_parameters(
        self,
        df_channel,
        df_impact=None,
        limit_modes=None,
        modal_damping=None,
        _all=False,
        return_channel_only=False,
        n_dim=3,
    ):
        """
        FEM model reduction to the defined input/output locations and directions.

        :param df_channel: locations and directions of responses where frfs will be generated
        :type df_channel: pandas.DataFrame
        :param df_impact: locations and directions of impacts where frfs will be generated
        :type df_impact: pandas.DataFrame
        :param limit_modes: number of modes used for frf synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        """
        # truncation
        if limit_modes == None:
            no_modes = self.no_modes
        else:
            if limit_modes > len(self.nodes):
                no_modes = len(self.nodes)
            else:
                no_modes = limit_modes

        # eigenvalues
        if self.damped_solver:
            _eig_val2 = self.eig_val[:no_modes]
        else:
            _eig_val2 = self.angular_eig_freq[:no_modes] ** 2
        # damping

        if modal_damping is None:
            damping = np.zeros(no_modes)
        elif isinstance(
            modal_damping,
            (int, float, np.int32, np.int64, np.float32, np.float64),
        ):
            damping = np.repeat(modal_damping, no_modes)
        elif isinstance(modal_damping, (list, tuple, np.ndarray)):
            modal_damping = np.asarray(modal_damping).ravel()
            if len(modal_damping) == 1:
                damping = np.repeat(modal_damping, no_modes)
            elif len(modal_damping) == no_modes:
                damping = modal_damping
            else:
                raise Exception('Input for "modal damping" not valid.')
        else:
            raise Exception('Input for "modal damping" not valid.')

        # response DoF
        unique_nodes_chn, direction_nodes_chn = self.data_preparation(
            df_channel, n_dim
        )
        index_chn = self.find_nearest_locations(unique_nodes_chn)
        response_points = index_chn + 1
        loc1 = self.loc_definition(response_points)

        # response eigenvector reduction/transformation
        if _all:
            m_p_chan_all = self.eig_vec[:, :no_modes]
            m_p_chan_sensors = (
                block_diag(*direction_nodes_chn)
                @ self.eig_vec[loc1, :no_modes]
            )
            m_p_chan = np.vstack([m_p_chan_sensors, m_p_chan_all])

        else:
            m_p_chan = (
                block_diag(*direction_nodes_chn)
                @ self.eig_vec[loc1, :no_modes]
            )

        if return_channel_only == True:
            return (_eig_val2, damping, m_p_chan)
        else:
            # excitation DoF
            unique_nodes_imp, direction_nodes_imp = self.data_preparation(
                df_impact, n_dim
            )
            index_imp = self.find_nearest_locations(unique_nodes_imp)
            excitation_points = index_imp + 1
            loc2 = self.loc_definition(excitation_points)
            # excitation eigenvector reduction/transformation
            m_p_imp = (
                block_diag(*direction_nodes_imp)
                @ self.eig_vec[loc2, :no_modes]
            )
            m_p = np.einsum('ij,kj->jik', m_p_chan, m_p_imp)
            return (no_modes, _eig_val2, damping, m_p)

    def frf_synth_full(
        self, f_start=1, f_end=2000, f_resolution=1, frf_type="receptance"
    ):
        """
        Synthetisation of frequency response functions using the full harmonic method.

        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param frf_type: define calculated frf type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        """

        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start + 1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)

        freq = np.arange(f_start, f_end, f_resolution)

        omega = 2 * np.pi * _freq

        k_temp = np.array(self._k[np.newaxis]).repeat(len(_freq), axis=0)
        m_temp = np.array(self.m[np.newaxis]).repeat(len(_freq), axis=0)
        frf_matrix = np.linalg.inv(
            k_temp - np.einsum("i,ijk->ijk", omega**2, m_temp)
        )

        if frf_type == "receptance":
            _temp = frf_matrix

        elif frf_type == "mobility":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, (1j * 2 * np.pi * _freq)
            )

        elif frf_type == "accelerance":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, -((2 * np.pi * _freq) ** 2)
            )

        self.frf = _temp
        self.freq = freq

    def frf_synth(
        self,
        df_channel,
        df_impact,
        f_start=1,
        f_end=2000,
        f_resolution=1,
        limit_modes=None,
        modal_damping=None,
        frf_type="receptance",
        _all=False,
        n_dim=3,
    ):
        """
        Synthetisation of frequency response functions using the mode superposition method.

        :param df_channel: locations and directions of responses where frfs will be generated
        :type df_channel: pandas.DataFrame
        :param df_impact: locations and directions of impacts where frfs will be generated
        :type df_impact: pandas.DataFrame
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for frf synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated frf type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        :param _all: synthetize response at all nodes - can be usefull to animate frfs
        :type _all, optional: boolean
        :param n_dim: number of DoFs per one node in Model (default is 3)
        :type n_dim, optional: boolean
        """

        no_modes, _eig_val2, damping, m_p = self.transform_modal_parameters(
            df_channel=df_channel,
            df_impact=df_impact,
            limit_modes=limit_modes,
            modal_damping=modal_damping,
            _all=_all,
            n_dim=n_dim,
        )

        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start + 1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)

        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * _freq
        ome2 = ome**2

        if self.damped_solver:
            m_p_undamped = m_p[~self.damped_modes[:no_modes]]
            m_p = m_p[self.damped_modes[:no_modes]]
            _eig_val2 = _eig_val2[self.damped_modes[:no_modes]]
            m_p_conj = np.conj(m_p)
            den_1 = 1.0j * ome - _eig_val2[:no_modes, np.newaxis]
            den_2 = 1.0j * ome - _eig_val2[:no_modes, np.newaxis].conj()

            frf_matrix = np.einsum('ijk,il->ljk', m_p, 1 / den_1) + np.einsum(
                'ijk,il->ljk', m_p_conj, 1 / den_2
            )
            if not np.all(self.damped_modes[:no_modes]):
                frf_matrix += np.einsum(
                    'ijk,l->ljk', m_p_undamped, -1 / (ome2)
                )

        else:
            denominator = (
                _eig_val2[:no_modes, np.newaxis] - ome2
            ) + np.einsum(
                'ij,i->ij',
                (ome * self.angular_eig_freq[:no_modes, np.newaxis]),
                (2 * 1j * damping[:no_modes]),
            )

            frf_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = frf_matrix

        elif frf_type == "mobility":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, (1j * 2 * np.pi * _freq)
            )

        elif frf_type == "accelerance":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, -((2 * np.pi * _freq) ** 2)
            )

        self.frf = _temp
        self.freq = freq

    def full_dof_frf_synth(
        self,
        df_imp,
        df_sen,
        f_start=1,
        f_end=2000,
        f_resolution=1,
        limit_modes=None,
        modal_damping=None,
        frf_type="receptance",
        _all=False,
    ):
        """
        Generate frfs on exact location of impacts and sensors by projecting frfs from three closest nodes in numercial model.
        Modal superpostition method is used for frf generation.  Gereated are all 3 translations and three rotations for every DoFs.

        :param df_imp: locations and directions of impacts where frfs will be generated
        :type df_imp: pandas.DataFrame
        :param df_sen: locations and directions of sensors where frfs will be generated
        :type df_sen: pandas.DataFrame
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for frf synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated frf type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        :param _all: synthetize response at all nodes - can be usefull to animate frfs
        :type _all, optional: boolean
        :param n_dim: number of DoFs per one node in Model (default is 3)
        :type n_dim, optional: boolean
        """

        imp_coord = np.asarray(
            [df_imp['Position_1'], df_imp['Position_2'], df_imp['Position_3']]
        ).T
        sen_coord = np.asarray(
            [df_sen['Position_1'], df_sen['Position_2'], df_sen['Position_3']]
        ).T

        # finding three nearest nodes
        ind_imp = self.find_nearest_locations(imp_coord, n=imp_coord.shape[1])
        ind_sen = self.find_nearest_locations(sen_coord, n=sen_coord.shape[1])

        # generating data frame for impacts
        df_imp_ = np.zeros(
            (int(3 * 3 * ind_imp.shape[0]), 3)
        )  # assume nine nearest impacts for VPT
        for k in range(self.nodes[ind_imp].shape[0]):
            df_imp_[9 * k : 9 + 9 * k, :] = np.repeat(
                np.asarray([[self.nodes[ind_imp][k, :]]][0]), 3, axis=1
            )  # assume nine nearest impacts for VPT
        df_imp = pd.DataFrame(
            data=df_imp_, columns=('Position_1', 'Position_2', 'Position_3')
        )
        df_imp['Direction_1'] = np.tile(
            [1, 0, 0, 1, 0, 0, 1, 0, 0], self.nodes[ind_imp].shape[0]
        )
        df_imp['Direction_2'] = np.tile(
            [0, 1, 0, 0, 1, 0, 0, 1, 0], self.nodes[ind_imp].shape[0]
        )
        df_imp['Direction_3'] = np.tile(
            [0, 0, 1, 0, 0, 1, 0, 0, 1], self.nodes[ind_imp].shape[0]
        )
        df_imp['Grouping'] = np.repeat([np.arange(ind_imp.shape[0])], 9)
        df_imp['Quantity'] = np.tile(
            np.repeat(['Acceleration'], 9), ind_imp.shape[0]
        )

        # generating data frame for channels
        df_chn_ = np.zeros(
            (int(3 * 3 * ind_sen.shape[0]), 3)
        )  # assume three nearest sensors (9 channels) for VPT
        for l in range(self.nodes[ind_sen].shape[0]):
            df_chn_[9 * l : 9 + 9 * l, :] = np.repeat(
                np.asarray([[self.nodes[ind_sen][l, :]]][0]), 3, axis=1
            )  # assume three nearest sensors for VPT
        df_chn = pd.DataFrame(
            data=df_chn_, columns=('Position_1', 'Position_2', 'Position_3')
        )
        df_chn['Direction_1'] = np.tile(
            [1, 0, 0, 1, 0, 0, 1, 0, 0], self.nodes[ind_sen].shape[0]
        )
        df_chn['Direction_2'] = np.tile(
            [0, 1, 0, 0, 1, 0, 0, 1, 0], self.nodes[ind_sen].shape[0]
        )
        df_chn['Direction_3'] = np.tile(
            [0, 0, 1, 0, 0, 1, 0, 0, 1], self.nodes[ind_sen].shape[0]
        )
        df_chn['Grouping'] = np.repeat([np.arange(ind_sen.shape[0])], 9)
        df_chn['Quantity'] = np.tile(
            np.repeat(['Acceleration'], 9), ind_sen.shape[0]
        )

        # generating frf
        self.frf_synth(
            df_chn,
            df_imp,
            f_start,
            f_end,
            f_resolution,
            limit_modes,
            modal_damping,
            frf_type,
            _all,
        )

        # generating data frame for impact virtual points
        df_vp_imp_ = np.zeros((int(6 * imp_coord.shape[0]), 3))
        for ii in range(imp_coord.shape[0]):
            df_vp_imp_[6 * ii : 6 + 6 * ii, :] = np.asarray(
                [imp_coord[ii, :]] * 6
            )
        df_vp_imp = pd.DataFrame(
            data=df_vp_imp_, columns=('Position_1', 'Position_2', 'Position_3')
        )
        df_vp_imp['Direction_1'] = np.tile(
            [1, 0, 0, 1, 0, 0], imp_coord.shape[0]
        )
        df_vp_imp['Direction_2'] = np.tile(
            [0, 1, 0, 0, 1, 0], imp_coord.shape[0]
        )
        df_vp_imp['Direction_3'] = np.tile(
            [0, 0, 1, 0, 0, 1], imp_coord.shape[0]
        )
        df_vp_imp['Quantity'] = np.tile(
            np.repeat(['Acceleration', 'Rotational Acceleration'], 3),
            imp_coord.shape[0],
        )
        # df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], imp_coord.shape[0])
        df_vp_imp['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)

        df_vp_imp['Description'] = np.tile(
            ['fx', 'fy', 'fz', 'mx', 'my', 'mz'], imp_coord.shape[0]
        )

        # generating data frame for channel virtual points
        df_vp_chn_ = np.zeros((int(6 * sen_coord.shape[0]), 3))
        for jj in range(sen_coord.shape[0]):
            df_vp_chn_[6 * jj : 6 + 6 * jj, :] = np.asarray(
                [sen_coord[jj, :]] * 6
            )
        df_vp_chn = pd.DataFrame(
            data=df_vp_chn_, columns=('Position_1', 'Position_2', 'Position_3')
        )
        df_vp_chn['Direction_1'] = np.tile(
            [1, 0, 0, 1, 0, 0], sen_coord.shape[0]
        )
        df_vp_chn['Direction_2'] = np.tile(
            [0, 1, 0, 0, 1, 0], sen_coord.shape[0]
        )
        df_vp_chn['Direction_3'] = np.tile(
            [0, 0, 1, 0, 0, 1], sen_coord.shape[0]
        )
        df_vp_chn['Quantity'] = np.tile(
            np.repeat(['Acceleration', 'Rotational Acceleration'], 3),
            sen_coord.shape[0],
        )
        # df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], sen_coord.shape[0])
        df_vp_chn['Grouping'] = np.repeat([np.arange(imp_coord.shape[0])], 6)
        df_vp_chn['Description'] = np.tile(
            ['ux', 'uy', 'uz', 'tx', 'ty', 'tz'], sen_coord.shape[0]
        )

        # empty array
        frf_FDoF = np.zeros(
            (self.frf.shape[0], ind_sen.shape[0] * 6, ind_imp.shape[0] * 6),
            dtype=complex,
        )

        # apply VPT
        for res_ in df_chn['Grouping'].unique():
            for exc_ in df_imp['Grouping'].unique():
                # Read impacts and VP impacts
                _df_imp = df_imp[df_imp['Grouping'] == exc_]
                _df_vp_imp = df_vp_imp[df_vp_imp['Grouping'] == exc_]
                # Set impacts Group to match responses Group
                # print("a", _df_imp['Grouping'], _df_vp_imp['Grouping'])
                # _df_imp['Grouping'] = res_
                # _df_vp_imp['Grouping'] = res_
                # print("b",_df_imp['Grouping'], _df_vp_imp['Grouping'])

                # Read responses and VP responses
                _df_chn = df_chn[df_chn['Grouping'] == res_]
                _df_vp_chn = df_vp_chn[df_vp_chn['Grouping'] == res_]
                vpt_ = VPT(_df_chn, _df_imp, _df_vp_chn, _df_vp_imp)
                vpt_.apply_VPT(
                    self.freq,
                    self.frf[
                        :, 9 * res_ : 9 * res_ + 9, 9 * exc_ : 9 * exc_ + 9
                    ],
                )
                frf_FDoF[
                    :, 6 * res_ : 6 * res_ + 6, 6 * exc_ : 6 * exc_ + 6
                ] = vpt_.vptData

        return frf_FDoF

    @staticmethod
    def custom_frf_synth(
        angular_eig_freq,
        eig_vec_chn,
        eig_vec_imp,
        f_start=1,
        f_end=2000,
        f_resolution=1,
        limit_modes=None,
        modal_damping=None,
        frf_type="receptance",
    ):
        """
        Synthetisation of frequency response functions using the mode superposition method.
        frfs are generated for all combinations of inputed eigen vectors.

        :param angular_eig_freq: eigen frequencies of cinsidered system in unit: rad/s
        :type angular_eig_freq: numpy.array
        :param eig_vec_chn: eigen vectors of channels where frfs will be generated
        :type eig_vec_chn: numpy.array
        :param eig_vec_imp: eigen vectors of impacts where frfs will be generated
        :type eig_vec_imp: numpy.array
        :param f_start: starting point of the frequency range
        :type f_start: int or float
        :param f_end: endpoint of the frequency range
        :type f_end: int or float
        :param f_resolution: resolution of frequency range
        :type f_resolution: int or float
        :param limit_modes: number of modes used for frf synthesis
        :type limit_modes: int
        :param modal_damping: viscose modal damping ratio (constant for whole frequency range or ``None``)
        :type modal_damping: float or None
        :param frf_type: define calculated frf type (``receptance``, ``mobility`` or ``accelerance``)
        :type frf_type: str
        """
        if limit_modes == None:
            no_modes = len(angular_eig_freq)
        else:
            no_modes = limit_modes

        modal_damping = np.asarray(modal_damping).ravel()
        if modal_damping.all() == None:
            damping = np.zeros(no_modes)
        elif len(modal_damping) == 1:
            damping = np.repeat(modal_damping, no_modes)
        elif len(modal_damping) == no_modes:
            damping = modal_damping
        else:
            raise Exception('Input for "modal damping" not valid.')

        if f_start == 0:
            # approximation at 0Hz
            _freq = np.arange(f_start + 1e-3, f_end, f_resolution)
        else:
            _freq = np.arange(f_start, f_end, f_resolution)

        freq = np.arange(f_start, f_end, f_resolution)

        ome = 2 * np.pi * _freq
        ome2 = ome**2
        _eig_val2 = angular_eig_freq**2

        m_p_chn = eig_vec_chn[:, :no_modes]

        m_p_imp = eig_vec_imp[:, :no_modes]

        m_p = np.einsum('ij,kj->jik', m_p_chn, m_p_imp)

        denominator = (_eig_val2[:no_modes, np.newaxis] - ome2) + np.einsum(
            'ij,i->ij',
            (ome * angular_eig_freq[:no_modes, np.newaxis]),
            (2 * 1j * damping[:no_modes]),
        )

        frf_matrix = np.einsum('ijk,il->ljk', m_p, 1 / denominator)

        if frf_type == "receptance":
            _temp = frf_matrix

        elif frf_type == "mobility":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, (1j * 2 * np.pi * _freq)
            )

        elif frf_type == "accelerance":
            _temp = np.einsum(
                'ijk,i->ijk', frf_matrix, -((2 * np.pi * _freq) ** 2)
            )

        frf = _temp

        return freq, frf

    def add_noise(self, n1=2e-2, n2=2e-1, n3=2e-1, n4=5e-2):
        """
        Additive noise to synthesized frfs by random values as per standard normal distribution with defined scaling factors.

        :param n1: amplitude of real part shift scalied with frf absolute amplitude
        :type n1: float
        :param n2: amplitude of imag part shift scalied with frf absolute amplitude
        :type n2: float
        :param n3: amplitude of real part shift
        :type n3: float
        :param n4: amplitude of real part shift
        :type n4: float
        """
        rand1 = n1 * np.random.randn(*self.frf.shape)
        rand2 = n2 * np.random.randn(*self.frf.shape) * 1j
        rand3 = n3 * np.random.randn(*self.frf.shape)
        rand4 = n4 * np.random.randn(*self.frf.shape) * 1j

        noise = (
            np.einsum("ijk,ijk->ijk", np.abs(self.frf), rand1)
            + np.einsum("ijk,ijk->ijk", np.abs(self.frf), rand2)
            + rand3
            + rand4
        )

        self.frf_noise = self.frf + noise

    def extract_surface(self):
        """
        Extract the surface mesh from the current mesh and return a new Model
        instance with the surface mesh.
        :return: A new Model instance with the surface mesh.
        :rtype: Model
        """
        import pyvista as pv

        if isinstance(self.mesh, pv.core.pointset.PolyData):
            print(
                "This mesh is already a PolyData object, nothing to extract."
            )
            return
        elif isinstance(self.mesh, pv.core.pointset.UnstructuredGrid):
            mesh = self.mesh.extract_surface()
            ix = mesh['vtkOriginalPointIds']
            nodes = mesh.points
            pts = mesh.points.copy()

            # Extracting the modes at surface nodes
            # Check if dof_ref indices continuously increase
            c1 = np.all(
                self.dof_ref[::3, 0][1:] - self.dof_ref[::3, 0][:-1] > 0
            )
            c2 = np.all(
                self.dof_ref[1::3, 0][1:] - self.dof_ref[1::3, 0][:-1] > 0
            )
            c3 = np.all(
                self.dof_ref[2::3, 0][1:] - self.dof_ref[2::3, 0][:-1] > 0
            )
            if c1 and c2 and c3:
                _modes = self.eig_vec.copy()
                _modes = _modes.reshape(len(self.nodes), 3, self.no_modes)
                _modes_surface = _modes[ix]
                eig_vec = _modes_surface.reshape(-1, self.no_modes)
            else:
                _modes = np.empty(
                    (self.no_modes, len(self.nodes), 3),
                    dtype=self.eig_vec.dtype,
                )
                print("Extracting modes at surface nodes...")
                for i in range(self.no_modes):
                    _modes[i] = self.get_modeshape(i)
                print("Modes extracted.")
                _modes_surface = _modes[:, ix]
                eig_vec = _modes_surface.reshape(self.no_modes, -1).T
            # Generate new dof_ref of the surface mesh
            ndof = eig_vec.shape[0]
            dof_ref = np.empty((ndof, 2), dtype=int)
            dof_ref[:, 0] = np.repeat(np.arange(1, len(nodes) + 1), 3)
            dof_ref[:, 1] = np.tile(np.arange(3), len(nodes))

            # Extracting the eig_vec_strain if it exists
            if self.eig_vec_strain is None:
                eig_vec_strain = None
            else:
                _eig_vec_strain = self.eig_vec_strain.copy()
                _eig_vec_strain = _eig_vec_strain.reshape(
                    len(nodes), -1, self.no_modes
                )[ix]
                eig_vec_strain = _eig_vec_strain.reshape(-1, self.no_modes)
            return type(self)(
                nodes=nodes,
                pts=pts,
                mesh=mesh,
                dof_ref=dof_ref,
                k=None,
                _k=None,
                m=None,
                eig_val=self.eig_val,
                angular_eig_freq=self.angular_eig_freq,
                eig_vec=eig_vec,
                eig_vec_strain=eig_vec_strain,
                no_modes=self.no_modes,
                rotation_included=False,
                damped_solver=self.damped_solver,
                scale=self.scale,
                units=self.units,
                _all=self._all,
            )
        else:
            raise TypeError(
                f"Mesh type {type(self.mesh)} unsupported. Must \
                be either a PolyData or UnstructuredGrid object."
            )


def _warn_changed_no_modes_rst(no_modes_old, no_modes_new):
    warnings.warn(
        f"Parameter ``no_modes`` is set to {no_modes_old}, but the .rst "
        f"file from Ansys includes {no_modes_new} natural frequencies "
        f"and mode shapes. \n Therefore, ``no_modes`` is changed to "
        f"{no_modes_new}."
    )
