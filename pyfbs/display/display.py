import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import pandas as pd
from time import time, sleep
from PyQt5.QtWidgets import QAction
from PyQt5 import QtGui
import imageio
from ..utility import (
    rotation_matrix_from_vectors,
    angle_between,
    angle,
    rotation_matrix,
)
from ..utility import dict_animation
import keyboard as kb
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os
from ..mck import Model

# static color variables
RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
BACKGROUND = "#FFFFFF"

# static variables
COLUMNS_ACC = [
    "Name",
    "Description",
    "Quantity",
    "Grouping",
    "Position_1",
    "Position_2",
    "Position_3",
    "Orientation_1",
    "Orientation_2",
    "Orientation_3",
]
COLUMNS_CHN = [
    "Name",
    "Description",
    "Quantity",
    "Grouping",
    "Position_1",
    "Position_2",
    "Position_3",
    "Direction_1",
    "Direction_2",
    "Direction_3",
]


class View3D:
    """
    A 3D display where structure, impacts, accelerometer and channels can be quickly displayed. Additionaly, all objects
    can be interactively placed on the mesh from a STL file. Also the 3D display supports basic animations.
    The units of the 3D display are in milimeters, use scale factor for different units.

    :param show_origin: Display the CSYS in origin
    :type show_origin: bool, optional
    :param show_axes: Display axes in the bottom left corner
    :type show_axes: bool, optional
    :param title: A title of the 3D display
    :type title: str, optional
    """

    def __init__(
        self, show_origin=True, show_axes=False, title=None, **kwargs
    ):
        self.plot = BackgroundPlotter(show=True, **kwargs)

        if title != None:
            self.plot.app_window.setWindowTitle("pyFBS - " + str(title))
        else:
            self.plot.app_window.setWindowTitle("pyFBS ")

        # set the pyFBS logo
        icon = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "logo-small.png",
        )
        self.plot.app_window.setWindowIcon(QtGui.QIcon(icon))

        self.plot.background_color = BACKGROUND
        self.arrow_source = pv.Arrow(start=(0.0, 0.0, 0.0))

        if show_origin:
            self.add_csys([0, 0, 0])

        if show_axes:
            self.plot.add_axes(labels_off=True)

        self.plot.enable_parallel_projection()

        # Static global variables
        self.global_acc = []
        self.acc_visible = False

        self.global_imp = None
        self.imp_visible = False

        self.global_chn = None
        self.imp_visible = False

        self.global_vps = []
        self.vps_visible = False

        self.global_points = None
        self.points_visible = False

        self.global_labels = []
        self.labels_visible = False

        self.name = None
        self.name_ev = None

        self.displayed_bodies = []
        self.mesh_dict = {}

        self.obj_animation = None
        self.modeshape_animation = None

        self.take_gif = False
        self.gif_dir = "sample.gif"

        # Interactive objects
        self.all_accs_dynamic = []
        self.all_imps_dynamic = []
        self.all_vps_dynamic = []

        # Toolbars
        self.show_hide_toolbar = self.plot.app_window.addToolBar(
            'Show/hide Actors'
        )
        self.animate_toolbar = self.plot.app_window.addToolBar(
            'Animate Modeshape'
        )
        self.animate_clear_toolbar = self.plot.app_window.addToolBar(
            'Clear Modeshape'
        )

        # Temp
        self.scale = 1
        self.toggle = None
        self.size = 10

    def add_modeshape(
        self,
        mode,
        mesh_or_name=None,
        fps=30,
        r_scale=10,
        no_frames=60,
        run_animation=False,
        add_note=False,
    ):
        """
        Add a modeshape animation to the 3D display.

        :param mode: Mode shape to animate.
        :type mode: np.ndarray with shape (number of points, number of modes)
        :param mesh_or_name: Mesh to animate retured by View3D.add_mesh() or
            View3D.add_model(), or a string with the name of the mesh. If None,
            the first mesh in the mesh dictionary will be used.
        :type mesh_or_name: str | pyvista.core.pointset.PolyData |
            pyvista.core.pointset.UnstructuredGrid | None
        :param fps: Frames per second for the animation.
        :type fps: int
        :param r_scale: Scale factor for the mode shape to be animated.
        :type r_scale: int | float
        :param no_frames: Number of frames to animate.
        :type no_frames: int
        :param run_animation: Run animation at start.
        :type run_animation: bool
        :param add_note: Add a note to a corner of the 3D display.
        :type add_note: bool
        """
        name = None
        if mesh_or_name is None:
            name = list(self.mesh_dict.keys())[0]
            mesh = self.mesh_dict[name]['mesh']
            ix = self.mesh_dict[name]['vtkOriginalPointIds']
            points = self.mesh_dict[name]['points']
        elif isinstance(mesh_or_name, str):
            if mesh_or_name in self.mesh_dict.keys():
                name = mesh_or_name
                mesh = self.mesh_dict[name]['mesh']
                ix = self.mesh_dict[name]['vtkOriginalPointIds']
                points = self.mesh_dict[name]['points']
            else:
                raise ValueError("Mesh name not found in the mesh dictionary.")
        elif isinstance(mesh_or_name, (pv.PolyData, pv.UnstructuredGrid)):
            for key, value in self.mesh_dict.items():
                if value['mesh'] is mesh_or_name:
                    name = key
                    mesh = value['mesh']
                    ix = self.mesh_dict[name]['vtkOriginalPointIds']
                    points = self.mesh_dict[name]['points']
                    break
        else:
            raise TypeError(
                "Mesh must be a string, PolyData, UnstructuredGrid or None."
            )
        if name is None:
            raise ValueError(
                "Mesh not found in the mesh dictionary. Please add a mesh \
                    first using add_mesh() or add_model()."
            )
        if ix is None:
            _mode = mode
        else:
            _mode = mode[ix]

        mode_dict = dict_animation(
            _mode,
            'modeshape',
            pts=points,
            mesh=mesh,
            fps=fps,
            r_scale=r_scale,
            no_points=no_frames,
        )
        self._add_modeshape(
            mode_dict, run_animation=run_animation, add_note=add_note
        )

    def _add_modeshape(
        self, dict_animation, run_animation=False, add_note=False
    ):
        """
        Add a modeshape animation to the 3D display.

        :param dict_animation: Mode shape dictionary
        :type dict_animation: dict
        :param run_animation: Run animation at start.
        :type run_animation: bool, optional
        :param add_note: Add a note to a corner of 3D display.
        :type add_note: bool, optional
        """

        if self.modeshape_animation == None:
            self.add_action(
                self.animate_toolbar,
                "Animate modeshape",
                self.animate_modeshape,
            )
            self.add_action(
                self.animate_clear_toolbar,
                "Clear modeshape",
                self.clear_modeshape,
            )

        self.modeshape_animation = dict_animation

        if add_note:
            _freq = self.modeshape_animation["freq"]
            _damp = self.modeshape_animation["damp"]
            _mcf = self.modeshape_animation["mcf"]

            self.plot.add_text(
                "Frequency = %4.1f Hz\nDamping = %4.3f%%\nMCF = %4.1f%%"
                % (_freq, _damp, _mcf),
                position='upper_right',
                font_size=10,
                color="k",
                font="times",
                name="Mode",
            )

        if run_animation:
            self.animate_modeshape()

    def _set_scalars(self, mesh, scalars_name, scalars):
        mesh[scalars_name] = scalars

    def _set_scalars_placeholder(self, mesh, scalars_name, scalars):
        """Do nothing, placeholder for when scalars are not needed."""
        return

    def animate_modeshape(self):
        """
        Animate a mode shape in the 3D display.
        """

        frameperiod = 1.0 / self.modeshape_animation["fps"]

        now = time()
        nextframe = now + frameperiod

        ann = self.modeshape_animation["animation_pts"]
        ann_secondary = self.modeshape_animation["animation_pts_secondary"]

        if self.take_gif:
            self.plot.open_gif(self.gif_dir)

        if self.modeshape_animation["scalars"]:
            set_lim = np.sqrt(np.mean(ann**2, axis=0))
            self.plot.update_scalar_bar_range(
                clim=[np.min(set_lim), np.max(set_lim)]
            )
            if (
                self.modeshape_animation["animate_secondary_mode_shape"]
                == True
            ):
                self.plot.update_scalar_bar_range(
                    clim=[np.min(ann_secondary), np.max(ann_secondary)]
                )

        if type(self.modeshape_animation["mesh"]) is not list:
            self.modeshape_animation["mesh"] = [
                self.modeshape_animation["mesh"]
            ]

        set_scalar_list = []
        for mesh in self.modeshape_animation["mesh"]:
            if mesh.active_scalars_name is None:
                set_scalar_list.append(self._set_scalars_placeholder)
            elif self.modeshape_animation["scalars"]:
                set_scalar_list.append(self._set_scalars)
        if self.modeshape_animation["animate_secondary_mode_shape"]:
            scalars = ann_secondary
        else:
            scalars = np.sqrt(np.mean(ann**2, axis=1)).reshape(
                self.modeshape_animation["or_pts"].shape[0], -1
            )
        for mesh in self.modeshape_animation["mesh"]:
            mesh.points = self.modeshape_animation["or_pts"]
        for i in range(ann.shape[2]):
            add_val = ann[:, :, i]
            for j, mesh in enumerate(self.modeshape_animation["mesh"]):
                mesh.points = self.modeshape_animation["or_pts"] + add_val
                scalars_name = mesh.active_scalars_name
                set_scalar_list[j](mesh, scalars_name, scalars[:, i])

            self.plot.render()
            if self.take_gif:
                self.plot.write_frame()

            while now < nextframe:
                sleep(nextframe - now)
                now = time()
            nextframe += frameperiod

        if self.take_gif:
            gif = imageio.mimread(self.gif_dir, memtest=False)
            imageio.mimsave(self.gif_dir, gif, fps=30)

    def clear_modeshape(self):
        """
        Clear mode shape from the 3D display.
        """

        for mesh in self.modeshape_animation["mesh"]:
            mesh.points = self.modeshape_animation["or_pts"]
            self.plot.render()
            scalars_name = mesh.active_scalars_name
            mesh[scalars_name] = np.zeros(
                self.modeshape_animation["or_pts"].shape[0]
            )
        self.plot.update_scalar_bar_range(clim=[-100, 100])

    def add_objects_animation(
        self, dict_animation, run_animation=False, add_note=False
    ):
        """
        Add an object animation to the 3D display.

        :param dict_animation: Object animation information.
        :type dict_animation: dict
        :param run_animation: Run animation at start.
        :type run_animation: bool, optional
        :param add_note: Add a note to a corner of the 3D display.
        :type add_note: bool, optional
        """

        if self.obj_animation == None:
            self.add_action(
                self.animate_toolbar, "Animate objects", self.animate_objects
            )

        self.obj_animation = dict_animation

        if add_note:
            _text = "Frequency = %4.1f Hz" % (self.obj_animation["freq"])
            self.plot.add_text(
                _text,
                position='upper_right',
                font_size=10,
                color="k",
                font="times",
                name="Mode",
            )

        if run_animation:
            self.animate_objects()

    def animate_objects(self):
        """
        Animate objects in the 3D display.
        """

        frameperiod = 1.0 / self.obj_animation["fps"]

        now = time()
        nextframe = now + frameperiod

        ann = self.obj_animation["animation_pts"]

        object_list = self.obj_animation["objects_list"]

        if self.take_gif:
            self.plot.open_gif(self.gif_dir)

        for i in range(ann.shape[2]):
            add_val = ann[:, :, i]

            for _object, loc in zip(object_list, add_val):
                for _pts, _mesh in zip(_object[0], _object[1]):

                    self.plot.update_coordinates(
                        _pts + loc, mesh=_mesh, render=False
                    )

            self.plot.render()
            if self.take_gif:
                self.plot.write_frame()

            while now < nextframe:
                sleep(nextframe - now)
                now = time()
            nextframe += frameperiod

        if self.take_gif:
            gif = imageio.mimread(self.gif_dir, memtest=False)
            imageio.mimsave(self.gif_dir, gif, fps=30)

    def add_action(self, toolbar, key, function):
        """
        Connects a toolbar button with a certain function

        :param toolbar: Toolbar parameter
        :param key: Name of the toolbar
        :param function: Function to connect the action to
        """

        action = QAction(key, self.plot.app_window)
        action.triggered.connect(function)
        toolbar.addAction(action)

    def add_csys(self, position=[0, 0, 0], size=10):
        """
        Add a coordinate system at a certain position.

        :param position: Position of the CSYS [x, y, z]
        :type position: list, optional
        :param size: Size of the CSYS
        :type size: float, optional
        """

        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(1, 0, 0))
        arrow.points *= size
        arrow.points += np.asarray(position)
        self.plot.add_mesh(arrow, color=RED)

        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0, 1, 0))
        arrow.points *= size
        arrow.points += np.asarray(position)
        self.plot.add_mesh(arrow, color=GREEN)

        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0, 0, 1))
        arrow.points *= size
        arrow.points += np.asarray(position)
        self.plot.add_mesh(arrow, color=BLUE)

    def add_stl(self, stl_path, name="model", **kwargs):
        """
        Adds a mesh to the 3D display from STL file.

        :param stl_path: Path to the .stl file
        :type stl_path: str
        :param color: Color of the mesh
        :type color: str, optional
        """

        mesh = pv.PolyData(stl_path)
        return self.add_mesh(mesh, name=name, cmap=None, **kwargs)

    def add_model(self, model, name="model", cmap="coolwarm", **kwargs):
        """
        Adds a pyfbs.mck.Model mesh to the 3D display.

        :param model: Model to be added
        :type model: pyfbs.mck.Model
        :param name: Name of the mesh
        :type name: str, optional
        :param cmap: If not None, applies a colormap to the mesh.
        :type str | None | matplotlib colormap: optional
        :param kwargs: Additional keyword arguments accepted by
            pyvista.Plotter.add_mesh()
        """

        if not isinstance(model, Model):
            raise TypeError("Model must be a pyfbs.mck.Model object.")

        mesh = model.mesh
        return self.add_mesh(mesh, name=name, cmap=cmap, **kwargs)

    def add_mesh(self, mesh, name="model", cmap="coolwarm", **kwargs):
        """
        Adds a mesh to the 3D display.

        :param mesh: Mesh to be added
        :type mesh: pyvista.PolyData | pyvista.UnstructuredGrid
        :param name: Name of the mesh
        :type name: str
        :param cmap: If not None, applies a colormap to the mesh.
        :type str | matplotlib colormap | None
        :param kwargs: Additional keyword arguments accepted by
            pyvista.Plotter.add_mesh()
        """

        if isinstance(mesh, pv.core.pointset.PolyData):
            _mesh = pv.PolyData(mesh, deep=True)
            ix = None
        elif isinstance(mesh, pv.core.pointset.UnstructuredGrid):
            _mesh = mesh.extract_surface()
            ix = _mesh['vtkOriginalPointIds']
        else:
            raise TypeError(
                "Mesh must be a PolyData or UnstructuredGrid object."
            )
        if cmap is not None:
            scalars = np.ones(_mesh.points.shape[0])
        else:
            scalars = None
        actor = self.plot.add_mesh(
            _mesh, name=name, scalars=scalars, cmap=cmap, **kwargs
        )
        self.displayed_bodies.append([name, actor])
        self.mesh_dict[name] = {
            'mesh': _mesh,
            'points': _mesh.points.copy(),
            'vtkOriginalPointIds': ix,
        }
        # initialize camera so that the view does not reset after dynamically
        # adding objects
        self.plot.camera.direction
        return _mesh

    def add_impact(self, position, direction, size=10, color=RED, **kwargs):
        """
        Adds an impact to the 3D display.

        :param position: Position of the impact [x,y,z]
        :type position: list
        :param direction: Direction of the impact [dx,dy,dz]
        :type direction: list
        :param size: Size of the impact
        :type size: float, optional
        :param color: Color of the impact
        :type color: str, optional
        """
        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=direction)
        arrow.translate(-1 * np.asarray(direction), inplace=True)
        arrow.points *= size
        arrow.translate(np.asarray(position), inplace=True)
        imp_actor = self.plot.add_mesh(
            arrow, color=color, reset_camera=False, **kwargs
        )

        return arrow, imp_actor

    def add_channel(self, position, direction, size=10, color=BLUE, **kwargs):
        """
        Adds a channel to the 3D display.

        :param position: Position of the channel [x,y,z]
        :type position: list
        :param direction: Direction of the channel [dx,dy,dz]
        :type direction: list
        :param size: Size of the channel
        :type size: float, optional
        :param color: Color of the channel
        :type color: str, optional
        """

        # arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=direction)
        # arrow.points *= size
        # arrow.points += np.asarray(position)
        # chn_actor = self.plot.add_mesh(
        #     arrow, color=color, reset_camera=False, **kwargs
        # )

        # return arrow, chn_actor
        pass

    def create_accelerometer(self, position, orientation, size=10):
        """
        Create a 3D model of an accelerometer.

        :param position: Position of the accelerometer [x,y,z]
        :type position: list
        :param orientation: Orientation of the accelerometer [ax,ay,az]
        :type orientation: list
        :param size: Size of the accelerometer
        :type size: float, optional
        :return: Accelerometer object
        """

        box = pv.Box(
            bounds=(
                -size / 2,
                size / 2,
                -size / 2,
                size / 2,
                -size / 2,
                size / 2,
            )
        )
        cable = pv.Cylinder(
            center=(-size / 2 - size / 8, 0, 0),
            direction=(-1, 0, 0),
            radius=size / 5,
            height=size / 4,
        )

        ray_x = pv.Line(
            np.asarray([0, 0, 0]) - size / 2,
            np.asarray([size, 0, 0]) - size / 2,
        )
        ray_y = pv.Line(
            np.asarray([0, 0, 0]) - size / 2,
            np.asarray([0, size, 0]) - size / 2,
        )
        ray_z = pv.Line(
            np.asarray([0, 0, 0]) - size / 2,
            np.asarray([0, 0, size]) - size / 2,
        )

        accelerometer = [box, cable, ray_x, ray_y, ray_z]
        _new = position

        r = R.from_euler(
            'xyz',
            [orientation[0], orientation[1], orientation[2]],
            degrees=True,
        )
        rot = r.as_matrix()

        for item in accelerometer:
            item.points = (rot @ item.points.T).T

            item.translate(_new, inplace=True)

        return accelerometer

    def add_accelerometer(self, acc):
        """
        Add an accelerometer to the 3D display.

        :param acc: Accelerometer object
        :type acc: list
        :return: List of accelerometer mesh actors
        """

        acc_1 = self.plot.add_mesh(
            acc[0],
            opacity=0.5,
            show_edges=True,
            color="#8c8c8c",
            reset_camera=False,
        )
        acc_2 = self.plot.add_mesh(
            acc[1],
            opacity=0.5,
            show_edges=False,
            color="#8c8c8c",
            reset_camera=False,
        )
        acc_3 = self.plot.add_mesh(
            acc[2], color=RED, line_width=5, reset_camera=False
        )
        acc_4 = self.plot.add_mesh(
            acc[3], color=GREEN, line_width=5, reset_camera=False
        )
        acc_5 = self.plot.add_mesh(
            acc[4], color=BLUE, line_width=5, reset_camera=False
        )

        return [acc_1, acc_2, acc_3, acc_4, acc_5]

    def add_vp(self, position, size=10, color=GREEN, **kwargs):
        """
        Add a virtual point to the 3D display.

        :param position: Position of the virtual point [x,y,z]
        :type position: array(float)
        :param size: Size of the virtual point
        :type size: float, optional
        :param color: Color of the virtual point
        :type color: str, optional
        """

        sphere = pv.Sphere(radius=size, center=position)
        vp_actor = self.plot.add_mesh(
            sphere, color=color, reset_camera=False, **kwargs
        )

        return sphere, vp_actor

    def acc_callback(
        self, point, orientation=None, fixed_rotation=None, normal=None
    ):
        """
        Interactive accelerometer callback function.

        :param point: Point in 3D space
        :type point: array(float)
        :param orientation: Orientation in 3D space
        :type orientation: array(float), optional
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        """
        i = int(
            len(self.all_accs_dynamic)
            + len(self.all_imps_dynamic)
            + len(self.all_vps_dynamic)
        )
        size = self.size

        if np.asarray(orientation).all() == None:
            acc = self.create_accelerometer(
                [size / 2, size / 2, size / 2], [0, 0, 0], size=size
            )
            rot = np.diag([1] * 3)
        else:
            acc = self.create_accelerometer(
                [size / 2, size / 2, size / 2], orientation, size=size
            )
            r = R.from_euler('xyz', orientation, degrees=True)
            rot = r.as_matrix()

        self.add_accelerometer(acc)
        _gg = DynamicPosition(
            acc,
            self.plot,
            i,
            mesh=self.mesh,
            size=size,
            rot=rot,
            fixed_rotation=self.fixed_rotation,
            snap_outward=True,
            acc_normal=self.acc_normal,
            grouping=self.grouping,
        )
        self.plot.add_sphere_widget(
            _gg.callback,
            center=_gg.points,
            color=["k", "r", "g", "b"],
            radius=size / 15,
        )
        if normal is None:
            _gg.translate(point)
        else:
            _gg.snap_outward = False
            _point = point - normal * size / 2
            _gg.translate(_point, snap=True)
            _gg.translate(point)
            _gg.snap_outward = True
        _gg.turn_on = True
        self.all_accs_dynamic.append(_gg)

    def add_acc_dynamic(
        self,
        mesh,
        predefined=None,
        scale=1,
        fixed_rotation=None,
        size=10,
        acc_normal=None,
        grouping=None,
    ):
        """
        Add a set of predefined accelerometers to the 3D display and toggle
        the possibility to add additional accelerometers.

        :param mesh: A mesh on which to snap
        :type mesh: array(float)
        :param predefined: Predefined set of accelerometers
        :type predefined: pd.DataFrame, optional
        :param scale: distance scaling factor
        :type scale: float
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        :param acc_normal: Accelerometer normal direction for snapping
        :type acc_normal: str | None. Valid options are "x", "y", "z", "-x",
            "-y", "-z", or None
        :param grouping: Grouping for the accelerometer
        """
        self.size = size
        self.fixed_rotation = fixed_rotation
        self.mesh = mesh
        self.mesh.compute_normals(auto_orient_normals=True, inplace=True)

        self.toggle = "acc"
        self.acc_normal = acc_normal
        self.grouping = grouping

        if isinstance(predefined, pd.DataFrame):
            for i, row in predefined.iterrows():
                point = np.asarray(
                    [
                        row["Position_1"] * scale,
                        row["Position_2"] * scale,
                        row["Position_3"] * scale,
                    ]
                )
                orientation = np.asarray(
                    [
                        row["Orientation_1"],
                        row["Orientation_2"],
                        row["Orientation_3"],
                    ]
                )
                self.acc_callback(
                    point,
                    orientation=orientation,
                    fixed_rotation=self.fixed_rotation,
                )

        self.plot.disable_picking()
        self.plot.enable_surface_point_picking(
            callback=self.acc_surface_callback,
            show_message=False,
            show_point=False,
        )
        self.plot.add_text(
            'Use right mouse click to add an accelerometer, hold the letter '
            'T to not snap to mesh.',
            color="k",
            font="times",
            font_size=10,
            name="text",
        )

    def imp_callback(self, point, direction=None, fixed_rotation=None):
        """
        Interactive impact callback function.

        :param point: Point in 3D space
        :type point: array(float)
        :param orientation: Orientation in 3D space
        :type orientation: array(float)
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        """

        i = int(
            len(self.all_accs_dynamic)
            + len(self.all_imps_dynamic)
            + len(self.all_vps_dynamic)
        )
        size = self.size
        if np.asarray(direction).all() == None:
            imp, _ = self.add_impact(
                [size / 2, size / 2, size / 2], [0, 0, 1], size=self.size
            )
            rot = np.diag([1] * 3)
        else:
            imp, _ = self.add_impact(
                [size / 2, size / 2, size / 2], direction, size=self.size
            )
            rot = rotation_matrix_from_vectors(direction, [0, 0, 1]).T

        _gg = DynamicPosition(
            [imp],
            self.plot,
            i,
            mesh=self.mesh,
            size=self.size,
            rot=rot,
            snap_outward=False,
            fixed_rotation=self.fixed_rotation,
            toggle="impact",
            grouping=self.grouping,
        )
        self.plot.add_sphere_widget(
            _gg.callback,
            center=_gg.points,
            color=["k", "r", "g", "b"],
            radius=self.size / 15,
        )
        _gg.translate(point, snap=True)
        _gg.turn_on = True
        self.all_imps_dynamic.append(_gg)

    def add_imp_dynamic(
        self,
        mesh,
        predefined=None,
        scale=1,
        fixed_rotation=None,
        size=10,
        grouping=None,
    ):
        """
        Add a set of predefined impacts to the 3D display and toggle the
        possibility to add additional impacts.

        :param mesh: A mesh on which to snap
        :type mesh: array(float)
        :param predefined: Predefined set of impacts
        :type predefined: pd.DataFrame, optional
        :param scale: distance scaling factor
        :type scale: float
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        :param grouping: Grouping for the impact
        """
        self.size = size

        self.mesh = mesh
        self.mesh.compute_normals(auto_orient_normals=True, inplace=True)
        self.fixed_rotation = fixed_rotation
        self.grouping = grouping

        self.toggle = "impact"

        if isinstance(predefined, pd.DataFrame):
            for i, row in predefined.iterrows():
                point = np.asarray(
                    [
                        row["Position_1"] * scale,
                        row["Position_2"] * scale,
                        row["Position_3"] * scale,
                    ]
                )
                direction = np.asarray(
                    [
                        row["Direction_1"],
                        row["Direction_2"],
                        row["Direction_3"],
                    ]
                )
                self.imp_callback(
                    point,
                    direction=direction,
                    fixed_rotation=self.fixed_rotation,
                )

        self.plot.disable_picking()
        self.plot.enable_surface_point_picking(
            callback=self.impact_surface_callback,
            show_message=False,
            show_point=False,
        )
        self.plot.add_text(
            'Use right mouse click to add an impact, hold the letter T to '
            'not snap to mesh.',
            color="k",
            font="times",
            font_size=10,
            name="text",
        )

    def vp_callback(self, point, fixed_rotation=None):
        """
        Interactive virtual point callback function.

        :param point: Point in 3D space
        :type point: array(float)
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        """

        i = int(
            len(self.all_accs_dynamic)
            + len(self.all_imps_dynamic)
            + len(self.all_vps_dynamic)
        )
        size = 4

        acc, _ = self.add_vp(
            [size / 2, size / 2, size / 2], size=size, opacity=0.1
        )

        _gg = DynamicPosition(
            [acc],
            self.plot,
            i,
            mesh=self.mesh,
            size=4,
            snap_outward=False,
            fixed_rotation=self.fixed_rotation,
            grouping=self.grouping,
        )
        self.plot.add_sphere_widget(
            _gg.callback,
            center=_gg.points,
            color=["k", "r", "g", "b"],
            radius=4 / 15,
        )

        _gg.turn_on = True
        _gg.translate(point, snap=True)

        self.all_vps_dynamic.append(_gg)

    def add_vp_dynamic(
        self,
        mesh,
        predefined=None,
        scale=1,
        fixed_rotation=None,
        grouping=None,
    ):
        """
        Add a set of predefined virtual points to the 3D display and toggle '
        'the possibility to add
        additional virtual points.

        :param mesh: A mesh on which to snap
        :type mesh: array(float)
        :param predefined: Predefined set of virtual points
        :type predefined: pd.DataFrame, optional
        :param scale: distance scaling factor
        :type scale: float
        :param fixed_rotation: fixed rotation angle
        :type fixed_rotation: float
        """

        self.mesh = mesh
        self.grouping = grouping

        if isinstance(predefined, pd.DataFrame):

            x = predefined["Position_1"].unique()
            y = predefined["Position_2"].unique()
            z = predefined["Position_3"].unique()
            position = np.asarray([x, y, z]).T
            position *= scale
            for pos in position:
                self.vp_callback(pos, fixed_rotation=self.fixed_rotation)

        self.plot.disable_picking()
        self.plot.enable_surface_point_picking(
            callback=self.vp_surface_callback,
            show_message=False,
            show_point=False,
        )
        self.plot.add_text(
            "Press P too add a VP (hold down letter T to disable snapping to "
            "mesh).",
            font_size=10,
            color="k",
            font="times",
            name="text",
        )

    def get_imp_data(self):
        """
        Returns positional information on current impacts in the 3D display.

        :return: pd.DataFrame containing positional information on impacts
        """

        columns_chann = [
            "Name",
            "Description",
            "Quantity",
            "Grouping",
            "Position_1",
            "Position_2",
            "Position_3",
            "Direction_1",
            "Direction_2",
            "Direction_3",
        ]
        df = pd.DataFrame(columns=columns_chann)

        for i, _imp in enumerate(self.all_imps_dynamic):
            pos, _dir = _imp.get_pos_orient(one_dir=2)

            data_chn = np.asarray(
                [
                    [
                        "Impact " + str(1 + i),
                        None,
                        None,
                        _imp.grouping,
                        pos[0],
                        pos[1],
                        pos[2],
                        _dir[0],
                        _dir[1],
                        _dir[2],
                    ]
                ]
            )

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = pd.concat([df, df_row], ignore_index=True)

        return df

    def get_acc_data(self):
        """
        Returns positional information on interactive accelerometers in the 3D
        display.

        :return: pd.DataFrame containing positional information on
            accelerometers
        """

        columns_chann = [
            "Name",
            "Description",
            "Quantity",
            "Grouping",
            "Position_1",
            "Position_2",
            "Position_3",
            "Orientation_1",
            "Orientation_2",
            "Orientation_3",
        ]
        df = pd.DataFrame(columns=columns_chann)

        for i, _acc in enumerate(self.all_accs_dynamic):
            pos, euler_dir = _acc.get_pos_orient(euler_angles=True)
            data_chn = np.asarray(
                [
                    [
                        "Sensor " + str(1 + i),
                        None,
                        None,
                        _acc.grouping,
                        pos[0],
                        pos[1],
                        pos[2],
                        euler_dir[0],
                        euler_dir[1],
                        euler_dir[2],
                    ]
                ]
            )

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = pd.concat([df, df_row], ignore_index=True)

        return df

    def get_vp_data(self):
        """
        Returns positional information on interactive virtual points in the 3D
        display.

        :return: pd.DataFrame containing positional information on virtual
            points
        """

        columns_chann = [
            "Name",
            "Description",
            "Quantity",
            "Grouping",
            "Position_1",
            "Position_2",
            "Position_3",
            "Orientation_1",
            "Orientation_2",
            "Orientation_3",
        ]
        df = pd.DataFrame(columns=columns_chann)

        for i, _acc in enumerate(self.all_vps_dynamic):
            pos, euler_dir = _acc.get_pos_orient(euler_angles=True)
            data_chn = np.asarray(
                [
                    [
                        "VP " + str(1 + i),
                        None,
                        None,
                        _acc.grouping,
                        pos[0],
                        pos[1],
                        pos[2],
                        euler_dir[0],
                        euler_dir[1],
                        euler_dir[2],
                    ]
                ]
            )

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = pd.concat([df, df_row], ignore_index=True)

        return df

    def show_acc(self, df, size=10, overwrite=True, scale=1):
        """
        Add accelerometers to the 3D display.

        :param df: A DataFrame containing information on the accelerometers
        :type df: pd.DataFrame
        :param size: size of the accelerometer
        :type size: float, optional
        :param overwrite: Toggle the option to overwrite currently displayed
            accelerometers
        :type overwrite: bool, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_acc != []:
            if overwrite:
                self.acc_visible = True
                self.show_hide_accelerometers()
                self.global_acc = []
            else:
                pass
        else:
            self.add_action(
                self.show_hide_toolbar,
                "Sensors",
                self.show_hide_accelerometers,
            )

        for i, row in df.iterrows():

            acc_mesh = self.create_accelerometer(
                (
                    row["Position_1"] * scale,
                    row["Position_2"] * scale,
                    row["Position_3"] * scale,
                ),
                (
                    row["Orientation_1"],
                    row["Orientation_2"],
                    row["Orientation_3"],
                ),
                size=size,
            )

            acc_actor = self.add_accelerometer(acc_mesh)
            acc_pts = []

            for i in range(5):
                acc_pts.append(acc_mesh[i].points.copy())

            self.global_acc.append([acc_pts, acc_mesh, acc_actor])
        self.acc_visible = True

    def show_imp(
        self, df, color=RED, overwrite=True, scale=1, size=10, **kwargs
    ):
        """
        Add impacts to the 3D display.

        :param df: A DataFrame containing information on the impacts
        :type df: pd.DataFrame
        :param color: Color of the channel
        :type color: str, optional
        :param overwrite: Toggle option to overwrite currently displayed
            impacts
        :type overwrite: bool, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_imp is not None:
            if overwrite:
                self.imp_visible = True
                self.show_hide_impacts()
                self.global_imp = None
            else:
                pass
        else:
            self.add_action(
                self.show_hide_toolbar, "Impacts", self.show_hide_impacts
            )

        points = (
            df[["Position_1", "Position_2", "Position_3"]].to_numpy() * scale
        )
        directions = df[
            ["Direction_1", "Direction_2", "Direction_3"]
        ].to_numpy()
        points = points - size * directions
        points_polydata = pv.PolyData(points)
        points_polydata['vectors'] = directions
        _arrow = self.arrow_source.copy()
        _arrow.points *= size
        imp_actor_old = None
        if self.global_imp is not None:
            imp_mesh, imp_actor_old, color_dict = self.global_imp
            color_key = len(color_dict)
            color_dict[color_key] = color
            points_polydata['categories'] = np.full(points.shape[0], color_key)
            points_polydata = points_polydata.color_labels(
                color_dict, scalars='categories', inplace=False
            )
            arrows = points_polydata.glyph(
                orient='vectors', geom=_arrow  # , scale=False
            )
            arrows = imp_mesh.merge(arrows)
        elif self.global_imp is None:
            color_dict = {0: color}
            points_polydata['categories'] = np.zeros(
                points.shape[0], dtype=int
            )
            points_polydata = points_polydata.color_labels(
                color_dict, scalars='categories', inplace=False
            )
            arrows = points_polydata.glyph(
                orient='vectors', geom=_arrow  # , scale=False
            )

        imp_actor = self.plot.add_mesh(
            arrows,
            rgb=True,
            scalars='categories_rgb',
            reset_camera=False,
            **kwargs
        )

        if imp_actor_old is not None:
            self.plot.remove_actor(imp_actor_old)
        self.global_imp = [arrows, imp_actor, color_dict]

        self.imp_visible = True

    def show_chn(
        self, df, color=BLUE, overwrite=True, scale=1, size=10, **kwargs
    ):
        """
        Add channels to the 3D display.

        :param df: A DataFrame containing information on the channels
        :type df: pd.DataFrame
        :param color: Color of the channel
        :type color: str, optional
        :param overwrite: Toggle option to overwrite currently displayed
            channels
        :type overwrite: bool, optional
        :param size: size of the accelerometer
        :type size: float, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_chn is not None:
            if overwrite:
                self.chn_visible = True
                self.show_hide_channels()
                self.global_chn = None
            else:
                pass
        else:
            self.add_action(
                self.show_hide_toolbar, "Channels", self.show_hide_channels
            )

        points = (
            df[["Position_1", "Position_2", "Position_3"]].to_numpy() * scale
        )
        directions = df[
            ["Direction_1", "Direction_2", "Direction_3"]
        ].to_numpy()
        points_polydata = pv.PolyData(points)
        points_polydata['vectors'] = directions
        _arrow = self.arrow_source.copy()
        _arrow.points *= size
        chn_actor_old = None
        if self.global_chn is not None:
            chn_mesh, chn_actor_old, color_dict = self.global_chn
            color_key = len(color_dict)
            color_dict[color_key] = color
            points_polydata['categories'] = np.full(points.shape[0], color_key)
            points_polydata = points_polydata.color_labels(
                color_dict, scalars='categories', inplace=False
            )
            arrows = points_polydata.glyph(
                orient='vectors', geom=_arrow  # , scale=False
            )
            arrows = chn_mesh.merge(arrows)
        elif self.global_chn is None:
            color_dict = {0: color}
            points_polydata['categories'] = np.zeros(
                points.shape[0], dtype=int
            )
            points_polydata = points_polydata.color_labels(
                color_dict, scalars='categories', inplace=False
            )
            arrows = points_polydata.glyph(
                orient='vectors', geom=_arrow  # , scale=False
            )

        chn_actor = self.plot.add_mesh(
            arrows,
            rgb=True,
            scalars='categories_rgb',
            reset_camera=False,
            **kwargs
        )

        if chn_actor_old is not None:
            self.plot.remove_actor(chn_actor_old)
        self.global_chn = [arrows, chn_actor, color_dict]

        self.chn_visible = True

    def show_vp(
        self, df, color=GREEN, overwrite=True, size=10, scale=1, **kwargs
    ):
        """
        Add virtual points to the 3D display.

        :param df: A DataFrame containing information on the virtual points
        :type df: pd.DataFrame
        :param color: Color of the virtual point
        :type color: str, optional
        :param overwrite: Toggle option to overwrite currently displayed
            channels
        :type overwrite: bool, optional
        :param size: size of the accelerometer
        :type size: float, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_vps != []:
            if overwrite:
                self.vps_visible = True
                self.show_hide_vps()
                self.global_vps = []
            else:
                pass
        else:
            self.add_action(self.show_hide_toolbar, "VPs", self.show_hide_vps)

        ind = np.unique(df["Grouping"], return_index=True)[1]

        x = df.iloc[ind]["Position_1"]
        y = df.iloc[ind]["Position_2"]
        z = df.iloc[ind]["Position_3"]
        position = np.asarray([x, y, z]).T
        position *= scale

        for position_ in position:
            vp_mesh, vp_actor = self.add_vp(
                position_, color=color, size=size, **kwargs
            )
            self.global_vps.append([vp_mesh, vp_actor])

        self.vps_visible = True

    def show_points(
        self, df, color=GREEN, overwrite=True, size=10, scale=1, **kwargs
    ):
        if self.global_points is not None:
            if overwrite:
                self.points_visible = True
                self.show_hide_points()
                self.global_points = None
        else:
            self.add_action(
                self.show_hide_toolbar, "Points", self.show_hide_points
            )
        points = (
            df[['Position_1', 'Position_2', 'Position_3']].to_numpy() * scale
        )
        point_cloud = pv.PolyData(points)

        self.add_mesh(
            mesh=point_cloud,
            name="points",
            render_points_as_spheres=True,
            point_size=size,
            style='points',
            color=color,
            **kwargs
        )
        actor = self.displayed_bodies[-1][-1]
        self.global_points = [point_cloud, actor]
        self.points_visible = True

    def show_hide_points(self):
        """
        Toggles visibility of the channels in the 3D display.
        """

        if self.points_visible == False:
            if self.global_points is not None:
                self.plot.add_actor(self.global_points[1], reset_camera=False)
            self.points_visible = True

        else:
            if self.global_points is not None:
                self.plot.remove_actor(
                    self.global_points[1], reset_camera=False
                )
            self.points_visible = False

    def label_acc(
        self, df, name="Accelerometers", font_size=12, scale=1, **kwargs
    ):
        """
        Add labels of accelerometers to the 3D display.

        :param df: A DataFrame containing relevant information on the sensors
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing
            notations
        :type name: str, optional
        :param font_size: Size of the label font
        :type font_size: float
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_labels == []:
            self.add_action(
                self.show_hide_toolbar, "Clear Labels", self.clear_labels
            )

        positions = []
        labels = []
        for i, row in df.iterrows():
            positions.append(
                [
                    row["Position_1"] * scale,
                    row["Position_2"] * scale,
                    row["Position_3"] * scale,
                ]
            )
            labels.append(row["Name"])

        self.plot.add_point_labels(
            positions,
            labels,
            font_size=font_size,
            name=name,
            shape_opacity=0.5,
            show_points=False,
            **kwargs
        )
        self.global_labels.append([[positions, labels], name])
        self.labels_visible = True

    def label_imp(self, df, name="Impacts", font_size=12, scale=1, **kwargs):
        """
        Add labels of impacts to the 3D display.

        :param df: A DataFrame containing information on the impacts
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing
            notations
        :type name: str, optional
        :param font_size: Size of the label font
        :type font_size: float, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_labels == []:
            self.add_action(
                self.show_hide_toolbar, "Clear Labels", self.clear_labels
            )

        positions = []
        labels = []
        for i, row in df.iterrows():

            positions.append(
                [
                    row["Position_1"] * scale,
                    row["Position_2"] * scale,
                    row["Position_3"] * scale,
                ]
            )
            labels.append(row["Name"])

        self.plot.add_point_labels(
            positions,
            labels,
            font_size=font_size,
            name=name,
            shape_color=RED,
            font_family="times",
            shape_opacity=0.5,
            show_points=False,
            **kwargs
        )
        self.global_labels.append([[positions, labels], name])
        self.labels_visible = True

    def label_chn(
        self, df, name="Channels", size=10, font_size=12, scale=1, **kwargs
    ):
        """
        Adds labels of channels to the 3D display.

        :param df: A DataFrame containing relevant information about the
            channels
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing
            notations
        :type name: str, optional
        :param size: Size of the channel arrow
        :type size: float, optional
        :param font_size: Size of the label font
        :type font_size: float, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_labels == []:
            self.add_action(
                self.show_hide_toolbar, "Clear Labels", self.clear_labels
            )

        positions = []
        labels = []
        for i, row in df.iterrows():
            x = row["Direction_1"] * size
            y = row["Direction_2"] * size
            z = row["Direction_3"] * size

            positions.append(
                [
                    row["Position_1"] * scale + x,
                    row["Position_2"] * scale + y,
                    row["Position_3"] * scale + z,
                ]
            )
            labels.append(row["Name"])

        self.plot.add_point_labels(
            positions,
            labels,
            font_size=font_size,
            name=name,
            shape_color=BLUE,
            font_family="times",
            shape_opacity=0.5,
            show_points=False,
            **kwargs
        )
        self.global_labels.append([[positions, labels], name])
        self.labels_visible = True

    def label_vp(self, df, name="VPs", font_size=12, scale=1, **kwargs):
        """
        Adds labels to virtual point from the DataFrame to 3D display.

        :param df: A DataFrame containing information on the virtual points
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing
            notations
        :type name: str, optional
        :param font_size: Size of the label font
        :type font_size: float, optional
        :param scale: distance scaling factor
        :type scale: float
        """

        if self.global_labels == []:
            self.add_action(
                self.show_hide_toolbar, "Clear Labels", self.clear_labels
            )

        ind = np.unique(df["Grouping"], return_index=True)[1]

        x = df.iloc[ind]["Position_1"]
        y = df.iloc[ind]["Position_2"]
        z = df.iloc[ind]["Position_3"]
        position = np.asarray([x, y, z]).T
        position *= scale

        L = df.iloc[ind]["Grouping"]

        self.plot.add_point_labels(
            position,
            L,
            font_size=font_size,
            name=name,
            font_family="times",
            shape_opacity=0.5,
            shape_color=GREEN,
            show_points=False,
            **kwargs
        )
        self.global_labels.append([[position, L], name])
        self.labels_visible = True

    def show_hide_accelerometers(self):
        """
        Toggles visibility of the accelerometers in the 3D display.
        """

        if self.acc_visible == False:
            for _acc in self.global_acc:
                for item in _acc[2]:
                    self.plot.add_actor(item, reset_camera=False)
            self.acc_visible = True

        else:
            for _acc in self.global_acc:
                for item in _acc[2]:
                    self.plot.remove_actor(
                        item, reset_camera=False
                    )  # ,render = False)

            self.acc_visible = False

    def show_hide_impacts(self):
        """
        Toggles visibility of the impacts in the 3D display.
        """

        if self.imp_visible == False:
            for _imp in self.global_imp:
                self.plot.add_actor(_imp[1], reset_camera=False)
            self.imp_visible = True

        else:
            for _imp in self.global_imp:
                self.plot.remove_actor(_imp[1], reset_camera=False)
            self.imp_visible = False

    def show_hide_channels(self):
        """
        Toggles visibility of the channels in the 3D display.
        """

        if self.chn_visible == False:
            if self.global_chn is not None:
                self.plot.add_actor(self.global_chn[1], reset_camera=False)
            self.chn_visible = True

        else:
            if self.global_chn is not None:
                self.plot.remove_actor(self.global_chn[1], reset_camera=False)

            self.chn_visible = False

    def show_hide_vps(self):
        """
        Toggles visibility of the virtual points in the 3D display.
        """

        if self.vps_visible == False:
            for _vp in self.global_vps:
                self.plot.add_actor(_vp[1], reset_camera=False)
            self.vps_visible = True

        else:
            for _vp in self.global_vps:
                self.plot.remove_actor(_vp[1], reset_camera=False)

            self.vps_visible = False

    def clear_labels(self):
        """
        Clear all displayed labels in the 3D display.
        """
        for _label in self.global_labels:
            self.plot.remove_actor(_label[1], reset_camera=False)

        self.labels_visible = False

    def _ray_trace(self, point):
        # Ray trace for precise placement
        direction = point - self.plot.camera_position[0]
        direction /= np.linalg.norm(direction)
        start = point - 1000 * direction  # Ray start
        end = point + 10000 * direction  # Ray end

        # Perform ray trace
        point_ray, ix = self.mesh.ray_trace(start, end, first_point=True)

        # Use ray result if valid
        final_point = point_ray if len(point_ray) > 0 else None
        return final_point

    def _ray_traced_point_and_normal(self, point):
        # Ray trace to find the closest point on the mesh
        final_point = self._ray_trace(point)
        if final_point is None:
            return None, None
        # Proceed with placement
        cell_id = self.mesh.find_closest_cell(final_point)
        normal = self.mesh.cell_normals[cell_id]
        camera_direction = self.plot.camera.direction
        if np.dot(normal, camera_direction) > 0:
            normal = -normal
        return final_point, normal

    def impact_surface_callback(self, point):
        final_point, normal = self._ray_traced_point_and_normal(point)
        if final_point is None:
            return
        self.imp_callback(final_point, direction=-normal)

    def acc_surface_callback(self, point):
        final_point, normal = self._ray_traced_point_and_normal(point)
        if final_point is None:
            return
        rot = rotation_matrix_from_vectors(
            np.array(-normal), np.array([0.0, 0.0, 1.0])
        )
        local_orientation = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        local_orientation = rot @ (local_orientation)
        r = R.from_matrix(local_orientation)
        orientation = np.asarray(r.as_euler('xyz', degrees=True))
        # just added static size of the accelerometer
        final_point += np.array(+normal) / 2 * self.size
        self.acc_callback(final_point, orientation=orientation, normal=normal)

    def vp_surface_callback(self, point):
        final_point, normal = self._ray_traced_point_and_normal(point)
        if final_point is None:
            return
        self.vp_callback(final_point)


class DynamicPosition:
    """
    A wrapper for object interaction within the 3D display.

    :param objects: A list of objects to be interacted with withing the 3D
        display
    :type objects: list
    :param p: A pv.BackgroundPlotter instance in which the object are
        interacted with
    :type: pv.BackgroundPlotter
    :param N: An unique iteration number for sphere widgets
    :type N: int
    :param mesh: A mesh for object snapping
    :type mesh: array(float), optional
    :param snap_outward: An option for snapping outward of the mesh
    :type snap_outward: bool, optional
    :param size: size of the bounding box for the object
    :type size: float, optional
    :param rot: default orientation matrix of the object
    :tpye rot: array(float), optional
    :param fixed_rotation: fixed rotation angle
    :type fixed_rotation: float
    :param acc_normal: Accelerometer normal direction for snapping
    :type acc_normal: str | None. Valid options are "x", "y", "z", "-x", "-y",
        "-z", or None
    """

    def __init__(
        self,
        objects,
        p,
        N,
        mesh=None,
        snap_outward=True,
        size=1,
        rot=np.diag([1] * 3),
        fixed_rotation=None,
        toggle="acc",
        acc_normal=None,
        grouping=None,
    ):
        # set size and static points
        self.size = size
        self.points = (
            np.array([[0.0, 0.0, 0.0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
            * size
        )

        self.points = (rot.T @ (self.points).T).T + size / 2

        # Creates a bounding box
        self.box = pv.Box((-size, size, -size, size, -size, size))
        self.box.translate([size, size, size], inplace=True)
        self.box.points /= 2

        self.fixed_theta = fixed_rotation
        # defines the objects
        objects.insert(0, self.box)
        self.objects = objects

        # get the number of dynamic stuff in the display window
        self.N = int(N * 4)

        # local orientation of the bounding box/object
        self.local_orientation = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # local positions of the point widgets
        self.local_widgets = (
            np.array([[0.0, 0.0, 0.0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
            * size
        )
        if toggle == "impact":
            self.local_widgets[
                -1
            ] *= -1  # flip the z-rotation widget for impact

        # local normals on which the snapping happens
        self.local_normals = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
            ]
        ).T

        # ray_size on which the snapping to the mesh happens
        ray_size = 4 * size
        self.local_rays = (
            np.asarray(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ]
            ).T
            * ray_size
        )

        # allign everything with new rotational matrix
        self.local_orientation = rot @ (self.local_orientation)
        self.local_widgets = self.local_widgets @ rot.T
        self.local_normals = rot @ self.local_normals
        self.local_rays = rot @ self.local_rays

        if not toggle == "impact" and acc_normal is not None:
            acc_normal_dict = {
                "x": 0,
                "y": 1,
                "z": 2,
                "-x": 3,
                "-y": 4,
                "-z": 5,
            }
            acc_normal = acc_normal_dict[acc_normal]

        # computes mesh normals
        self.mesh = mesh
        self.mesh.compute_normals(auto_orient_normals=True, inplace=True)

        # disables the rotation of the object
        self.turn_on = False

        self.snap_outward = snap_outward

        # define display
        self.p = p
        self.toggle = toggle
        self.acc_normal = acc_normal

        # define grouping number
        self.grouping = grouping

    def get_pos_orient(
        self, euler_angles=False, one_dir=None, eps=1e-10, scale=1
    ):
        """
        Extracts the positional and orientational data of the object.

        :param euler_angles: Define orientation with Euler angles
        :type euler_angles: bool, optional
        :param one_dir: Extracts the direction of only one axis
        :type one_dir: int, optional
        :param eps: A round-off error
        :type eps: float, optional
        :return:
        """

        position = self.box.center_of_mass()

        # return euler_angles
        if euler_angles:
            r = R.from_matrix(self.local_orientation)
            orientation = r.as_euler('xyz', degrees=True)

        # only one direction
        elif one_dir != None:
            r = R.from_matrix(self.local_orientation)
            r = r.as_matrix().T
            orientation = r[one_dir, :]

        # whole orientation
        else:
            orientation = self.local_orientation

        # set to zero for very small numbers
        orientation[np.abs(orientation) < eps] = 0

        return position / scale, orientation

    def translate(self, point, snap=False):
        """
        Translation to the corresponding point. If snapping is enabled the
        object snapps to the mesh surface and it's oriented based on the mesh
        normal.

        :param point: Point in 3D space
        :type point: array(float)
        :param snap: Enable or disable snapping to mesh
        :type snap: bool, optional
        """

        # default option - no rotation
        rot = np.diag([1.0, 1.0, 1.0])
        camera_direction = self.p.camera.direction
        f = None

        if snap and not (kb.is_pressed('t')):
            direction = np.asarray(point) - np.asarray(
                self.p.camera_position[0]
            )
            direction = direction / np.linalg.norm(direction)
            start = point - 1000 * direction
            end = point + 10000 * direction
            points, ind = self.mesh.ray_trace(start, end, first_point=True)

            # fast upgrade to option B
            if self.snap_outward and points.size != 0:
                f = self.mesh.cell_normals[ind[0]]
                point = points - f / 2 * self.size
                direction = np.asarray(point) - np.asarray(
                    self.p.camera_position[0]
                )
                direction = direction / np.linalg.norm(direction)
                start = point - 1000 * direction
                end = point + 10000 * direction
                points, ind = self.mesh.ray_trace(start, end, first_point=True)

            # if there is an intersection and if "t" is not pressed go forward
            if points.size != 0 and not (kb.is_pressed('t')):
                # find the nearest normal
                v2 = self.mesh.cell_normals[ind[0]]
                th = []
                for _loc in self.local_normals.T:
                    th.append(angle_between(_loc, v2))
                closest_orient = self.local_normals.T[np.argmin(th)]

                # find orientation between box orientation and cell normal
                f = self.mesh.cell_normals[ind[0]]
                t = closest_orient  # + np.random.random(3) / 1e20
                if self.toggle == "impact":
                    closest_orient = self.local_normals.T[2]  # always Z axis
                    t = closest_orient  # + np.random.random(3) / 1e20
                    f = -1 * self.mesh.cell_normals[ind[0]]
                    if np.dot(f, camera_direction) < 0:
                        f = -1 * f
                else:
                    if np.dot(f, camera_direction) > 0:
                        f = -1 * f

                # push box 0.5 away from the normal
                if self.snap_outward:
                    point = points + f / 2 * self.size
                else:
                    point = points

                # define rotational matrix to allign with the surface normal
                # if 't' not in vars() or  'f' not in vars():
                #     rot = rotation_matrix_from_vectors(
                #        np.asarray([0.,0.,1.]).T, np.asarray([0.,0.,1.]).T
                #     )
                # else:
                rot = rotation_matrix_from_vectors(t, f)
        # move everything to a new location
        _new = point - self.box.center_of_mass()

        # snaps to the mesh and moves point widgets to the new location
        if snap:
            # translates
            for item in self.objects:
                item.translate(_new, inplace=True)

            self.p.sphere_widgets[self.N + 0].SetCenter(point)
            t_new = self.box.center_of_mass()

            self.local_widgets = self.local_widgets @ rot.T
            for k in range(3):
                self.p.sphere_widgets[self.N + k + 1].SetCenter(
                    self.local_widgets[k + 1, :] + t_new
                )

            # orient the local csys of accelerometer with the new rotation
            self.local_orientation = rot @ (self.local_orientation)
            self.local_normals = rot @ self.local_normals
            self.local_rays = rot @ self.local_rays
            # rotate everything within accelerometer
            for item in self.objects:
                item.points = (rot @ (item.points - t_new).T).T + t_new
            if (
                not self.toggle == "impact"
                and self.acc_normal is not None
                and f is not None
            ):
                rot2 = rotation_matrix_from_vectors(
                    self.local_normals.T[self.acc_normal], f
                )
                # orient the local csys of accelerometer with the new rotation
                self.local_orientation = rot2 @ (self.local_orientation)
                self.local_normals = rot2 @ self.local_normals
                self.local_rays = rot2 @ self.local_rays
                self.local_widgets = self.local_widgets @ rot2.T
                for k in range(3):
                    self.p.sphere_widgets[self.N + k + 1].SetCenter(
                        self.local_widgets[k + 1, :] + t_new
                    )
                # rotate everything within accelerometer
                for item in self.objects:
                    item.points = (rot2 @ (item.points - t_new).T).T + t_new

        else:
            for item in self.objects:
                item.translate(_new, inplace=True)
            for i in range(4):
                self.p.sphere_widgets[self.N + i].SetCenter(
                    _new
                    + np.asarray(self.p.sphere_widgets[self.N + i].GetCenter())
                )

    def callback(self, point, i):
        """
        A callback function for interaction with a sphere-widget.

        :param point: Point in 3D space
        :type point: array(float)
        :param i: point ID
        :type: int
        """
        # 3D translation in space
        if i == 0:
            self.translate(point, snap=True)

        # 3D rotation in space
        else:
            if self.turn_on:
                # get the center of acc
                _new = self.box.center_of_mass()
                _vec1 = np.asarray(point - _new)
                _vec2 = np.asarray(self.local_widgets[i, :])

                # define the rotational matrix based on angle of rotation
                # theta = angle(_vec1, _vec2)
                theta = -1 * angle(_vec1, _vec2)

                # overwrite calculated rotational angle
                if self.fixed_theta != None:
                    theta = self.fixed_theta * (np.pi / 180)

                # rot = M(self.local_orientation[i - 1, :], theta)
                rot = rotation_matrix(_vec2, theta)

                # rotate everything within accelerometer
                for item in self.objects:
                    item.points = (rot @ (item.points - _new).T).T + _new

                # orient the local csys of accelerometer with the new rotation
                self.local_orientation = rot @ (self.local_orientation)
                self.local_normals = rot @ self.local_normals
                self.local_rays = rot @ self.local_rays

                # position all widgets to the new position
                for k in range(4):
                    self.local_widgets[k, :] = rot @ self.local_widgets[k, :]
                    self.p.sphere_widgets[self.N + k].SetCenter(
                        self.local_widgets[k, :] + _new
                    )
