import pyvista as pv
import pandas as pd
from time import time,sleep
from PyQt5.QtWidgets import QAction
from PyQt5 import  QtGui
import imageio
from pyFBS.utility import *
import keyboard as kb
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import os

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"

BACKGROUND = "#FFFFFF"


class view3D():
    """
    A 3D visualization tool for the pyFBS. The units of the 3D display are in milimeters.

    :param show_origin: Display the CSYS in origin
    :type show_origin: bool
    """
    def __init__(self,show_origin = True,show_axes = True,title = None,**kwargs):
        self.plot = pv.BackgroundPlotter(show = True,**kwargs)

        if title != None:
            self.plot.app_window.setWindowTitle("pyFBS - " + str(title))
        else:
            self.plot.app_window.setWindowTitle("pyFBS ")



        icon = str(Path(__file__).parents[1]) + os.sep + "data" + os.sep + "icon.ico"
        self.plot.app_window.setWindowIcon(QtGui.QIcon(icon))
        self.plot.background_color = BACKGROUND
        #self.plot.enable_parallel_projection()

        if show_origin:
            self.add_csys([0,0,0])

        if show_axes:
            self.plot.add_axes(labels_off=True)

        # Static Variables
        self.global_acc = []
        self.acc_visible = False

        self.global_imp = []
        self.imp_visible = False

        self.global_chn = []
        self.imp_visible = False

        self.global_vps = []
        self.vps_visible = False

        self.global_labels = []
        self.labels_visible = False

        self.name = None
        self.name_ev = None

        # Toolbars
        self.show_hide_toolbar = self.plot.app_window.addToolBar('Show/hide Actors')
        self.animate_toolbar = self.plot.app_window.addToolBar('Animate Modeshape')

        self.displayed_bodies = []

        self.obj_animation = None
        self.modeshape_animation = None

        self.take_gif = False
        self.gif_dir = "sample.gif"

        #
        self.all_accs_dynamic = []
        self.all_imps_dynamic = []
        self.all_vps_dynamic = []

    def add_modeshape(self,dict_shape,run_animation = False,add_note = False):
        """
        Add a modeshape animation to the 3D view.

        :param dict_animation: Modeshape animation information.
        :type dict_animation: dict
        :param run_animation: Run animation at start.
        :type run_animation: bool
        :param add_note: Add a note to a corner.
        :type add_note: bool
        """
        if self.modeshape_animation == None:
            self.add_action(self.animate_toolbar, "Animate modeshape", self.animate_modeshape)

        self.modeshape_animation = dict_shape

        if add_note:
            _freq = self.modeshape_animation["freq"]
            _damp = self.modeshape_animation["damp"]
            _mcf = self.modeshape_animation["mcf"]

            self.plot.add_text("Frequency = %4.1f Hz\nDamping = %4.3f%%\nMCF = %4.1f%%" % (_freq,_damp,_mcf),
                                 position='upper_right', font_size=10, color="k", font="times", name="Mode")

        if run_animation:
            self.animate_modeshape()

    def animate_modeshape(self):
        """
        Animate modeshape from *add_modeshape* function.
        """
        frameperiod = 1.0 / self.modeshape_animation["fps"]

        now = time()
        nextframe = now + frameperiod

        ann = self.modeshape_animation["animation_pts"]

        if self.take_gif:
            self.plot.open_gif(self.gif_dir)

        if self.modeshape_animation["scalars"]:
            set_lim = np.sqrt(np.mean(ann ** 2, axis=0))
            self.plot.update_scalar_bar_range(clim=[np.min(set_lim), np.max(set_lim)])


        for i in range(ann.shape[2]):
            add_val = ann[:, :, i]

            self.plot.update_coordinates(self.modeshape_animation["or_pts"] + add_val, mesh=self.modeshape_animation["mesh"],render = False)
            if self.modeshape_animation["scalars"]:
                self.plot.update_scalars(np.sqrt(np.mean(add_val ** 2, axis=1)).reshape(self.modeshape_animation["or_pts"].shape[0]), mesh=self.modeshape_animation["mesh"] ,render = False)

            self.plot.render()
            if self.take_gif:
                self.plot.write_frame()

            while now < nextframe:
                sleep(nextframe - now)
                now = time()
            nextframe += frameperiod

        if self.take_gif:
            gif = imageio.mimread(self.gif_dir)
            imageio.mimsave(self.gif_dir, gif, fps=30)

    def clear_modeshape(self):
        self.plot.update_coordinates(self.modeshape_animation["or_pts"], mesh=self.modeshape_animation["mesh"],render = True)
        self.plot.update_scalars(np.zeros(self.modeshape_animation["or_pts"].shape[0]), mesh=self.modeshape_animation["mesh"] ,render = False)
        self.plot.update_scalar_bar_range(clim=[0,100])


    def add_objects_animation(self,dict_animation,run_animation = False,add_note = False):
        """
        Add an object animation to the 3D view.

        :param dict_animation: Object animation information.
        :type dict_animation: dict
        :param run_animation: Run animation at start.
        :type run_animation: bool
        :param add_note: Add a note to the corner.
        :type add_note: bool
        """
        if self.obj_animation == None:
            self.add_action(self.animate_toolbar, "Animate objects", self.animate_objects)

        self.obj_animation = dict_animation

        if add_note:
            _text = "Frequency = %4.1f Hz" % (self.obj_animation["freq"])
            self.plot.add_text(_text, position='upper_right', font_size=10, color="k", font="times", name="Mode")

        if run_animation:
            self.animate_objects()

    def animate_objects(self):
        """
        Animate objects from *add_objects_animation* function.
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

                    self.plot.update_coordinates(_pts + loc, mesh = _mesh,render = False)

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

    def add_action(self,toolbar, key, function):
        """
        Connects a toolbar button with a certain function

        :param toolbar: Toolbar.
        :param key: Name of the toolbar.
        :param function: Function to connect to.
        """
        action = QAction(key, self.plot.app_window)
        action.triggered.connect(function)
        toolbar.addAction(action)


    def add_csys(self,position = [0,0,0], size = 10):
        """
        Adds a coordinate system at a certain position.

        :param position: Position of the CSYS [x,y,z]
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

    def add_stl(self, stl_path, name = "model", **kwargs):
        """
        Adds a mesh to 3D view from .stl file.

        :param stl_path: Path to the .stl file
        :type stl_path: str
        :param color: Color of the mesh
        :type color: str, optional
        """
        mesh = pv.PolyData(stl_path)
        actor = self.plot.add_mesh(mesh,name = name,**kwargs)
        self.displayed_bodies.append([name,actor])

        return mesh


    def add_impact(self, position, direction, size = 10, color = RED, **kwargs):
        """
        Adds an impact to 3D view.

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
        arrow.translate(-1*np.asarray(direction))
        arrow.points *= size
        arrow.translate(np.asarray(position))
        imp_actor = self.plot.add_mesh(arrow, color=color,reset_camera = False, **kwargs)

        return arrow,imp_actor


    def add_channel(self, position, direction, size = 10, color = BLUE,**kwargs):
        """
        Adds a channel to 3D view.

        :param position: Position of the channel [x,y,z]
        :type position: list
        :param direction: Direction of the channel [dx,dy,dz]
        :type direction: list
        :param size: Size of the channel
        :type size: float, optional
        :param color: Color of the channel
        :type color: str, optional
        """

        arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=direction)
        arrow.points *= size
        arrow.points += np.asarray(position)
        chn_actor = self.plot.add_mesh(arrow, color=color,reset_camera =False, **kwargs)

        return arrow,chn_actor


    def create_accelerometer(self,position,orientation,size = 10):
        """
        Create a 3D model of accelerometer.

        :param position: Position of the accelerometer [x,y,z]
        :type position: list
        :param orientation: Orientation of the accelerometer [ax,ay,az]
        :type orientation: list
        :param size: Size of the accelerometer
        :type size: float, optional

        :return: Accelerometer object
        """

        box = pv.Box(bounds=(-size/2, size/2, -size/2, size/2, -size/2, size/2))
        cable = pv.Cylinder(center=(-size/2-size/8,0,0),direction = (-1,0,0),radius = size/5,height = size/4)

        ray_x = pv.Line(np.asarray([0, 0, 0]) - size / 2, np.asarray([size, 0, 0]) - size / 2)
        ray_y = pv.Line(np.asarray([0, 0, 0]) - size / 2, np.asarray([0, size, 0]) - size / 2)
        ray_z = pv.Line(np.asarray([0, 0, 0]) - size / 2, np.asarray([0, 0, size]) - size / 2)

        accelerometer = [box, cable,ray_x,ray_y,ray_z]
        _new = position

        r = R.from_euler('xyz', [orientation[0], orientation[1], orientation[2]], degrees=True)
        rot = r.as_matrix()

        for item in accelerometer:
            item.points = (rot@item.points.T).T

            item.translate(_new)

        return accelerometer

    def add_accelerometer(self, acc):
        """
        Adds an accelerometer to 3D view.

        :param acc: Accelerometer object
        """
        acc_1 = self.plot.add_mesh(acc[0], opacity=0.5, show_edges=True, color="#8c8c8c", reset_camera =False)
        acc_2 = self.plot.add_mesh(acc[1], opacity=0.5, show_edges=False, color="#8c8c8c", reset_camera =False)
        acc_3 = self.plot.add_mesh(acc[2], color=RED, line_width=5, reset_camera =False)
        acc_4 = self.plot.add_mesh(acc[3], color=GREEN, line_width=5, reset_camera =False)
        acc_5 = self.plot.add_mesh(acc[4], color=BLUE, line_width=5, reset_camera =False)

        return [acc_1,acc_2,acc_3,acc_4,acc_5]


    def add_vp(self,position,size = 10,color = GREEN,**kwargs):
        """
        Adds a virtual point to 3D view.

        :param position: Position of the virtual point [x,y,z]
        :type position: list
        :param size: Size of the virtual point
        :type size: float, optional
        :param color: Color of the virtual point
        :type color: str, optional
        """
        sphere = pv.Sphere(radius = size, center = position)
        vp_actor = self.plot.add_mesh(sphere, color=color,**kwargs,reset_camera= False)
        return sphere,vp_actor


    def acc_callback(self,point, orientation = None):
        """
        Description

        :param point:
        :return:
        """
        i = int(len(self.all_accs_dynamic)+len(self.all_imps_dynamic)+len(self.all_vps_dynamic))
        size = 10
        if orientation == None:
            acc = self.create_accelerometer([size/2, size/2, size/2], [0, 0, 0], size=size)
            rot = np.diag([1]*3)
        else:
            acc = self.create_accelerometer([size/2, size/2, size/2], orientation, size=size)
            r = R.from_euler('xyz', orientation, degrees=True)
            rot = r.as_matrix()
        self.add_accelerometer(acc)
        _gg = DynamicPosition(acc, self.plot, i, mesh=self.mesh, size=10,rot = rot)
        self.plot.add_sphere_widget(_gg.callback, center=_gg.points, color=["k", "r", "g", "b"], radius=10 / 15)
        _gg.translate(point)
        _gg.turn_on = True
        self.all_accs_dynamic.append(_gg)

    def add_acc_dynamic(self, mesh, predefined=None):
        """
        Description

        :param mesh:
        :param predefined:
        :return:
        """
        self.mesh = mesh

        if isinstance(predefined, pd.DataFrame):
            for i, row in predefined.iterrows():
                point = [row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000]
                orientation = [row["Orientation_1"], row["Orientation_2"], row["Orientation_3"]]
                self.acc_callback(point, orientation=orientation)

        self.plot.enable_point_picking(callback=self.acc_callback, color="r", show_message="", show_point=False)
        self.plot.add_text("Press P too add an accelerometer (hold down letter T to disable snapping to mesh).",
                           font_size=10, color="k", font="times", name="text")

    def imp_callback(self,point, direction = None):
        """
        Description

        :param point:
        :return:
        """
        i = int(len(self.all_accs_dynamic)+len(self.all_imps_dynamic)+len(self.all_vps_dynamic))
        size = 10
        if direction == None:
            imp, _ = self.add_impact([size/2, size/2, size/2], [0, 0, 1], size=10)
            rot = np.diag([1]*3)
        else:
            #imp, _ = self.add_impact([size/2, size/2, size/2],[0, 0, 1], size=10)
            imp, _ = self.add_impact([size/2, size/2, size/2],direction, size=10)
            # somethin wrong here o.O

            #rot = np.diag([1]*3)
            rot = rotation_matrix_from_vectors(direction,[0, 0, 1]).T

        _gg = DynamicPosition([imp], self.plot, i, mesh=self.mesh, size=10,rot = rot,snap_outward = False)
        self.plot.add_sphere_widget(_gg.callback, center=_gg.points, color=["k", "r", "g", "b"], radius=10 / 15)
        _gg.translate(point)
        _gg.turn_on = True
        self.all_imps_dynamic.append(_gg)

    def add_imp_dynamic(self, mesh, predefined=None):
        """
        Description

        :param mesh:
        :param predefined:
        :return:
        """
        self.mesh = mesh

        if isinstance(predefined, pd.DataFrame):
            for i, row in predefined.iterrows():
                point = [row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000]
                direction = [row["Direction_1"], row["Direction_2"], row["Direction_3"]]
                self.imp_callback(point, direction=direction)

        self.plot.enable_point_picking(callback=self.imp_callback, color="r", show_message="", show_point=False)
        self.plot.add_text("Press P too add an impact (hold down letter t to disable snapping to mesh).", font_size = 10,color = "k",font  = "times",name = "text")

    def vp_callback(self,point):
        """
        Description

        :param point:
        :return:
        """
        i = int(len(self.all_accs_dynamic)+len(self.all_imps_dynamic)+len(self.all_vps_dynamic))
        size = 4

        acc, _ = self.add_vp([size/2, size/2, size/2], size=size, opacity=.1)

        _gg = DynamicPosition([acc], self.plot, i, mesh=self.mesh, size=4, snap_outward=False)
        self.plot.add_sphere_widget(_gg.callback, center=_gg.points, color=["k", "r", "g", "b"], radius=4 / 15)

        _gg.turn_on = True
        _gg.translate(point)

        self.all_vps_dynamic.append(_gg)

    def add_vp_dynamic(self, mesh, predefined=None):
        """
        Description

        :param mesh:
        :param predefined:
        :return:
        """
        self.mesh = mesh

        if isinstance(predefined, pd.DataFrame):

            x = predefined["Position_1"].unique()
            y = predefined["Position_2"].unique()
            z = predefined["Position_3"].unique()
            position = np.asarray([x, y, z]).T
            position *= 1000
            for pos in position:
                self.vp_callback(pos)

        self.plot.enable_point_picking(callback=self.vp_callback, color="r", show_message="", show_point=False)
        self.plot.add_text("Press P too add a VP (hold down letter T to disable snapping to mesh).", font_size=10, color="k", font="times", name="text")

    def get_imp_data(self):
        """
        Description

        :return:
        """
        columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                         "Grouping", "Position_1", "Position_2", "Position_3", "Direction_1", "Direction_2",
                         "Direction_3"]
        df = pd.DataFrame(columns=columns_chann)

        for i, _imp in enumerate(self.all_imps_dynamic):
            pos, _dir = _imp.get_pos_orient(one_dir=2)

            data_chn = np.asarray([["Impact " + str(1 + i), None, None, None, None, None, None, None, None, pos[0],
                                    pos[1], pos[2], _dir[0], _dir[1], _dir[2]]])

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = df.append(df_row, ignore_index=True)

        return df

    def get_acc_data(self):
        """
        Description

        :return:
        """
        columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                         "Grouping", "Position_1", "Position_2", "Position_3", "Orientation_1", "Orientation_2",
                         "Orientation_3"]
        df = pd.DataFrame(columns=columns_chann)

        for i, _acc in enumerate(self.all_accs_dynamic):
            pos, euler_dir = _acc.get_pos_orient(euler_angles=True)
            data_chn = np.asarray([["Sensor " + str(1 + i), None, None, None, None, None, None, None, None, pos[0],pos[1],pos[2],euler_dir[0],euler_dir[1],euler_dir[2] ]])

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = df.append(df_row, ignore_index=True)

        return df

    def get_vp_data(self):
        """
        Description

        :return:
        """
        columns_chann = ["Name", "Description", "Type", "DirectionLabel", "Quantity", "Unit", "Component", "NodeNumber",
                         "Grouping", "Position_1", "Position_2", "Position_3", "Orientation_1", "Orientation_2",
                         "Orientation_3"]
        df = pd.DataFrame(columns=columns_chann)

        for i, _acc in enumerate(self.all_vps_dynamic):
            pos, euler_dir = _acc.get_pos_orient(euler_angles=True)
            data_chn = np.asarray([["VP " + str(1 + i), None, None, None, None, None, None, None, None, pos[0],pos[1],pos[2],euler_dir[0],euler_dir[1],euler_dir[2] ]])

            df_row = pd.DataFrame(data=data_chn, columns=columns_chann)
            df = df.append(df_row, ignore_index=True)

        return df

    def show_acc(self,df,size = 10,overwrite = True):
        """
        Adds accelerometers from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the accelerometers
        :type df: pd.DataFrame
        """
        if self.global_acc != []:
            if overwrite:
                self.acc_visible = True
                self.show_hide_accelerometers()
                self.global_acc = []
            else:
                pass
        else:
            self.add_action(self.show_hide_toolbar, "Sensors", self.show_hide_accelerometers)

        for i, row in df.iterrows():

            acc_mesh = self.create_accelerometer((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                                         (row["Orientation_1"], row["Orientation_2"], row["Orientation_3"]),size = size)

            acc_actor = self.add_accelerometer(acc_mesh)
            acc_pts = []

            for i in range(5):
                acc_pts.append(acc_mesh[i].points.copy())

            self.global_acc.append([acc_pts,acc_mesh,acc_actor])
        self.acc_visible = True



    def show_imp(self,df,color = RED,overwrite = True,**kwargs):
        """
        Adds impacts from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the impacts
        :type df: pd.DataFrame
        """

        if self.global_imp != []:
            if overwrite:
                self.imp_visible = True
                self.show_hide_impacts()
                self.global_imp = []
            else:
                pass
        else:
            self.add_action(self.show_hide_toolbar, "Impacts", self.show_hide_impacts)

        for i, row in df.iterrows():
            imp_mesh,imp_actor = self.add_impact((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                         (row["Direction_1"], row["Direction_2"], row["Direction_3"]),color = color, **kwargs)
            self.global_imp.append([imp_mesh,imp_actor])

        self.imp_visible = True


    def show_chn(self,df,color = BLUE,overwrite = True,**kwargs):
        """
        Adds channels from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the channels
        :type df: pd.DataFrame
        """

        if self.global_chn != []:
            if overwrite:
                self.chn_visible = True
                self.show_hide_channels()
                self.global_chn = []
            else:
                pass
        else:
            self.add_action(self.show_hide_toolbar, "Channels", self.show_hide_channels)

        for i, row in df.iterrows():
            chn_mesh,chn_actor = self.add_channel((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                          (row["Direction_1"], row["Direction_2"], row["Direction_3"]),color = color,**kwargs)
            self.global_chn.append([chn_mesh,chn_actor])

        self.chn_visible = True


    def show_vp(self,df,color = GREEN,overwrite = True,size = 10,**kwargs):
        """
        Adds virtual points from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the virtual points
        :type df: pd.DataFrame
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

        x = df["Position_1"].unique()
        y = df["Position_2"].unique()
        z = df["Position_3"].unique()
        position = np.asarray([x, y, z]).T
        position *= 1000
        vp_mesh,vp_actor = self.add_vp(position,color = color,size = size,**kwargs)
        self.global_vps.append([vp_mesh, vp_actor])
        self.vps_visible = True


    def label_acc(self,df,name = "Accelerometers",**kwargs):
        """
        Adds labels to accelerometers from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the accelerometers
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        if self.global_labels == []:
            self.add_action(self.show_hide_toolbar, "Clear Labels", self.clear_labels)

        positions = []
        labels = []
        for i, row in df.iterrows():
            positions.append([row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000])
            labels.append(row["Name"])

        self.plot.add_point_labels(positions, labels, font_size=12,name = name,shape_opacity=.5,show_points=False,**kwargs)
        self.global_labels.append([[positions, labels], name])
        self.labels_visible = True

    def label_imp(self,df,name = "Impacts",**kwargs):
        """
        Adds labels to impacts from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the impacts
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        if self.global_labels == []:
            self.add_action(self.show_hide_toolbar, "Clear Labels", self.clear_labels)

        positions = []
        labels = []
        for i, row in df.iterrows():

            positions.append([row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000])
            labels.append(row["Name"])

        self.plot.add_point_labels(positions, labels, font_size=12,name = name,shape_color = RED,font_family = "times",shape_opacity=0.5,show_points=False,**kwargs)
        self.global_labels.append([[positions, labels], name])
        self.labels_visible = True

    def label_chn(self,df,name = "Channels",size = 10,**kwargs):
        """
        Adds labels to channels from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the channels
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        if self.global_labels == []:
            self.add_action(self.show_hide_toolbar, "Clear Labels", self.clear_labels)

        positions = []
        labels = []
        for i, row in df.iterrows():
            x = row["Direction_1"]*size
            y = row["Direction_2"]*size
            z = row["Direction_3"]*size

            positions.append([row["Position_1"]*1000+x, row["Position_2"]*1000+y, row["Position_3"]*1000+z])
            labels.append(row["Name"])

        self.plot.add_point_labels(positions, labels, font_size=12, name=name, shape_color=BLUE, font_family = "times",shape_opacity=0.5,show_points=False,**kwargs)
        self.global_labels.append([[positions,labels],name])
        self.labels_visible = True

    def label_vp(self,df,name = "VPs",**kwargs):
        """
        Adds labels to virtual point from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the virtual points
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        if self.global_labels == []:
            self.add_action(self.show_hide_toolbar, "Clear Labels", self.clear_labels)

        x = df["Position_1"].unique()
        y = df["Position_2"].unique()
        z = df["Position_3"].unique()
        position = np.asarray([x, y, z]).T
        position *= 1000

        L = df["Grouping"].unique()

        self.plot.add_point_labels(position, L, font_size=12,name = name,font_family = "times",shape_opacity=0.5,shape_color = GREEN,show_points=False,**kwargs)
        self.global_labels.append([[position, L], name])
        self.labels_visible = True


    def show_hide_accelerometers(self):
        """
        Show or hide all the accelerometers in the 3D view.
        """
        if self.acc_visible == False:
            for _acc in self.global_acc:
                for item in _acc[2]:
                    self.plot.add_actor(item,reset_camera =False)
            self.acc_visible = True

        else:
            for _acc in self.global_acc:
                for item in _acc[2]:
                    self.plot.remove_actor(item, reset_camera=False)#,render = False)

            self.acc_visible = False

    def show_hide_impacts(self):
        """
        Show or hide all the impacts in the 3D view.
        """
        if self.imp_visible == False:
            for _imp in self.global_imp:
                self.plot.add_actor(_imp[1],reset_camera =False)
            self.imp_visible = True

        else:
            for _imp in self.global_imp:
                self.plot.remove_actor(_imp[1],reset_camera =False)
            self.imp_visible = False

    def show_hide_channels(self):
        """
        Show or hide all the channels in the 3D view.
        """
        if self.chn_visible == False:
            for _chn in self.global_chn:
                self.plot.add_actor(_chn[1],reset_camera =False)
            self.chn_visible = True

        else:
            for _chn in self.global_chn:
                self.plot.remove_actor(_chn[1],reset_camera =False)

            self.chn_visible = False


    def show_hide_vps(self):
        """
        Show or hide all the VPs in the 3D view.
        """
        if self.vps_visible == False:
            for _vp in self.global_vps:
                self.plot.add_actor(_vp[1],reset_camera =False)
            self.vps_visible = True

        else:
            for _vp in self.global_vps:
                self.plot.remove_actor(_vp[1],reset_camera =False)

            self.vps_visible = False

    def clear_labels(self):
        """
        Clear all labels in the 3D view.
        """
        for _label in self.global_labels:
            self.plot.remove_actor(_label[1],reset_camera =False)

        self.labels_visible = False


class DynamicPosition():
    """
    Description

    """

    def __init__(self, objects, p, N, mesh=None, snap_outward=True, size=1, rot = np.diag([1]*3)):

        self.size = size
        self.points = np.array([[size / 2, size / 2, size / 2],
                                [0, size / 2, size / 2],
                                [size / 2, 0, size / 2],
                                [size / 2, size / 2, 0]])

        # Creates a bounding box
        self.box = pv.Box((-size, size, -size, size, -size, size))
        self.box.translate([size, size, size])
        self.box.points /= 2

        # defines the objects
        objects.insert(0, self.box)
        self.objects = objects

        # get the number of dynamic stuff in the display window
        self.N = int(N * 4)

        # local orientation of the bounding box/object
        self.local_orientation = np.asarray([[1, 0, 0],
                                             [0, 1, 0],
                                             [0, 0, 1]])

        # local positions of the point widgets
        self.local_widgets = np.array([[0.0, 0.0, 0.0],
                                       [-0.5, 0, 0],
                                       [0, -0.5, 0],
                                       [0, 0, -0.5]]) * size

        # local normals on which the snapping happens
        self.local_normals = np.asarray([[1, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 1],
                                         [-1, 0, 0],
                                         [0, -1, 0],
                                         [0, 0, -1]]).T

        # ray_size on which the snapping to the mesh happens
        ray_size = 4 * size
        self.local_rays = np.asarray([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1],
                                      [-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, -1]]).T * ray_size

        self.local_orientation = (rot @ (self.local_orientation))

        self.local_widgets = (rot @ (self.local_widgets).T).T
        self.local_normals = rot @ self.local_normals
        self.local_rays = rot @ self.local_rays

        # computes mesh normals
        self.mesh = mesh
        self.mesh.compute_normals(auto_orient_normals=True, inplace=True)

        # disables the rotation of the object
        self.turn_on = False

        self.snap_outward = snap_outward

        # define display
        self.p = p

    def get_pos_orient(self, euler_angles=False, one_dir=None, eps=1e-10):
        """
        Description

        :param euler_angles:
        :param one_dir:
        :param eps:
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
            orientation = r[one_dir,:]

        # whole orientation
        else:
            orientation = self.local_orientation

        # set to zero for very small numbers
        orientation[np.abs(orientation) < eps] = 0

        return position/1000, orientation

    def translate(self, point, snap=False):
        """
        Description

        :param point:
        :param snap:
        :return:
        """

        # definest rays to find intersection with the supplied mesh
        point1x = point + self.local_rays[:, 0]
        point2x = point + self.local_rays[:, 3]

        point1y = point + self.local_rays[:, 1]
        point2y = point + self.local_rays[:, 4]

        point1z = point + self.local_rays[:, 2]
        point2z = point + self.local_rays[:, 5]

        # performs ray trace in three direction
        points_x, ind_x = self.mesh.ray_trace(point1x, point2x)
        points_y, ind_y = self.mesh.ray_trace(point1y, point2y)
        points_z, ind_z = self.mesh.ray_trace(point1z, point2z)

        # stacks all the ray intersections
        points = np.vstack([points_x, points_y, points_z])
        ind = np.hstack([ind_x, ind_y, ind_z])

        # default option - no rotation
        rot = np.diag([1, 1, 1])

        # if there is an intersection and if "t" is not pressed go forward
        if points.size != 0 and not (kb.is_pressed('t')):
            list_ind = []
            # go through all the intersections
            for i in range(len(points)):
                _point = points[i]
                p1 = _point
                p2 = point
                gg = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))

                list_ind.append(gg)

            # find the closest to the box center
            _sel = np.argmin(list_ind)

            # find the nearest normal
            v2 = self.mesh.cell_normals[int(ind[_sel])]
            th = []
            for _loc in self.local_normals.T:
                th.append(angle_between(_loc, v2))
            closest_orient = self.local_normals.T[np.argmin(th)]

            # find orientation between box orientation and cell normal
            f = self.mesh.cell_normals[int(ind[_sel])]
            t = closest_orient + np.random.random(3) / 1e20

            # push box 0.5 away from the normal
            if self.snap_outward:
                point = points[_sel] + f / 2 * self.size
            else:
                point = points[_sel]

            # define rotational matrix to allign with the surface normal
            rot = rotation_matrix_from_vectors(t, f)

        # move everything to a new location
        _new = point - self.box.center_of_mass()

        # snaps to the mesh and moves point widgets to the new location
        if snap:
            # translates
            for item in self.objects:
                item.translate(_new)

            self.p.sphere_widgets[self.N + 0].SetCenter(point)
            t_new = self.box.center_of_mass()

            for k in range(3):
                self.local_widgets[k + 1, :] = rot @ self.local_widgets[k + 1, :]
                self.p.sphere_widgets[self.N + k + 1].SetCenter(self.local_widgets[k + 1, :] + t_new)

            # orient the local csys of accelerometer with the new rotation
            self.local_orientation = (rot @ (self.local_orientation))
            self.local_normals = rot @ self.local_normals
            self.local_rays = rot @ self.local_rays

            # rotate everything within accelerometer
            for item in self.objects:
                item.points = (rot @ (item.points - t_new).T).T + t_new

        else:
            for item in self.objects:
                item.translate(_new)
            for i in range(4):
                self.p.sphere_widgets[self.N + i].SetCenter(_new + np.asarray(self.p.sphere_widgets[self.N + i].GetCenter()))

    def callback(self, point, i):
        """
        Description

        :param point:
        :param i:
        :return:
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
                _vec2 = (np.asarray(self.local_widgets[i, :]))

                # define the rotational matrix based on angle of rotation
                theta = angle(_vec1, _vec2)
                rot = M(self.local_orientation[i - 1, :], theta)

                # rotate everything within accelerometer
                for item in self.objects:
                    item.points = (rot @ (item.points - _new).T).T + _new

                    # orient the local csys of accelerometer with the new rotation
                self.local_orientation = (rot @ (self.local_orientation))
                self.local_normals = rot @ self.local_normals
                self.local_rays = rot @ self.local_rays

                # position all widgets to the new position
                for k in range(4):
                    self.local_widgets[k, :] = rot @ self.local_widgets[k, :]
                    self.p.sphere_widgets[self.N + k].SetCenter(self.local_widgets[k, :] + _new)