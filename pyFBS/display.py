import pyvista as pv
import numpy as np
import pandas as pd
from time import time,sleep
from PyQt5.QtWidgets import  QAction
import imageio


RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"

BACKGROUND = "#FFFFFF"


class view3D():
    """
    A 3D visualization tool for the pyFBS.

    :param show_origin: Display the CSYS in origin
    :type show_origin: bool
    """
    def __init__(self,show_origin = True):
        self.plot = pv.BackgroundPlotter(show = True,window_size = [1240,640])
        self.plot.app_window.setWindowTitle("pyFBS v1.0")
        self.plot.background_color = BACKGROUND

        if show_origin:
            self.add_csys([0,0,0])

        # Variables
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

        self.show_hide_toolbar = self.plot.app_window.addToolBar('Show/hide Actors')
        self.animate_toolbar = self.plot.app_window.addToolBar('Animate Modeshape')

        self.displayed_bodies = []

        self.obj_animation = None
        self.modeshape_animation = None

        self.take_gif = False

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

            self.plot.add_text("Frequency = %4.1f Hz\nDamping = %4.3f%%\nComplexity = %4.1f%%" % (_freq,_damp,_mcf),
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
                self.plot.update_scalars(np.sqrt(np.mean(add_val ** 2, axis=1)).reshape(self.modeshape_animation["or_pts"].shape[0]),render = False)

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
            gif = imageio.mimread(self.gif_dir)
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
        for item in accelerometer:
            item.rotate_x(orientation[0])
            item.rotate_y(orientation[1])
            item.rotate_z(orientation[2])
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
        vp_actor = self.plot.add_mesh(sphere, color=color,**kwargs)
        return sphere,vp_actor


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

        self.plot.add_point_labels(positions, labels, font_size=12,name = name,shape_opacity=0.5,**kwargs)
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

        self.plot.add_point_labels(positions, labels, font_size=12,name = name,shape_color = RED,font_family = "times",shape_opacity=0.5,**kwargs)
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

        self.plot.add_point_labels(positions, labels, font_size=12, name=name, shape_color=BLUE, font_family = "times",shape_opacity=0.5,**kwargs)
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

        self.plot.add_point_labels(position, L, font_size=12,name = name,shape_opacity=0.5,shape_color = GREEN,**kwargs)
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
