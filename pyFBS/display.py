import pyvista as pv
import numpy as np
import pandas as pd

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"

BACKGROUND = "#D4D4D4"

class view3D():
    """
    A 3D visualization tool for the pyFBS.

    :param behaviour: Defines the behaviour of the 3D viewer
    :type behaviour: str
    :param show_origin: Display the CSYS in origin
    :type show_origin: bool
    """
    def __init__(self, behaviour = "static",show_origin = True):
        if behaviour == "static":
            self.plot = pv.BackgroundPlotter(show = True,window_size = [600,400])
            self.plot.background_color = BACKGROUND

        if show_origin:
            self.add_csys([0,0,0])

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

        #sphere = pv.Sphere(radius = 1, center=position)
        #self.plot.add_mesh(sphere, color="black")

    def add_stl(self, stl_path, color = None):
        """
        Adds a mesh to 3D view from .stl file.

        :param stl_path: Path to the .stl file
        :type stl_path: str
        :param color: Color of the mesh
        :type color: str, optional
        """
        mesh = pv.PolyData(stl_path)
        self.plot.add_mesh(mesh,color = color)

    def add_impact(self, position, direction, size = 10, color = RED):
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

        self.plot.add_mesh(arrow, color=color)

    def add_channel(self, position, direction, size = 10, color = BLUE):
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

        self.plot.add_mesh(arrow, color=color)


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
        self.plot.add_mesh(acc[0], opacity=0.5, show_edges=True, color="#8c8c8c", pickable=False)
        self.plot.add_mesh(acc[1], opacity=0.5, show_edges=False, color="#8c8c8c", pickable=False)

        self.plot.add_mesh(acc[2], color=RED, line_width=5, pickable=False)
        self.plot.add_mesh(acc[3], color=GREEN, line_width=5, pickable=False)
        self.plot.add_mesh(acc[4], color=BLUE, line_width=5, pickable=False)

    def add_vp(self,position,size = 10,color = GREEN):
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
        self.plot.add_mesh(sphere, color=color)


    def show_acc(self,df):
        """
        Adds accelerometers from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the accelerometers
        :type df: pd.DataFrame
        """
        for i, row in df.iterrows():
            acc = self.create_accelerometer((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                                         (row["Orientation_1"], row["Orientation_2"], row["Orientation_3"]))
            self.add_accelerometer(acc)

    def show_imp(self,df):
        """
        Adds impacts from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the impacts
        :type df: pd.DataFrame
        """
        for i, row in df.iterrows():
            self.add_impact((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                         (row["Direction_1"], row["Direction_2"], row["Direction_3"]))

    def show_chn(self,df):
        """
        Adds channels from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the channels
        :type df: pd.DataFrame
        """
        for i, row in df.iterrows():
            self.add_channel((row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000),
                          (row["Direction_1"], row["Direction_2"], row["Direction_3"]))


    def show_vp(self,df):
        """
        Adds virtual points from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the virtual points
        :type df: pd.DataFrame
        """
        x = df["Position_1"].unique()
        y = df["Position_2"].unique()
        z = df["Position_3"].unique()
        position = np.asarray([x, y, z]).T
        position *= 1000
        self.add_vp(position)

    def label_acc(self,df,name = None):
        """
        Adds labels to accelerometers from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the accelerometers
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        positions = []
        labels = []
        for i, row in df.iterrows():
            positions.append([row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000])
            labels.append(row["Name"])

        self.plot.add_point_labels(positions, labels, font=12,name = name,shape_opacity=0.5)

    def label_imp(self,df,name = None):
        """
        Adds labels to impacts from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the impacts
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        positions = []
        labels = []
        for i, row in df.iterrows():

            positions.append([row["Position_1"] * 1000, row["Position_2"] * 1000, row["Position_3"] * 1000])
            labels.append(row["NodeNumber"])

        self.plot.add_point_labels(positions, labels, font=12,name = name,shape_color = RED,shape_opacity=0.5)

    def label_chn(self,df,name = None,size = 10):
        """
        Adds labels to channels from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the channels
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        positions = []
        labels = []
        for i, row in df.iterrows():
            x = row["Direction_1"]*size
            y = row["Direction_2"]*size
            z = row["Direction_3"]*size

            positions.append([row["Position_1"]*1000+x, row["Position_2"]*1000+y, row["Position_3"]*1000+z])
            labels.append(row["NodeNumber"])

        self.plot.add_point_labels(positions, labels, font=12, name=name, shape_color=BLUE, shape_opacity=0.5)

    def label_vp(self,df,name = None):
        """
        Adds labels to virtual point from the DataFrame to 3D view.

        :param df: A DataFrame containing relevant information about the virtual points
        :type df: pd.DataFrame
        :param name: Name of the label which can be used to update existing notations
        :type name: str, optional
        """
        x = df["Position_1"].unique()
        y = df["Position_2"].unique()
        z = df["Position_3"].unique()
        position = np.asarray([x, y, z]).T
        position *= 1000

        L = df["Grouping"].unique()

        self.plot.add_point_labels(position, L, font=12,name = name,shape_opacity=0.5,shape_color = GREEN)

"""
The making of interactive 3D display
import pyvista as pv
import numpy as np
import numpy as np



points = np.array([[0,0,0],
                   [0,0.5,0.5],
                   [0.5,0,0.5],
                   [0.5,0.5,0]])




def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Accelerometer():
    def __init__(self,p,N):

        # accelerometer
        self.box = pv.Box()
        self.box.translate ([1,1,1])
        self.box.points /= 2

        p.add_mesh(self.box,opacity = 0.3, show_edges  = True, color = "#8c8c8c")

        ray_x = pv.Line([0,0,0], [1,0,0])
        p.add_mesh(ray_x, color="r", line_width=3)
        ray_y = pv.Line([0,0,0], [0,1,0])
        p.add_mesh(ray_y, color="g", line_width=3)
        ray_z = pv.Line([0,0,0], [0,0,1])
        p.add_mesh(ray_z, color="b", line_width=3)

        self.accelerometer = [self.box,ray_x,ray_y,ray_z]
        self.N = int(N*4)

    def callback(self,point, i):
        # 3D translation in space
        if i == 0:

            _new = point - self.box.center_of_mass() + [0.5,0.5,0.5]

            for item in self.accelerometer:
                item.translate(_new)

            for i in range(3):
                p.sphere_widgets[self.N+1+i].SetCenter(_new + np.asarray(p.sphere_widgets[self.N+1+i].GetCenter()))

        else:
            _new = self.box.center_of_mass() - [0.5,0.5,0.5]


            _vec1 = np.asarray(point-_new)
            _vec2 = np.asarray(points[i,:])

            if i == 3:
                print(_vec1[[0,1]])
                print(i,angle_between(_vec1[[0,1]], _vec2[[0,1]])*180/np.pi)

            for k in range(3):
                p.sphere_widgets[self.N+1+k].SetCenter(_new + points[1+k,:])

                
    def translate(self,point):
        _new = point - self.box.center_of_mass() + [0.5, 0.5, 0.5]
        for item in self.accelerometer:
            item.translate(_new)

        for i in range(4):
            p.sphere_widgets[self.N  + i].SetCenter(_new + np.asarray(p.sphere_widgets[self.N + i].GetCenter()))

if __name__ == '__main__':

    p = pv.Plotter()

    for i in range(10):
        _gg = Accelerometer(p,i)
        p.add_sphere_widget(_gg.callback, center=points, color = ["k","r","g","b"],radius = 0.05)
        _gg.translate(np.random.random(3)*10)
"""

if __name__ == '__main__':
    print("cat")