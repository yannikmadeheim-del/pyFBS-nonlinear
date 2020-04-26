import pyvista as pv
import numpy as np


from numpy import cross, eye, dot
from scipy.linalg import expm, norm

def M(axis, theta):
    """
    Euler-Rodrigues formula
    """
    return expm(cross(eye(3), axis/norm(axis)*theta))

class Accelerometer():
    def __init__(self, p, N):

        # accelerometer
        self.box = pv.Box()
        #self.box.translate([1, 1, 1])
        self.box.points /= 2

        p.add_mesh(self.box, opacity=0.3, show_edges=True, color="#8c8c8c")

        ray_x = pv.Line([-0.5, -0.5, -0.5], [0.5, -0.5, -0.5])
        p.add_mesh(ray_x, color="r", line_width=3)
        ray_y = pv.Line([-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5])
        p.add_mesh(ray_y, color="g", line_width=3)
        ray_z = pv.Line([-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5])
        p.add_mesh(ray_z, color="b", line_width=3)

        self.accelerometer = [self.box, ray_x, ray_y, ray_z]
        self.N = int(N * 4)

        self.local_orientation = np.asarray([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        self.local_widgets = np.array([[0, 0, 0],
                   [-.5, 0.2, 0.2],
                   [0.2, -.5, 0.2],
                   [0.2, 0.2, -.5]])

        self.def_rot = 0



    def callback(self, point, i):
        #print(point,i)

        if i == 0:
            _new = point - self.box.center_of_mass()

            for item in self.accelerometer:
                item.translate(_new)

            for i in range(3):
                p.sphere_widgets[self.N + 1 + i].SetCenter(
                    _new + np.asarray(p.sphere_widgets[self.N + 1 + i].GetCenter()))

        # 3D rotation
        else:
            # get the center of acc
            _new = self.box.center_of_mass()

            # define the rotational matrix based on angle of rotation
            # currently self.def_rot is fixed and is not depended on the position of sphere_widget
            rot = M(self.local_orientation[i-1, :], self.def_rot * np.pi / 180)

            # rotate everything within accelerometer
            for item in self.accelerometer:
                item.points = (rot @ (item.points - _new).T).T + _new

            # orient the local csys of accelerometer with the new rotation
            self.local_orientation = (rot @ (self.local_orientation).T).T

            # position all widgets to the new position
            for k in range(4):
                self.local_widgets[k, :] = rot@self.local_widgets[k, :]
                p.sphere_widgets[self.N +  k].SetCenter(self.local_widgets[k, :]  + _new)


    def translate(self, point):
        _new = point - self.box.center_of_mass()
        for item in self.accelerometer:
            item.translate(_new)

        for k in range(4):
            p.sphere_widgets[self.N + k].SetCenter(self.local_widgets[k, :] + point)


if __name__ == '__main__':

    p = pv.Plotter()
    p.background_color = "#D4D4D4"

    size = 0.5
    arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(1, 0, 0))
    arrow.points *= size
    p.add_mesh(arrow, color="r")

    arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0, 1, 0))
    arrow.points *= size
    p.add_mesh(arrow, color="g")

    arrow = pv.Arrow(start=(0.0, 0.0, 0.0), direction=(0, 0, 1))
    arrow.points *= size
    p.add_mesh(arrow, color="b")

    for i in range(3):
        _gg = Accelerometer(p, i)
        p.add_sphere_widget(_gg.callback, center=_gg.local_widgets, color=["k", "r", "g", "b"], radius=0.05)
        _gg.translate(np.random.random(3) * 10)

        _gg.def_rot = 10

    p.reset_camera()


    p.show()
