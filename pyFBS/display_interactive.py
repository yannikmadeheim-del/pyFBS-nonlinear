import pyvista as pv
import numpy as np
import numpy as np

points = np.array([[0, 0, 0],
                   [0, 0.5, 0.5],
                   [0.5, 0, 0.5],
                   [0.5, 0.5, 0]])


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class Accelerometer():
    def __init__(self, p, N):

        # accelerometer
        self.box = pv.Box()
        self.box.translate([1, 1, 1])
        self.box.points /= 2

        p.add_mesh(self.box, opacity=0.3, show_edges=True, color="#8c8c8c")

        ray_x = pv.Line([0, 0, 0], [1, 0, 0])
        p.add_mesh(ray_x, color="r", line_width=3)
        ray_y = pv.Line([0, 0, 0], [0, 1, 0])
        p.add_mesh(ray_y, color="g", line_width=3)
        ray_z = pv.Line([0, 0, 0], [0, 0, 1])
        p.add_mesh(ray_z, color="b", line_width=3)

        self.accelerometer = [self.box, ray_x, ray_y, ray_z]
        self.N = int(N * 4)

    def callback(self, point, i):
        print(point,i)
        # 3D translation in space
        if i == 0:

            _new = point - self.box.center_of_mass() + [0.5, 0.5, 0.5]

            for item in self.accelerometer:
                item.translate(_new)

            for i in range(3):
                p.sphere_widgets[self.N + 1 + i].SetCenter(
                    _new + np.asarray(p.sphere_widgets[self.N + 1 + i].GetCenter()))

        else:
            _new = self.box.center_of_mass() - [0.5, 0.5, 0.5]

            _vec1 = np.asarray(point - _new)
            _vec2 = np.asarray(points[i, :])

            #if i == 3:
            #    print(_vec1[[0, 1]])
            #    print(i, angle_between(_vec1[[0, 1]], _vec2[[0, 1]]) * 180 / np.pi)

            for k in range(3):
                p.sphere_widgets[self.N + 1 + k].SetCenter(_new + points[1 + k, :])

    def translate(self, point):
        _new = point - self.box.center_of_mass() + [0.5, 0.5, 0.5]
        for item in self.accelerometer:
            item.translate(_new)

        for i in range(4):
            p.sphere_widgets[self.N + i].SetCenter(_new + np.asarray(p.sphere_widgets[self.N + i].GetCenter()))


if __name__ == '__main__':

    p = pv.Plotter()
    p.background_color = "#D4D4D4"

    for i in range(2):
        _gg = Accelerometer(p, i)
        p.add_sphere_widget(_gg.callback, center=points, color=["k", "r", "g", "b"], radius=0.05)
        _gg.translate(np.random.random(3) * 10)

    p.show()
