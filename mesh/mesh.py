import numpy as np
import matplotlib.pylab as plt
import openmesh as om
from fuel_assembly.cladding import Cladding


class Mesh(object):
    def __init__(self, fuel_assembly, mesh_grid):
        self._fuel_assembly = fuel_assembly
        self._mesh_grid = mesh_grid
        self._mesh_shape = self._mesh_grid[0].shape
        self._mesh = om.TriMesh()

        xs = self._mesh_grid[0].flatten()
        ys = self._mesh_grid[1].flatten()

        nvs = len(xs)
        for i in range(nvs):
            self._mesh.add_vertex([xs[i], ys[i], 0])

        n = self._mesh_shape[0]
        for i in range(nvs - n):
            if i % n != 0:
                self._mesh.add_face(
                    self._mesh.vertex_handle(i),
                    self._mesh.vertex_handle(i + n - 1),
                    self._mesh.vertex_handle(i + n))

            if (i + 1) % n != 0:
                self._mesh.add_face(
                    self._mesh.vertex_handle(i),
                    self._mesh.vertex_handle(i + n),
                    self._mesh.vertex_handle(i + 1))

        self._face_centroid_hash_map = {}
        self._face_component_hash_map = {}
        for f_indices, f in zip(self._mesh.face_vertex_indices(), self._mesh.faces()):
            (x0, y0, z0) = self._mesh.point(self._mesh.vertex_handle(f_indices[0]))
            (x1, y1, z1) = self._mesh.point(self._mesh.vertex_handle(f_indices[1]))
            (x2, y2, z2) = self._mesh.point(self._mesh.vertex_handle(f_indices[2]))

            x = np.mean((x0, x1, x2))
            y = np.mean((y0, y1, y2))

            self._face_centroid_hash_map[f] = (x, y)
            self._face_component_hash_map[f] = self._fuel_assembly.find_component(x, y)

            if self._face_component_hash_map[f] is None:
                self._face_component_hash_map[f] = Cladding()

    def plot(self):

        for f_indices in self._mesh.face_vertex_indices():
            (x0, y0, z0) = self._mesh.point(self._mesh.vertex_handle(f_indices[0]))
            (x1, y1, z1) = self._mesh.point(self._mesh.vertex_handle(f_indices[1]))
            (x2, y2, z2) = self._mesh.point(self._mesh.vertex_handle(f_indices[2]))

            plt.plot([x0, x1], [y0, y1], color='k')
            plt.plot([x1, x2], [y1, y2], color='k')
            plt.plot([x2, x0], [y2, y0], color='k')

        for point, component in zip(self._face_centroid_hash_map.values(), self._face_component_hash_map.values()):
            plt.scatter(point[0], point[1], color=component.get_plot_color())

    def write_to_mesh(self, file_name):
        om.write_mesh(file_name, self._mesh)


class Mesh2D(Mesh):
    def __init__(self, *args):
        super().__init__(*args)
