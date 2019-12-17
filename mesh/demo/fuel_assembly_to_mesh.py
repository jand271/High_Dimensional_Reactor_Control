import numpy as np
import matplotlib.pyplot as plt
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.rod import ControlRod, FuelRod
from mesh.mesh import Mesh2D

if __name__ == "__main__":

    file_path_header = 'demo_fuel_assembly'

    fa = FuelAssembly()
    r = 0.05
    for x in np.linspace(fa._xlim[0] + r, fa._xlim[1] - r, 10):
        for y in np.linspace(fa._ylim[0] + r, fa._ylim[1] - r, 10):
            fa.add_rod(FuelRod(x, y, 0.05))

    for x in np.linspace(fa._xlim[0] + 3 * r, fa._xlim[1] - 3 * r, 9):
        for y in np.linspace(fa._ylim[0] + 3 * r, fa._ylim[1] - 3 * r, 9):
            fa.add_rod(ControlRod(x, y, 0.05))

    fa.plot()
    plt.title('Geometric Fuel Assembly')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_path_header + '.png')

    plt.clf()
    mesh = Mesh2D(fa, np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5)))
    mesh.plot()
    plt.title('Meshed Fuel Assembly')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(file_path_header + '_mesh.png')

    mesh.write_to_mesh(file_path_header + '.stl')
