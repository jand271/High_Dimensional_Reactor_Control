
import numpy as np
import matplotlib.pyplot as plt
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.rod import ControlRod, FuelRod

if __name__ == "__main__":

    fa = FuelAssembly()

    r = 0.05

    for x in np.linspace(fa._xlim[0] + r, fa._xlim[1] - r, 10):
        for y in np.linspace(fa._ylim[0] + r, fa._ylim[1] - r, 10):
            fa.add_component(FuelRod(x, y, 0.05))

    for x in np.linspace(fa._xlim[0] + 3 * r, fa._xlim[1] - 3 * r, 9):
        for y in np.linspace(fa._ylim[0] + 3 * r, fa._ylim[1] - 3 * r, 9):
            fa.add_component(ControlRod(x, y, 0.05))

    fa.plot()
    plt.show()
