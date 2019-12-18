
import numpy as np
import matplotlib.pyplot as plt
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.rod import ControlRod, FuelRod

if __name__ == "__main__":

    r = 0.05

    fa = FuelAssembly(ylim=(-r, r))

    for x in np.linspace(fa._xlim[0] + r, fa._xlim[1] - r, 10):
        fa.add_component(FuelRod(x, 0, 0.05))

    for x in np.linspace(fa._xlim[0] + 3 * r, fa._xlim[1] - 3 * r, 5):
        fa.add_component(ControlRod(x, 0, 0.05))

    fa.plot()
    plt.show()
