import numpy as np
import matplotlib.pyplot as plt
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.component import UnshapedComponent
from fuel_assembly.rod import Rod
from fuel_assembly.material import Material
from fem_model.temperature_fem_model import HeatExchangerFEMModel
from fenics import *

if __name__ == "__main__":

    material = Material(thermal_conductivity=1., specific_heat_capacity=1., density=1.)

    fa = FuelAssembly(default_component=UnshapedComponent(material=material, plot_color='k'))

    nh = 8
    nc = 9
    r = 0.1
    heat_power_density = 1000

    # cooling rods
    cooling_rods = [Rod(0, 0, r, plot_color='b', material=material)]
    R = 0.6
    for t in np.linspace(0, 2 * np.pi, nc):
        if t == 0:
            continue
        cooling_rods.append(Rod(R * np.cos(t), R * np.sin(t), r, plot_color='b', material=material))

    # heating rods
    heating_rods = []
    R = 0.33
    i = 0
    for t in np.linspace(0, 2 * np.pi, nh + 1):
        i += 1
        if t == 0:
            continue
        if i % 2 == 0:
            heating_rods.append(Rod(R * np.cos(t), R * np.sin(t), r, plot_color='r', material=material))
        else:
            cooling_rods.append(Rod(R * np.cos(t), R * np.sin(t), r, plot_color='b', material=material))
            nh += -1
            nc += 1

    cool_power_density = -heat_power_density * nh / nc

    for rod in heating_rods:
        rod.set_volumetric_power_density(heat_power_density)
        fa.add_component(rod)

    for rod in cooling_rods:
        rod.set_volumetric_power_density(cool_power_density)
        fa.add_component(rod)

    q = HeatExchangerFEMModel(fa, 0.05, nx=30, ny=30)

    fa.plot()
    q.step_time()
    q._component_hash_map.plot()
    plt.title('Fuel Assembly and Mesh')
    plt.show()
    plt.clf()

    for i in range(20):
        p = plot(q.step_time())
        plt.colorbar(p, format='%.1f K')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Fuel Assembly Temperature')
        plt.pause(0.5)
        plt.clf()
