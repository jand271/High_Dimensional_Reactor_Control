import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from fenics import *
from assembly_construction.component_assembly import FuelAssembly
from assembly_construction.component import UnshapedComponent
from assembly_construction.rod import Rod
from assembly_construction.material import Material
from modeling.temperature_fem_model import HeatExchangerFEMModel

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
        fa.add_component(rod, component_set='set_q_dot')

    for rod in cooling_rods:
        rod.set_volumetric_power_density(cool_power_density)
        fa.add_component(rod, component_set='controllable_q_dot')

    dt = 1
    model = HeatExchangerFEMModel(fa, dt, nx=30, ny=30)
    controller = PID(1, 1, 1, setpoint=500)

    t = 0
    for i in range(100):
        t += dt

        T = model.step_time()
        T_mean = np.mean(T.vector().get_local())
        q_dot = controller(T_mean)

        for component in fa.get_component_set('controllable_q_dot'):
            component.set_volumetric_power_density(q_dot)

    p = plot(T, vmin=490, vmax=510)
    plt.colorbar(p, format='%.1f K')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Fuel Assembly Temperature at t={:.2f}s'.format(t))
    plt.savefig('heat_exchanger_PID_steady_state.png')
