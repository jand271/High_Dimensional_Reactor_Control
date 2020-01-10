import numpy as np
import matplotlib.pyplot as plt
from assembly_construction.component_assembly import ComponentAssembly
from assembly_construction.component import UnshapedComponent
from assembly_construction.rod import Rod
from assembly_construction.material import Material
from modeling.temperature_fem_model import HeatExchangerFEMModel
from controllers.heat_exchanger_mpc_controller import HeatExchangerMPCController
from fenics import *

if __name__ == "__main__":

    material = Material(thermal_conductivity=1., specific_heat_capacity=1., density=1.)

    fa = ComponentAssembly(default_component=UnshapedComponent(material=material, plot_color='k'))

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

    dt = 1  # works with dt = 1!!!!!
    model = HeatExchangerFEMModel(fa, dt, nx=20, ny=20)
    controller = HeatExchangerMPCController(model, 500, max_removal_power_density=1000)

    fa.plot()
    model.step_time()
    model._component_hash_map.plot()
    plt.title('Fuel Assembly and Mesh')
    plt.savefig('Fuel_Assembly_and_Mesh.png')
    plt.clf()

    t = 0
    for i in range(10):
        t += dt

        T = model.step_time()
        q_dots = controller.update_then_calculate_optimal_actuation(T.vector().get_local())

        for q_dot, component in zip(q_dots, fa.get_component_set('controllable_q_dot')):
            component.set_volumetric_power_density(q_dot)

    p = plot(T, vmin=490, vmax=510)
    plt.colorbar(p, format='%.1f K')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Fuel Assembly Steady State Temperature at t={:.2f}s'.format(t))
    plt.savefig('heat_exchanger_MPC_steady_state.png')
