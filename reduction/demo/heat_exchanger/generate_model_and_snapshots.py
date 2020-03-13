import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from fenics import *
from assembly_construction.component_assembly import ComponentAssembly
from assembly_construction.component import UnshapedComponent
from assembly_construction.rod import Rod
from assembly_construction.material import Material
from modeling.temperature_fem_model import HeatExchangerFEMModel

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
    model = HeatExchangerFEMModel(fa, dt, nx=15, ny=15)

    fa.plot()
    plt.savefig('Fuel_Assembly.png')

    A, B, f = model.state_transition_model()
    important_vertices = model.get_vertices_of_component(fa.get_component_set('set_q_dot')[0])

    nx = A.shape[0]
    ns = 25

    from controllers.dlqr import AffineDLQR

    controller = AffineDLQR(A, B, f, 10)
    C = controller.F[:, :-1]

    # C = np.zeros((len(important_vertices), nx))
    # for i in range(len(important_vertices)):
    #     C[i, important_vertices[i]] = 1

    X = np.zeros((nx, ns))
    X[:, 0] = 500 * np.ones((nx,))
    u = -300 * np.ones((B.shape[1],))

    for i in range(1, ns):
        X[:, i] = A @ X[:, i - 1] + B @ u

    savemat('heat_exchanger_model.mat', {'A': A, 'B': B, 'f': f, 'C': C})
    savemat('snapshots_heat_exchanger_model.mat', {'X': X})
