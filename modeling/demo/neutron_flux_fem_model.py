import numpy as np
import matplotlib.pyplot as plt
from assembly_construction.premade_fuel_assemblies import ComponentAssemblyC
from modeling.neutron_diffusion_fem_model import NuclearReactorNeutronicsFEMModel
from fenics import *

if __name__ == "__main__":

    fa = ComponentAssemblyC()

    dt = 1
    q = NuclearReactorNeutronicsFEMModel(fa, dt, nx=20, ny=20)

    for rod in fa.get_component_set("control_rods"):
        rod.set_volumetric_neutron_source(-9.6075439454e12)  # slightly super or subcritical depending on mesh

    fa.plot()
    q.step_time()
    q._component_hash_map.plot()
    plt.title("Fuel Assembly and Mesh")
    plt.show()
    plt.clf()

    t = 0

    for i in range(5):
        t += dt
        p = plot(q.step_time())
        plt.colorbar(p, format="%.1e n/cm^2/s")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Fuel Assembly Neutron Flux at t={:.2f}s".format(t))
        plt.pause(0.5)
        plt.clf()
