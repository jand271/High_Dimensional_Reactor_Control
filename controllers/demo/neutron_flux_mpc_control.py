import numpy as np
import matplotlib.pyplot as plt
from assembly_construction.premade_fuel_assemblies import ComponentAssemblyC
from modeling.neutron_diffusion_fem_model import NuclearReactorNeutronicsFEMModel
from controllers.neutron_flux_mpc_controller import NeutronFluxMPCController
from fenics import *

if __name__ == "__main__":

    fa = ComponentAssemblyC()

    dt = 1
    model = NuclearReactorNeutronicsFEMModel(fa, dt, nx=10, ny=10)
    controller = NeutronFluxMPCController(model, 1e12, max_removal_neutron_source=1e14, R=1e-12 * np.eye(17))

    ss = controller.update_then_calculate_optimal_actuation(
        model._PHI0.values() * np.ones((model.get_number_of_vertices()))
    )
    for s, component in zip(ss, fa.get_component_set("control_rods")):
        component.set_volumetric_neutron_source(np.minimum(s, 0))

    t = 0
    for i in range(20):
        t += dt

        PHI = model.step_time()
        ss = controller.update_then_calculate_optimal_actuation(PHI.vector().get_local())

        for s, component in zip(ss, fa.get_component_set("control_rods")):
            component.set_volumetric_neutron_source(np.minimum(s, 0))

    p = plot(PHI, vmin=1e12, vmax=9e12)
    plt.colorbar(p, format="%.1e n/m^2/s")
    for component in fa.get_component_set("control_rods"):
        plt.annotate("{0:.0e}".format(component.get_volumetric_neutron_source()), component.get_position())
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Fuel Assembly Neutron Flux at t={:.2f}s".format(t))
    plt.savefig("neutron_flux_MPC_steady_state.png")
