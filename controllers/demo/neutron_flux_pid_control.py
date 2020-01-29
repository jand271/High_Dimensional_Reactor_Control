import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
from fenics import *
from assembly_construction.premade_fuel_assemblies import ComponentAssemblyC
from modeling.neutron_diffusion_fem_model import NuclearReactorNeutronicsFEMModel

if __name__ == "__main__":

    fa = ComponentAssemblyC()

    dt = 1
    model = NuclearReactorNeutronicsFEMModel(fa, dt, nx=20, ny=20)
    controller = PID(10, 0.1, 0, setpoint=4e12)

    t = 0
    for i in range(100):
        t += dt

        PHI = model.step_time()
        PHI_mean = np.mean(PHI.vector().get_local())
        s = controller(PHI_mean)

        for component in fa.get_component_set('control_rods'):
            component.set_volumetric_neutron_source(s)

    p = plot(PHI, vmin=4.6e12, vmax=9e12)
    plt.colorbar(p, format='%.1e n/m^2/s')
    for component in fa.get_component_set('control_rods'):
        plt.annotate('{0:.0e}'.format(component.get_volumetric_neutron_source()), component.get_position())
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Fuel Assembly Neutron Flux at t={:.2f}s'.format(t))
    plt.savefig('neutron_flux_PID_steady_state.png')
