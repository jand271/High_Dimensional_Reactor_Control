import matplotlib.pyplot as plt
from assembly_construction.component_assembly import ComponentAssembly
from assembly_construction.component import UnshapedComponent
from assembly_construction.material import Material
from modeling.temperature_fem_model import UniformPowerAndHeatSinkTemperatureFEMModel
from fenics import *

if __name__ == "__main__":

    fa = ComponentAssembly(
        default_component=UnshapedComponent(
            material=Material(thermal_conductivity=1.0, specific_heat_capacity=1.0, density=1.0)
        )
    )

    q = UniformPowerAndHeatSinkTemperatureFEMModel(fa, 0.05, nx=10, ny=10)

    q.set_uniform_power_density(12)
    q.set_uniform_boundary_power_flux(-6)

    for i in range(20):
        p = plot(q.step_time())
        plt.colorbar(p, format="%.1f K")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Fuel Assembly Temperature")
        plt.pause(0.5)
        plt.clf()

        if i == 10:
            q.set_uniform_power_density(-12)
            q.set_uniform_boundary_power_flux(6)
