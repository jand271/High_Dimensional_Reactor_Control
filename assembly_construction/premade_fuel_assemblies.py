import numpy as np
import matplotlib.pyplot as plt
from assembly_construction.component_assembly import ComponentAssembly
from assembly_construction.component import UnshapedComponent
from assembly_construction.rod import Rod
from assembly_construction.bar import SquareBar
from assembly_construction.material import Material


class ComponentAssemblyA(ComponentAssembly):

    def __init__(self):
        """ creates heat exchanger with 17 cylinder rods """
        material = Material(thermal_conductivity=1., specific_heat_capacity=1., density=1.)
        super().__init__(default_component=UnshapedComponent(material=material, plot_color='k'))

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
            self.add_component(rod, component_set='set_q_dot')

        for rod in cooling_rods:
            rod.set_volumetric_power_density(cool_power_density)
            self.add_component(rod, component_set='controllable_q_dot')


class ComponentAssemblyB(ComponentAssembly):

    def __init__(self):
        """ creates heat exchanger with 17 cylinder bars """
        material = Material(thermal_conductivity=1., specific_heat_capacity=1., density=1.)
        super().__init__(default_component=UnshapedComponent(material=material, plot_color='k'))

        nh = 8
        nc = 9
        r = 0.15
        heat_power_density = 1000

        # cooling bars
        cooling_bars = [SquareBar(0, 0, r, plot_color='b', material=material)]
        R = 0.6
        for t in np.linspace(0, 2 * np.pi, nc):
            if t == 0:
                continue
            cooling_bars.append(
                SquareBar(np.round(R * np.cos(t), 1), np.round(R * np.sin(t), 1), r, plot_color='b', material=material))

        # heating bars
        heating_bars = []
        R = 0.3
        i = 0
        for t in np.linspace(0, 2 * np.pi, nh + 1):
            i += 1
            if t == 0:
                continue
            if i % 2 == 0:
                heating_bars.append(SquareBar(np.round(R * np.cos(t), 1), np.round(R * np.sin(t), 1), r, plot_color='r',
                                              material=material))
            else:
                cooling_bars.append(SquareBar(np.round(R * np.cos(t), 1), np.round(R * np.sin(t), 1), r, plot_color='b',
                                              material=material))
                nh += -1
                nc += 1

        cool_power_density = -heat_power_density * nh / nc

        for bar in heating_bars:
            bar.set_volumetric_power_density(heat_power_density)
            self.add_component(bar, component_set='set_q_dot')

        for bar in cooling_bars:
            bar.set_volumetric_power_density(cool_power_density)
            self.add_component(bar, component_set='controllable_q_dot')


class ComponentAssemblyC(ComponentAssembly):

    def __init__(self):
        """ creates neuclear fuel assembly with 17 cylinder control rods """

        fissile_dummy_material = Material(
            thermal_conductivity=1.,
            density=1.,
            specific_heat_capacity=1.,
            diffusion_length=1.,
            absorption_macroscopic_cross_section=1.,
            fission_macroscopic_cross_section=1.)

        super().__init__(default_component=UnshapedComponent(material=fissile_dummy_material, plot_color='k'))

        dummy_material = Material(
            thermal_conductivity=1.,
            density=1.,
            specific_heat_capacity=1.,
            diffusion_length=1.,
            absorption_macroscopic_cross_section=1.,
            fission_macroscopic_cross_section=0.)

        control_rod_material = dummy_material

        nh = 8
        nc = 9
        r = 0.1

        # cooling rods
        control_rods = [Rod(0, 0, r, plot_color='b', material=control_rod_material)]
        R = 0.6
        for t in np.linspace(0, 2 * np.pi, nc):
            if t == 0:
                continue
            control_rods.append(Rod(R * np.cos(t), R * np.sin(t), r, plot_color='b', material=control_rod_material))

        R = 0.33
        i = 0
        for t in np.linspace(0, 2 * np.pi, nh + 1):
            if t == 0:
                continue
            control_rods.append(Rod(R * np.cos(t), R * np.sin(t), r, plot_color='b', material=control_rod_material))

        starting_neutron_poison_density = -1e5

        for rod in control_rods:
            rod.set_volumetric_neutron_source(starting_neutron_poison_density)
            self.add_component(rod, component_set='control_rods')

if __name__ == '__main__':
    fa_A = ComponentAssemblyA()
    fa_A.plot()
    plt.savefig('Fuel_Assembly_A.png')

    plt.clf()

    fa_B = ComponentAssemblyB()
    fa_B.plot()
    plt.savefig('Fuel_Assembly_B.png')

    plt.clf()

    fa_C = ComponentAssemblyC()
    fa_C.plot()
    plt.savefig('Fuel_Assembly_C.png')

    plt.clf()
