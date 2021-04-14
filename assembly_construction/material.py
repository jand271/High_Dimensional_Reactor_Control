class Material(object):
    def __init__(
        self,
        thermal_conductivity=None,
        density=None,
        specific_heat_capacity=None,
        diffusion_length=None,
        absorption_macroscopic_cross_section=None,
        fission_macroscopic_cross_section=None,
    ):
        """ Check Constructor Inputs """

        for material_property in [
            thermal_conductivity,
            density,
            specific_heat_capacity,
            diffusion_length,
            absorption_macroscopic_cross_section,
        ]:
            assert material_property is None or isinstance(
                material_property, float
            ), "Material property either not None or not a float"

        assert (
            fission_macroscopic_cross_section is None
            or isinstance(fission_macroscopic_cross_section, float)
            or fission_macroscopic_cross_section == 0
        ), "Material property either not None or not a float"

        self._thermal_conductivity = thermal_conductivity
        self._density = density
        self._specific_heat_capacity = specific_heat_capacity
        self._diffusion_length = diffusion_length
        self._absorption_macroscopic_cross_section = absorption_macroscopic_cross_section
        self._fission_macroscopic_cross_section = fission_macroscopic_cross_section

        """ Getters """

    def get_thermal_conductivity(self):
        return self._thermal_conductivity

    def get_density(self):
        return self._density

    def get_specific_heat_capacity(self):
        return self._specific_heat_capacity

    def get_diffusion_length(self):
        return self._diffusion_length

    def get_absorption_macroscopic_cross_section(self):
        return self._absorption_macroscopic_cross_section

    def get_fission_macroscopic_cross_section(self):
        return self._fission_macroscopic_cross_section


class UO2(Material):
    def __init__(self):
        super().__init__(
            thermal_conductivity=8000.0,
            density=1.0897e4,
            specific_heat_capacity=281.45,
            diffusion_length=1 / 3 / (1.51e1 * 1e-28),
            absorption_macroscopic_cross_section=2.34e22 * 1e6 * 5.94e2 * 1e-28,
            fission_macroscopic_cross_section=2.34e22 * 1e6 * 5.94e2 * 1e-28,
        )
        """
        Material Sources:
        <https://info.ornl.gov/sites/publications/Files/Pub57523.pdf>
        @ T = 500 K
            rho = 10.97 [g/cm3] -> 10970 [kg/m3]
            k = 1.0897e4 8000 [W/m/K]
            cp = 75 [J/mol/K] / 0.27003 [kg/mol] ->  281.45 [J/kg/K]
        <http://atom.kaeri.re.kr>
        @ E = 0.025 eV - Thermal Neutron Temperature
            fission microscopic cross section U235 5.94e2 * 1e-28 [m^2]
            absorption macroscopic cross section   7.23e2 * 1e-28 [m^2]
            scattering microscopic cross section   1.51e1 * 1e-28 [m^2]
        <https://www.nuclear-power.net/nuclear-power/reactor-physics/nuclear-engineering-fundamentals/neutron-nuclear-reactions/atomic-number-density/>
            N_U = 2.34e22 * 1e6 atoms of U per m^3
            Assuming only U235
            fission macroscopic cross section 2.34e22 * 1e6 * 5.94e2 * 1e-28
            absorption macroscopic cross section 2.34e22 * 1e6 * 7.23e2 * 1e-28
        <https://www.nuclear-power.net/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-coefficient/>
            diffusion length 1 / 3 / (1.51e1 * 1e-28) 
        """


class H20_500K(Material):
    def __init__(self):
        super().__init__(
            thermal_conductivity=0.74,
            density=831.5,
            specific_heat_capacity=4.67e3,
            diffusion_length=0.142 * 1e-2,
            absorption_macroscopic_cross_section=0.022 * 1e2,
            fission_macroscopic_cross_section=0.0,
        )
        """
        <https://www.engineeringtoolbox.com/>
        <https://www.nuclear-power.net/nuclear-power/reactor-physics/neutron-diffusion-theory/diffusion-length/>
        """


class HighBoronSteel(Material):
    def __init__(self):
        super().__init__(thermal_conductivity=1.0, density=1.0, specific_heat_capacity=1.0)
