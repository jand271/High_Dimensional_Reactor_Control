class Material(object):
    def __init__(self,
                 thermal_conductivity=None,
                 density=None,
                 specific_heat_capacity=None):
        """ Check Constructor Inputs """
        assert thermal_conductivity is None or isinstance(thermal_conductivity, float)
        assert density is None or isinstance(density, float)
        assert specific_heat_capacity is None or isinstance(specific_heat_capacity, float)

        self._thermal_conductivity = thermal_conductivity
        self._density = density
        self._specific_heat_capacity = specific_heat_capacity

        """ Getters """

    def get_thermal_conductivity(self):
        return self._thermal_conductivity

    def get_density(self):
        return self._density

    def get_specific_heat_capacity(self):
        return self._specific_heat_capacity


class UO2(Material):
    def __init__(self):
        super().__init__(thermal_conductivity=8000., density=1.0897e4, specific_heat_capacity=281.45)
        """
        Material Sources:
        https://info.ornl.gov/sites/publications/Files/Pub57523.pdf
        @ T = 500 K
            rho = 10.97 [g/cm3] -> 10970 [kg/m3]
            k = 1.0897e4 8000 [W/m/K]
            cp = 75 [J/mol/K] / 0.27003 [kg/mol] ->  281.45 [J/kg/K]
        """


class HighBoronSteel(Material):
    def __init__(self):
        super().__init__(thermal_conductivity=1., density=1., specific_heat_capacity=1.)
