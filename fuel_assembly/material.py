class Material(object):
    def __init__(self,
                 thermal_conductivity=None,
                 density=None,
                 specific_heat_capacity=None):
        """ Check Constructor Inputs """
        assert thermal_conductivity is None or isinstance(thermal_conductivity, float)
        assert density is None or isinstance(density, float)
        assert specific_heat_capacity is None or isinstance(specific_heat_capacity, float)

        self.thermal_conductivity = thermal_conductivity
        self.density = density
        self.specific_heat_capacity = specific_heat_capacity


class UO2(Material):
    def __init__(self):
        super().__init__(thermal_conductivity=1., density=1., specific_heat_capacity=1.)


class HighBoronSteel(Material):
    def __init__(self):
        super().__init__(thermal_conductivity=1., density=1., specific_heat_capacity=1.)
