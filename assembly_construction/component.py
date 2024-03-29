from abc import ABC, abstractmethod
from assembly_construction.material import Material


class Component(ABC):
    def __init__(self, x_position, y_position, plot_color, material=None):
        """
        Component Constructor
        :param x_position: x position of Component
        :param y_position: y position of Component
        :param plot_color: applicable plotting color
        """

        """ Check Constructor Inputs """
        assert x_position is None or isinstance(x_position, float) or isinstance(x_position, int)
        assert y_position is None or isinstance(y_position, float) or isinstance(y_position, int)
        assert material is None or isinstance(material, Material)

        self._x_position = x_position
        self._y_position = y_position
        self._plot_color = plot_color
        self._material = material

        self._volumetric_power_density = 0
        self._volumetric_neutron_source = 0

    """ Getters """

    def get_position(self):
        return self._x_position, self._y_position

    def get_plot_color(self):
        return self._plot_color

    def get_material(self):
        return self._material

    def get_volumetric_power_density(self):
        return self._volumetric_power_density

    def get_volumetric_neutron_source(self):
        return self._volumetric_neutron_source

    """ Setters """

    def set_volumetric_power_density(self, power_density):
        self._volumetric_power_density = power_density

    def set_volumetric_neutron_source(self, neutron_source):
        self._volumetric_neutron_source = neutron_source

    """ Abstract Methods """

    @abstractmethod
    def is_point_within(self, x, y):
        pass

    @abstractmethod
    def plot(self):
        pass


class UnshapedComponent(Component):
    def __init__(self, plot_color=None, material=None):
        super().__init__(None, None, plot_color, material=material)

    def is_point_within(self, x, y):
        return True

    def plot(self):
        pass
