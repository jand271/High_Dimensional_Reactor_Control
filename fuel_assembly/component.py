from abc import ABC, abstractmethod
from fuel_assembly.material import Material


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

    """ Getters """

    def get_position(self):
        return self._x_position, self._y_position

    def get_plot_color(self):
        return self._plot_color

    def get_material(self):
        return self._material

    """ Abstract Methods """

    @abstractmethod
    def is_point_within(self, x, y):
        pass

    @abstractmethod
    def plot(self):
        pass


class UnshapedComponent(Component):
    def __init__(self, material=None):
        super().__init__(None, None, None, material=material)

    def is_point_within(self, x, y):
        return True

    def plot(self):
        pass
