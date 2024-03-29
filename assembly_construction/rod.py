import matplotlib.pyplot as plt
from assembly_construction.component import Component
from assembly_construction.material import UO2, HighBoronSteel, H20_500K


class Rod(Component):

    def __init__(self, x_center, y_center, radius, plot_color='k', material=None):
        """
        Rod Constructor
        :param x_center: x position of Component
        :param y_center: y position of Component
        :param radius: radius of rod
        :param plot_color: applicable plotting color
        :param material: material of component
        """
        super().__init__(x_center, y_center, plot_color, material=material)

        """ Check Constructor Inputs """
        assert isinstance(radius, float) or isinstance(radius, int)

        self._radius = radius

    def is_point_within(self, x, y):
        """ Checks if input position (x,y) is within the rod """
        return (x - self._x_position) ** 2 + (y - self._y_position) ** 2 <= self._radius ** 2

    def plot(self):
        """ plots a circle patch of the rod on current axis """
        plt.gca().add_patch(
            plt.Circle((self._x_position, self._y_position), self._radius, color=self._plot_color, alpha=0.5))


class ControlRod(Rod):
    def __init__(self, x_center, y_center, radius):
        super().__init__(x_center, y_center, radius, plot_color='g', material=HighBoronSteel())


class FuelRod(Rod):
    def __init__(self, x_center, y_center, radius):
        super().__init__(x_center, y_center, radius, plot_color='r', material=UO2())


class CoolingPipe(Rod):
    def __init__(self, x_center, y_center, radius):
        super().__init__(x_center, y_center, radius, plot_color='b', material=H20_500K)
