import matplotlib.pyplot as plt
from assembly_construction.component import Component


class Bar(Component):
    def __init__(self, x_center, y_center, x_length, y_length, plot_color="k", material=None):
        """
        Bar Constructor
        :param x_center: x position of Component
        :param y_center: y position of Component
        :param x_length: length of bar in x direction
        :param y_length: length of bar in y direction
        :param plot_color: applicable plotting color
        :param material: material of component
        """
        super().__init__(x_center, y_center, plot_color, material=material)

        """ Check Constructor Inputs """
        assert isinstance(x_length, float) or isinstance(x_length, int)
        assert isinstance(y_length, float) or isinstance(y_length, int)

        self._x_length = x_length
        self._y_length = y_length

    def is_point_within(self, x, y):
        """ Checks if input position (x,y) is within the rod """
        return abs(x - self._x_position) <= self._x_length / 2 and abs(y - self._y_position) <= self._y_length / 2

    def plot(self):
        plt.gca().add_patch(
            plt.Rectangle(
                (self._x_position - self._x_length / 2, self._y_position - self._y_length / 2),
                self._x_length,
                self._y_length,
                color=self._plot_color,
                alpha=0.5,
            )
        )


class SquareBar(Bar):
    def __init__(self, x_center, y_center, side_length, plot_color="k", material=None):
        """
        Square Bar Constructor
        :param x_center: x position of Component
        :param y_center: y position of Component
        :param side_length: side length of bar
        :param plot_color: applicable plotting color
        :param material: material of component
        """
        super().__init__(x_center, y_center, side_length, side_length, plot_color=plot_color, material=material)
