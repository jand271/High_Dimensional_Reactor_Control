import matplotlib.pyplot as plt


class Rod(object):
    def __init__(self, x_position, y_position, radius, color='k'):
        """
        Rod Constructor
        :param x_position: center x position of Rod
        :param y_position: center y position of Rod
        :param radius: radius of Rod
        :param color: applicable plotting color
        """

        """ Check Constructor Inputs """
        assert isinstance(x_position, float) or isinstance(x_position, int)
        assert isinstance(y_position, float) or isinstance(y_position, int)
        assert isinstance(radius, float) or isinstance(radius, int)

        self._x_position = x_position
        self._y_position = y_position
        self._radius = radius
        self._plot_color = color

    def get_position(self):
        """
        returns a tuple of the rod (x,y) center position
        :return: (x,y)
        """
        return self._x_position, self._y_position

    def is_point_within(self, x, y):
        """
        Checks if input position (x,y) is within the rod
        :return: boolean
        """
        if (x - self._x_position) ** 2 + (y - self._y_position) ** 2 < self._radius ** 2:
            return True
        else:
            return False

    def plot(self):
        """
        plots a circle patch of the rod on current axis
        """
        plt.gca().add_patch(plt.Circle((self._x_position, self._y_position), self._radius, color=self._plot_color))


class ControlRod(Rod):
    def __init__(self, *args):
        super().__init__(*args, color='g')


class FuelRod(Rod):
    def __init__(self, *args):
        super().__init__(*args, color='r')
