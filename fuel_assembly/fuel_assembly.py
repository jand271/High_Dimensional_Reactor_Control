import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fuel_assembly.rod import Rod


class FuelAssembly(object):
    def __init__(self, xlim=(-1, 1), ylim=(-1, 1)):
        """
        FuelAssembly Constructor
        :param xlim: x limits for pyplot
        :param ylim: y limits for pyplot
        """
        self._xlim = xlim
        self._ylim = ylim

        self._rod_points = []  # list of all rod center points
        self._point_to_rod_hash_map = {}  # hash map mapping rod center point to rod object
        self._point_kd_tree = None  # KDTree for efficient find 2D nearest component 
        self._kd_tree_updated = False  # flag the keeps track of when the KDTree must be updated

    def add_rod(self, rod):
        """
        Adds input rod to FuelAssembly
        """
        assert isinstance(rod, Rod)  # checks input

        point = rod.get_position()
        self._rod_points.append(point)
        self._point_to_rod_hash_map[point] = rod
        self._kd_tree_updated = False  # KDTree must be updated upon next nearest component query

    def find_component(self, x, y):

        """
        finds and returns component in fuel assembly corresponding to input (x,y) or returns None if no component
        corresponds to input (x,y)
        """

        """ update kd tree if needed """
        if not self._kd_tree_updated:
            if len(self._rod_points) == 0:
                return None
            self._point_kd_tree = KDTree(self._rod_points)
            self._kd_tree_updated = True  # set flag to denote that tree need not be updated

        distance, nearest_point_index = self._point_kd_tree.query((x, y))
        nearest_point = self._rod_points[nearest_point_index]
        nearest_component = self._point_to_rod_hash_map[nearest_point]

        if nearest_component.is_point_within(x, y):
            return nearest_component
        else:
            return None

    def plot(self):
        """
        plot fuel assembly with pyplot
        """
        for rod in self._point_to_rod_hash_map.values():
            rod.plot()

        plt.xlim(self._xlim)
        plt.ylim(self._ylim)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.title("Fuel Assembly")

        plt.show()
