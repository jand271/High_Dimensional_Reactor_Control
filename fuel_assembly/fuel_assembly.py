import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fuel_assembly.component import Component


class FuelAssembly(object):
    def __init__(self, xlim=(-1, 1), ylim=(-1, 1), default_component=None):
        """
        FuelAssembly Constructor
        :param xlim: x limits for pyplot
        :param ylim: y limits for pyplot
        """
        self._xlim = xlim
        self._ylim = ylim
        self._default_component = default_component

        self._rod_points = []  # list of all rod center points
        self._point_to_rod_hash_map = {}  # hash map mapping rod center point to rod object
        self._point_kd_tree = None  # KDTree for efficient find 2D nearest component 
        self._kd_tree_updated = False  # flag the keeps track of when the KDTree must be updated

        # hash map maps control variable to applicable list of components sets
        self._component_sets = {}

    def get_domain_limits(self):
        """ get domain limits """
        return self._xlim[0], self._xlim[1], self._ylim[0], self._ylim[1]

    def add_component(self, component, component_set=None):
        """ Adds input component to FuelAssembly"""
        assert isinstance(component, Component)  # checks input

        point = component.get_position()
        self._rod_points.append(point)
        self._point_to_rod_hash_map[point] = component
        self._kd_tree_updated = False  # KDTree must be updated upon next nearest component query

        # add component to set if applicable
        if component_set is not None:
            if component_set not in self._component_sets:
                self._component_sets[component_set] = []
            self._component_sets[component_set].append(component)

    def find_component(self, x, y):
        """
        Finds and returns component in fuel assembly corresponding to input (x,y) or returns None if no component
        corresponds to input (x,y)
        """

        """ update kd tree if needed """
        if not self._kd_tree_updated:
            if len(self._rod_points) == 0:
                return self._default_component
            self._point_kd_tree = KDTree(self._rod_points)
            self._kd_tree_updated = True  # set flag to denote that tree need not be updated

        distance, nearest_point_index = self._point_kd_tree.query((x, y))
        nearest_point = self._rod_points[nearest_point_index]
        nearest_component = self._point_to_rod_hash_map[nearest_point]

        if nearest_component.is_point_within(x, y):
            return nearest_component
        else:
            return self._default_component

    def get_component_set(self, component_set):
        """ return set of components of component_set """
        return self._component_sets[component_set]

    def plot(self):
        """
        plot fuel assembly with pyplot
        """
        for rod in self._point_to_rod_hash_map.values():
            rod.plot()

        plt.xlim(self._xlim)
        plt.ylim(self._ylim)
        plt.gca().set_aspect('equal', adjustable='box')
