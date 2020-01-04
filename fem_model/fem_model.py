import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.component import Component
from fenics import *


class FEMModel(ABC):
    def __init__(self, fuel_assembly, dt, model_mesh=None, nx=10, ny=10):
        """
        Rod Constructor
        :param fuel_assembly: fuel assembly design
        :param dt: model step time
        :param model_mesh: input mesh
        :param nx, ny: mesh density parameters if default mesh is to be used
        """

        """ Check Constructor Inputs """
        assert isinstance(fuel_assembly, FuelAssembly)
        assert isinstance(dt, float) or isinstance(dt, int)
        assert model_mesh is None or isinstance(model_mesh, Mesh)
        assert isinstance(nx, int)
        assert isinstance(ny, int)

        # use default mesh if none given
        if model_mesh is None:
            x0, x1, y0, y1 = fuel_assembly.get_domain_limits()
            p0 = Point(x0, y0)
            p1 = Point(x1, y1)
            model_mesh = RectangleMesh(p0, p1, nx, ny, diagonal='crossed')

        self._fuel_assembly = fuel_assembly
        self._dt = dt
        self._mesh = model_mesh
        self._component_hash_map = self.ComponentHashMap(fuel_assembly)

        """ Vertex hash map maps each component to the enclosed vertices in the mesh """
        self._vertex_hash_map = {}
        self._number_of_vertices = self._mesh.coordinates().shape[0]
        for vertex_index, vertex in zip(range(self._number_of_vertices), self._mesh.coordinates()):
            component = self._component_hash_map.find_component(tuple(vertex))
            if component not in self._vertex_hash_map:
                self._vertex_hash_map[component] = []
            self._vertex_hash_map[component].append(vertex_index)

    def get_vertices_of_component(self, component):
        return self._vertex_hash_map[component]

    def get_number_of_vertices(self):
        return self._number_of_vertices

    @abstractmethod
    def setup_problem(self):
        pass

    @abstractmethod
    def step_time(self):
        pass

    @abstractmethod
    def state_transition_model(self):
        pass

    """ Nested Classes """

    class ComponentHashMap:
        """
        Cached hash map that maps position to component in a fuel assembly. This nested object prevents
        unnecessary searches for the component associated to a position.
        """

        def __init__(self, fuel_assembly):
            """
            ComponentHashMap Constructor
            :param fuel_assembly: fuel assembly of this hash map
            """
            self._fuel_assembly = fuel_assembly
            self._component_hash_map = {}  # initialize empty hash map

        def find_component(self, point):
            """
            Returns the associated component of input position point
            With each query, the query and result is cached into a hash map to prevent unnecessary calls to the fuel
            assembly search method
            """
            if point not in self._component_hash_map:  # check if point is not already cached in hash map
                # save result in hash map
                self._component_hash_map[point] = self._fuel_assembly.find_component(point[0], point[1])

            return self._component_hash_map[point]

        def plot(self):
            for (point, component) in self._component_hash_map.items():
                plt.scatter(point[0], point[1], color=component.get_plot_color(), s=1)

    class Coefficient(ABC, UserExpression):
        """Fenics implementation for positionally dependent coefficients"""

        def __init__(self, component_hash_map):
            super().__init__()
            self._component_hash_map = component_hash_map

        @abstractmethod
        def eval(self, value, vertex):
            pass
