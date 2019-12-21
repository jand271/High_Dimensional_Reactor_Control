import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC
from fem_model.fem_model import FEMModel
from fenics import *


class HeatEquationModel(FEMModel, ABC):
    class K(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent thermal conductivity"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(tuple(vertex)).get_material().get_thermal_conductivity()

    class Rho(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent density"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(tuple(vertex)).get_material().get_density()

    class Cp(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent specific heat capacity"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(
                tuple(vertex)).get_material().get_specific_heat_capacity()

    class q_dot(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent volumetric power density"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(tuple(vertex)).get_volumetric_power_density()


class UniformPowerAndHeatSinkTemperatureFEMModel(HeatEquationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ Fenics Problem Formulation """
        self._V = FunctionSpace(self._mesh, 'P', 1)

        self._Tn1 = TrialFunction(self._V)  # Temperature at T(n+1)
        self._v = TestFunction(self._V)  # weighting function

        # Functional Assumptions of model
        self._uniform_power_density = Constant(0)
        self._uniform_boundary_power_flux = Constant(0)

        self._T0 = Constant(500)

        self._Tn = project(self._T0, self._V)  # Temperature at T(n)

        # positionally dependent coefficients
        self._k = self.K(self._component_hash_map)
        self._rho = self.Rho(self._component_hash_map)
        self._cp = self.Cp(self._component_hash_map)

        self._a, self._L = None, None

        self.setup_problem()

    def setup_problem(self):
        F = self._rho * self._cp / self._dt * (self._Tn1 - self._Tn) * self._v * dx \
            + self._k * dot(grad(self._Tn1), grad(self._v)) * dx \
            - self._uniform_power_density * self._v * dx \
            - self._uniform_boundary_power_flux * self._v * ds

        self._a, self._L = lhs(F), rhs(F)

    def step_time(self):
        Tn1 = Function(self._V)
        solve(self._a == self._L, Tn1)
        self._Tn.assign(Tn1)
        return Tn1

    def set_uniform_power_density(self, power_density):
        self._uniform_power_density = Constant(power_density)
        self.setup_problem()

    def set_uniform_boundary_power_flux(self, power_flux):
        self._uniform_boundary_power_flux = Constant(power_flux)
        self.setup_problem()


class HeatExchangerFEMModel(HeatEquationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ Fenics Problem Formulation """
        self._V = FunctionSpace(self._mesh, 'P', 1)

        self._Tn1 = TrialFunction(self._V)  # Temperature at T(n+1)
        self._v = TestFunction(self._V)  # weighting function

        self._T0 = Constant(500)

        self._Tn = project(self._T0, self._V)  # Temperature at T(n)

        # positionally dependent coefficients and variables
        self._k = self.K(self._component_hash_map)
        self._rho = self.Rho(self._component_hash_map)
        self._cp = self.Cp(self._component_hash_map)
        self._q_dot = self.q_dot(self._component_hash_map)

        self._a, self._L = None, None

        self.setup_problem()

    def setup_problem(self):
        F = self._rho * self._cp / self._dt * (self._Tn1 - self._Tn) * self._v * dx \
            + self._k * dot(grad(self._Tn1), grad(self._v)) * dx \
            - self._q_dot * self._v * dx

        self._a, self._L = lhs(F), rhs(F)

    def step_time(self):
        Tn1 = Function(self._V)
        solve(self._a == self._L, Tn1)
        self._Tn.assign(Tn1)
        return Tn1

    def state_transition_model(self):
        M = assemble(
            - self._rho * self._cp / self._dt * self._Tn1 * self._v * dx
            - self._k * dot(grad(self._Tn1), grad(self._v)) * dx
        ).array()
        M_inverse = np.linalg.inv(M)

        K = assemble(
            - self._rho * self._cp / self._dt * self._Tn1 * self._v * dx
        ).array()

        A = np.dot(M_inverse, K)

        v = assemble(- self._v * dx).get_local()

        q_dot_controllable_set = self._fuel_assembly.get_component_set('controllable_q_dot')

        B = np.zeros((A.shape[0], len(q_dot_controllable_set)))
        for component_index, component in zip(range(len(q_dot_controllable_set)), q_dot_controllable_set):
            vs = self.get_vertices_of_component(component)
            B[vs, component_index] = 1
        B = np.dot(M_inverse, np.multiply(v[:, np.newaxis], B))

        q_dot_uncontrollable_set = self._fuel_assembly.get_component_set('set_q_dot')

        f = np.zeros((A.shape[0],))
        for component_index, component in zip(range(len(q_dot_controllable_set)), q_dot_uncontrollable_set):
            vs = self.get_vertices_of_component(component)
            f[vs] = component.get_volumetric_power_density()

        f = np.dot(M_inverse, np.multiply(v, f))

        return A, B, f
