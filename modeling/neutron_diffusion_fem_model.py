import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC
from modeling.fem_model import FEMModel
from fenics import *


class NeutronDiffusionEquationModel(FEMModel, ABC):
    class D(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent diffusion length"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(tuple(vertex)).get_material().get_diffusion_length()

    class Sigma_a(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent absorption macroscopic cross section"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(
                tuple(vertex)).get_material().get_absorption_macroscopic_cross_section()

    class Sigma_f(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent fission macroscopic cross section"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(
                tuple(vertex)).get_material().get_fission_macroscopic_cross_section()

    class S(FEMModel.Coefficient):
        """Fenics implementation for positionally dependent volumetric power density"""

        def eval(self, value, vertex):
            value[0] = self._component_hash_map.find_component(tuple(vertex)).get_volumetric_neutron_source()


class NuclearReactorNeutronicsFEMModel(NeutronDiffusionEquationModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """ Fenics Problem Formulation """
        self._V = FunctionSpace(self._mesh, 'P', 1)

        self._PHIn1 = TrialFunction(self._V)  # Neutron Flux at Phi(n+1)
        self._v = TestFunction(self._V)  # weighting function

        # typical neutron flux of reactor vessel per wikipedia
        # <https://en.wikipedia.org/wiki/Neutron_flux>
        self._PHI0 = Constant(6.5e19 / 3.154e7)
        # <https://en.wikipedia.org/wiki/Neutron_temperature>
        # self._v_n = 2.19e3  # neutron speed [m/s]
        self._v_n = 1.
        # <https://en.wikipedia.org/wiki/Nuclear_fission>
        self._nu = 2.5  # average neutrons produced per fission

        self._PHIn = project(self._PHI0, self._V)  # Neutron Flux at T(n)

        # positionally dependent coefficients and variables
        self._D = self.D(self._component_hash_map)
        self._Sigma_a = self.Sigma_a(self._component_hash_map)
        self._Sigma_f = self.Sigma_f(self._component_hash_map)
        self._S = self.S(self._component_hash_map)

        self._a, self._L = None, None

        self.setup_problem()

    def setup_problem(self):
        F = - 1 / self._v_n / self._dt * self._PHIn1 * self._v * dx \
            - self._D * dot(grad(self._PHIn1), grad(self._v)) * dx \
            + 1 / self._v_n / self._dt * self._PHIn * self._v * dx \
            + self._S * self._v * dx \
            + (self._nu * self._Sigma_f - self._Sigma_a) * self._PHIn1 * self._v * dx

        self._a, self._L = lhs(F), rhs(F)

    def step_time(self):
        PHIn1 = Function(self._V)
        solve(self._a == self._L, PHIn1)
        self._PHIn.assign(PHIn1)
        return PHIn1

    def state_transition_model(self):
        M = assemble(
            1 / self._v_n / self._dt * self._PHIn1 * self._v * dx \
            + self._D * dot(grad(self._PHIn1), grad(self._v)) * dx \
            + (self._nu * self._Sigma_f - self._Sigma_a) * self._PHIn1 * self._v * dx
        ).array()
        M_inverse = np.linalg.inv(M)

        K = assemble(
            1 / self._v_n / self._dt * self._PHIn1 * self._v * dx
        ).array()

        A = np.dot(M_inverse, K)

        v = assemble(self._v * dx).get_local()

        control_rod_set = self._fuel_assembly.get_component_set('control_rods')

        B = np.zeros((A.shape[0], len(control_rod_set)))
        for component_index, component in zip(range(len(control_rod_set)), control_rod_set):
            vs = self.get_vertices_of_component(component)
            B[vs, component_index] = 1
        B = np.dot(M_inverse, np.multiply(v[:, np.newaxis], B))

        f = np.zeros((A.shape[0],))

        return A, B, f
