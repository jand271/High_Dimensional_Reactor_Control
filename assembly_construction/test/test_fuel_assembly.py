import unittest
from assembly_construction.component_assembly import ComponentAssembly
from assembly_construction.rod import ControlRod, FuelRod
from assembly_construction.component import UnshapedComponent


class TestComponentAssembly(unittest.TestCase):
    def test_constructor(self):
        fa1 = ComponentAssembly()
        assert (-1, 1, -1, 1) == fa1.get_domain_limits()

        fa2 = ComponentAssembly(xlim=(-2, 3), ylim=(-5, 7))
        assert (-2, 3, -5, 7) == fa2.get_domain_limits()

        fa3 = ComponentAssembly(ylim=(-5, 7))
        assert (-1, 1, -5, 7) == fa3.get_domain_limits()

        fa4 = ComponentAssembly(xlim=(-2, 3))
        assert (-2, 3, -1, 1) == fa4.get_domain_limits()

    def test_kd_tree(self):
        fa = ComponentAssembly()

        assert fa.find_component(0, 0) is None

        cr0 = ControlRod(0, 0, 0.5)
        fr1 = FuelRod(1.0, 1.0, 0.5)
        fr2 = FuelRod(-1, 1, 0.5)
        fr3 = FuelRod(-1.0, -1.0, 0.5)
        fr4 = FuelRod(1, -1, 0.5)

        fa.add_component(cr0)

        assert fa.find_component(0, 0) is cr0

        fa.add_component(fr1)
        fa.add_component(fr2)
        fa.add_component(fr3)
        fa.add_component(fr4)

        assert fa.find_component(0, 0) is cr0

        assert fa.find_component(1, 1) is fr1
        assert fa.find_component(-1, 1) is fr2
        assert fa.find_component(-1, -1) is fr3
        assert fa.find_component(1, -1) is fr4

        assert fa.find_component(1.1, 1.1) is fr1
        assert fa.find_component(-1.1, 1.1) is fr2
        assert fa.find_component(-1.1, -1.1) is fr3
        assert fa.find_component(1.1, -1.1) is fr4

        assert fa.find_component(0.25, 0.25) is cr0
        assert fa.find_component(-0.25, 0.25) is cr0
        assert fa.find_component(-0.25, -0.25) is cr0
        assert fa.find_component(0.25, -0.25) is cr0

        assert fa.find_component(2, 2) is None
        assert fa.find_component(-2, 2) is None
        assert fa.find_component(-2, -2) is None
        assert fa.find_component(2, -2) is None

    def test_kd_tree_with_default_component(self):
        default_component = UnshapedComponent()
        fa = ComponentAssembly(default_component=default_component)

        assert fa.find_component(0, 0) is default_component

        cr0 = ControlRod(0, 0, 0.5)
        fr1 = FuelRod(1.0, 1.0, 0.5)
        fr2 = FuelRod(-1, 1, 0.5)
        fr3 = FuelRod(-1.0, -1.0, 0.5)
        fr4 = FuelRod(1, -1, 0.5)

        fa.add_component(cr0)

        assert fa.find_component(0, 0) is cr0

        fa.add_component(fr1)
        fa.add_component(fr2)
        fa.add_component(fr3)
        fa.add_component(fr4)

        assert fa.find_component(0, 0) is cr0

        assert fa.find_component(1, 1) is fr1
        assert fa.find_component(-1, 1) is fr2
        assert fa.find_component(-1, -1) is fr3
        assert fa.find_component(1, -1) is fr4

        assert fa.find_component(1.1, 1.1) is fr1
        assert fa.find_component(-1.1, 1.1) is fr2
        assert fa.find_component(-1.1, -1.1) is fr3
        assert fa.find_component(1.1, -1.1) is fr4

        assert fa.find_component(0.25, 0.25) is cr0
        assert fa.find_component(-0.25, 0.25) is cr0
        assert fa.find_component(-0.25, -0.25) is cr0
        assert fa.find_component(0.25, -0.25) is cr0

        assert fa.find_component(2, 2) is default_component
        assert fa.find_component(-2, 2) is default_component
        assert fa.find_component(-2, -2) is default_component
        assert fa.find_component(2, -2) is default_component


if __name__ == '__main__':
    unittest.main()
