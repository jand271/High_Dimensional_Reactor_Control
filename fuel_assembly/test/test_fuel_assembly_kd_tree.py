import unittest
from fuel_assembly.fuel_assembly import FuelAssembly
from fuel_assembly.rod import ControlRod, FuelRod


class TestFuelAssemblyKDTree(unittest.TestCase):
    def test_simple(self):
        fa = FuelAssembly()

        assert fa.find_component(0, 0) is None

        cr0 = ControlRod(0, 0, 0.5)
        fr1 = FuelRod(1.0, 1.0, 0.5)
        fr2 = FuelRod(-1, 1, 0.5)
        fr3 = FuelRod(-1.0, -1.0, 0.5)
        fr4 = FuelRod(1, -1, 0.5)

        fa.add_rod(cr0)

        assert fa.find_component(0, 0) is cr0

        fa.add_rod(fr1)
        fa.add_rod(fr2)
        fa.add_rod(fr3)
        fa.add_rod(fr4)

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


if __name__ == '__main__':
    unittest.main()
