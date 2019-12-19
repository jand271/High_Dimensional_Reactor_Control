import unittest
from fuel_assembly.rod import Rod, ControlRod, FuelRod
from fuel_assembly.material import Material, UO2, HighBoronSteel


class TestRod(unittest.TestCase):
    def test_constructor(self):
        r1 = Rod(0, 0, 1)
        assert (0, 0) == r1.get_position()

        r2 = Rod(0, 0, 1, plot_color='k')

        material = Material()
        r3 = Rod(0, 0, 1, plot_color='k', material=material)
        assert material == r3.get_material()

        fr = FuelRod(0, 0, 1)
        assert isinstance(fr.get_material(), UO2)

        cr = ControlRod(0, 0, 1)
        assert isinstance(cr.get_material(), HighBoronSteel)


if __name__ == '__main__':
    unittest.main()
