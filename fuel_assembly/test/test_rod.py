import unittest
from fuel_assembly.rod import Rod, ControlRod, FuelRod
from fuel_assembly.material import Material, UO2, HighBoronSteel


class TestRod(unittest.TestCase):
    def test_contructor(self):
        r1 = Rod(0, 0, 1)
        assert (0, 0) == r1.get_position()

        r2 = Rod(0, 0, 1, plot_color='k')

        material = Material
        r3 = Rod(0, 0, 1, plot_color='k', material=material)
        assert material == r3.get_material()

        fr = FuelRod(0, 0, 1)
        assert UO2 == fr.get_material()

        cr = ControlRod(0, 0, 1)
        assert HighBoronSteel == cr.get_material()


if __name__ == '__main__':
    unittest.main()
