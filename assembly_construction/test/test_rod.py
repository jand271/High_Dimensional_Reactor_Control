import unittest
from assembly_construction.rod import Rod, ControlRod, FuelRod
from assembly_construction.material import Material, UO2, HighBoronSteel


class TestRod(unittest.TestCase):
    def test_constructor(self):
        r1 = Rod(0, 0, 1)
        assert (0, 0) == r1.get_position()

        r2 = Rod(0, 0, 1, plot_color='k')
        assert 'k' == r2.get_plot_color()

        material = Material()
        r3 = Rod(0, 0, 1, plot_color='k', material=material)
        assert material == r3.get_material()

        fr = FuelRod(0, 0, 1)
        assert isinstance(fr.get_material(), UO2)

        cr = ControlRod(0, 0, 1)
        assert isinstance(cr.get_material(), HighBoronSteel)

    def test_is_point_within(self):
        r = Rod(0, 0, 1)

        assert r.is_point_within(0, 0)
        assert r.is_point_within(0, 1)
        assert r.is_point_within(1, 0)
        assert r.is_point_within(0, -1)
        assert r.is_point_within(-1, 0)
        assert r.is_point_within(0.5, 0.5)
        assert r.is_point_within(0.5, -0.5)
        assert r.is_point_within(-0.5, 0.5)
        assert r.is_point_within(-0.5, -0.5)

        assert not r.is_point_within(0, 2)
        assert not r.is_point_within(2, 0)
        assert not r.is_point_within(0, -2)
        assert not r.is_point_within(-2, 0)
        assert not r.is_point_within(1.5, 1.5)
        assert not r.is_point_within(1.5, -1.5)
        assert not r.is_point_within(-1.5, 1.5)
        assert not r.is_point_within(-1.5, -1.5)


if __name__ == '__main__':
    unittest.main()
