import unittest
from assembly_construction.material import Material
from assembly_construction.bar import Bar, SquareBar


class TestSquareBar(unittest.TestCase):
    def test_constructor(self):
        b1 = SquareBar(0, 0, 1)
        assert (0, 0) == b1.get_position()

        b2 = SquareBar(0, 0, 1, plot_color="k")
        assert "k" == b2.get_plot_color()

        material = Material()
        b3 = SquareBar(0, 0, 1, plot_color="k", material=material)
        assert material == b3.get_material()

    def test_is_point_within(self):
        r = SquareBar(0, 0, 1)

        assert r.is_point_within(0, 0)
        assert r.is_point_within(0.5, 0.5)
        assert r.is_point_within(0.5, -0.5)
        assert r.is_point_within(-0.5, 0.5)
        assert r.is_point_within(-0.5, -0.5)
        assert r.is_point_within(0, 0.25)
        assert r.is_point_within(0.25, 0)
        assert r.is_point_within(0, -0.25)
        assert r.is_point_within(-0.25, 0)

        assert not r.is_point_within(0, 2)
        assert not r.is_point_within(2, 0)
        assert not r.is_point_within(0, -2)
        assert not r.is_point_within(-2, 0)
        assert not r.is_point_within(1.5, 1.5)
        assert not r.is_point_within(1.5, -1.5)
        assert not r.is_point_within(-1.5, 1.5)
        assert not r.is_point_within(-1.5, -1.5)


class TestBar(unittest.TestCase):
    def test_constructor(self):
        b1 = Bar(0, 0, 1, 2)
        assert (0, 0) == b1.get_position()

        b2 = Bar(0, 0, 1, 2, plot_color="k")
        assert "k" == b2.get_plot_color()

        material = Material()
        b3 = Bar(0, 0, 1, 2, plot_color="k", material=material)
        assert material == b3.get_material()

    def test_is_point_within(self):
        r = Bar(0, 0, 1, 2)

        assert r.is_point_within(0, 0)
        assert r.is_point_within(0.5, 1.0)
        assert r.is_point_within(0.5, -1.0)
        assert r.is_point_within(-0.5, 1.0)
        assert r.is_point_within(-0.5, -1.0)
        assert r.is_point_within(0, 0.5)
        assert r.is_point_within(0.25, 0)
        assert r.is_point_within(0, -0.5)
        assert r.is_point_within(-0.25, 0)

        assert not r.is_point_within(0, 4)
        assert not r.is_point_within(2, 0)
        assert not r.is_point_within(0, -4)
        assert not r.is_point_within(-2, 0)
        assert not r.is_point_within(1.5, 3.0)
        assert not r.is_point_within(1.5, -3)
        assert not r.is_point_within(-1.5, 3.0)
        assert not r.is_point_within(-1.5, -3)


if __name__ == "__main__":
    unittest.main()
