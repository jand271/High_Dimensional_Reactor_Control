import unittest
from fuel_assembly.component import UnshapedComponent
from fuel_assembly.material import Material
from fuel_assembly.material import UO2


class TestComponent(unittest.TestCase):
    def test_constructor(self):
        UnshapedComponent()

        material = Material()
        c = UnshapedComponent(material=material)
        assert material == c.get_material()

        material = UO2()
        c = UnshapedComponent(material=material)
        assert material == c.get_material()


if __name__ == '__main__':
    unittest.main()
