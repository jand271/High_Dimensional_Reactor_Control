import unittest
from assembly_construction.component import UnshapedComponent
from assembly_construction.material import Material
from assembly_construction.material import UO2


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
