import unittest
from fuel_assembly.component import UnshapedComponent
from fuel_assembly.material import Material


class TestComponent(unittest.TestCase):
    def test_contructor(self):
        UnshapedComponent()

        material = Material
        c = UnshapedComponent(material=material)
        assert material == c.get_material()


if __name__ == '__main__':
    unittest.main()
