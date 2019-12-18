import unittest
from fuel_assembly.material import Material


class TestMaterial(unittest.TestCase):
    def test_contructor(self):
        Material()


if __name__ == '__main__':
    unittest.main()
