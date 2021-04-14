import unittest
from assembly_construction.material import Material


class TestMaterial(unittest.TestCase):
    def test_constructor(self):
        Material()


if __name__ == "__main__":
    unittest.main()
