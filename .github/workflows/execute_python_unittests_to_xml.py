import unittest
import sys
import xmlrunner

if __name__ == "__main__":

    path_to_execute_tests = sys.argv[1]
    path_to_save_xml_file = sys.argv[2]
    test_suite = unittest.TestLoader().discover(path_to_execute_tests)
    rest_results = xmlrunner.XMLTestRunner(verbosity=2, output=path_to_save_xml_file).run(test_suite)
