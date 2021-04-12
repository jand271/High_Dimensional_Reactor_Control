import sys
from unittest import TestLoader
from xmlrunner import XMLTestRunner
from coverage import Coverage

if __name__ == "__main__":

    path_to_execute_tests = sys.argv[1]
    path_to_save_test_xml_file = sys.argv[2]
    path_so_save_coverage_xml_file = sys.argv[3]

    cov = Coverage()
    cov.start()

    test_suite = TestLoader().discover(path_to_execute_tests)
    test_results = XMLTestRunner(verbosity=2, output=path_to_save_test_xml_file).run(test_suite)

    cov.stop()
    cov.xml_report(outfile=path_so_save_coverage_xml_file)
