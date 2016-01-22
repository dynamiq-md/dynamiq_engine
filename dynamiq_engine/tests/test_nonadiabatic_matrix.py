import dynamiq_engine as dynq
import numpy as np
from tools import *
import stubs
from dynamiq_engine.nonadiabatic_matrix import *

class testNonadiabaticMatrix(object):
    def setup(self):
        self.numbers_matrix = [[1.5, 2.0], [2.0, 0.0]]
        self.numbers_3x3 = [[1.5, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        self.functions_H = []
        self.mixed_H = []
        self.numbers_dict = {(0,0) : 1.5, (0,1) : 2.0, (1,0) : 2.0}
        self.functions_dict = [0.0]
        pass

    def check_build_from_number_matrix(self):
        na = NonadiabaticMatrix(self.numbers_matrix)
        assert_equal(na.dictionary, self.numbers_dict)

    def check_build_from_mixed_matrix(self):
        raise SkipTest

    def check_build_from_number_dictionary(self):
        na = NonadiabaticMatrix.from_dictionary(self.numbers_dict)
        assert_equal(na.matrix, self.numbers_matrix)
        #assert_equal(na.numeric_matrix(self.snap), self.numbers_matrix)
        na3 = NonadiabaticMatrix.from_dictionary(self.numbers_dict, 3)
        assert_equal(na3.n_electronic_states, 3)
        assert_equal(na3.matrix, self.numbers_3x3)
        raise SkipTest

    def check_build_from_mixed_dictionary(self):
        raise SkipTest

    def check_numeric_matrix_numeric_input(self):
        raise SkipTest

    def check_numeric_matrix_function_input(self):
        raise SkipTest

    def check_get_item(self):
        raise SkipTest

    def bad_entries_string(self):
        raise SkipTest
