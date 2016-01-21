import numpy as np

class NonadiabaticMatrix(object):
    def __init__(self, matrix):
        self.n_electronic_states = len(matrix)
        dct = {}
        for i in range(self.n_electronic_states):
            row = matrix[i]
            assert(len(row) == self.n_electronic_states,
                   "Input matrix not square")
            for j in range(self.n_electronic_states):
                elem = row[i]
                dct[(i,j)] = elem

        self.matrix = matrix
        self.dictionary = dct
        self.check_entries(dct.values())

    @staticmethod
    def check_entries(list_of_entries):
        pass # TODO

    @classmethod
    def from_dictionary(cls, dct, n_electronic_states=None):
        from itertools import chain
        res = cls.__new__()
        cls.check_entries(dct.values())
        if n_electronic_states is None:
            n_electronic_states = max(list(chain.from_iterable(dct.keys())))+1
        res.n_electronic_states = n_electronic_states
        res.dictionary = dct
        matrix = []
        for i in range(self.n_electronic_states):
            row_i = [0.0]*self.n_electronic_states
            for j in range(self.n_electronic_states):
                try:
                    row_i[j] = dct[(i,j)]
                except KeyError:
                    pass
            matrix.append(row_i)
                
        res.matrix = matrix
        return res

    def __get_item__(self, label):
        return self.dictionary[label]

    def numeric_matrix(self, snap):
        pass
        
