from model_reduction.model_reduction import ModelReduction


class MDModelReduction(ModelReduction):

    def __init__(self, A, B, C, r):
        """
        Computes vanilla POD model reduction basis of rank r on the input dynamic system with input snapshot matrix
        :param A: state transition matrix
        :param B: input to state matrix
        :param C: output matrix
        :param r: desired rank
        """

        """ Check constructor inputs """
        assert isinstance(r, int), "desired rank r must be an int"

        self.r = r

        super().__init__(A, B, C)

    def compute_reduction_basis(self):
        """ Computes reduction basis from the left singular vectors of state transition matrix """
        self.V = self.compute_svd_left_singular_vectors(self.A_full, self.r)
