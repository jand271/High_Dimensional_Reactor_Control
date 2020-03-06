import numpy as np
from scipy.linalg import eigh, sqrtm
from reduction.weighted_pod_model_reduction import WeightedPODModelReduction


class Carlberg(WeightedPODModelReduction):

    def compute_reduction_basis(self):
        self.V = self.algorithm_2(self.r, self.X, self.C_full.T @ self.C_full)

    def __str__(self):
        return 'carlberg_rank_' + str(self.r)

    @staticmethod
    def algorithm_1(rank, X, Theta):
        Theta_bar = X.T @ Theta @ X
        (r, c) = Theta_bar.shape
        Sigma_squared, V = eigh(Theta_bar, eigvals=(c - rank, c - 1))
        SinvV = np.flip(np.divide(V, np.sqrt(Sigma_squared)), 1)
        return X @ SinvV

    @staticmethod
    def algorithm_2(rank, X, Theta):
        square_root_Theta = sqrtm(Theta)
        assert np.sum(np.imag(sqrtm(Theta))) < 1e-6, \
            "np.sum(np.imag(scipy.linalg.sqrtm(Theta))) = {:e} > 1e-6".format(np.sum(np.imag(sqrtm(Theta))))
        square_root_Theta = np.real(square_root_Theta)
        X_bar = square_root_Theta @ X
        U, S, ZT = Carlberg.compute_truncated_svd(X_bar, rank)
        SinvZ = np.divide(ZT.T, S)
        return X @ SinvZ