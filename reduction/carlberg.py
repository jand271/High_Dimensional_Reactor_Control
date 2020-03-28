import numpy as np
from scipy.linalg import eigh, sqrtm
from reduction.weighted_pod_model_reduction import WeightedPODModelReduction


class Carlberg(WeightedPODModelReduction):

    def compute_reduction_basis(self):
        Theta = self.C_full.T @ self.C_full
        Sigma_squared, V = eigh(Theta)
        #Sigma_squared = np.maximum(Sigma_squared, 1e-10)
        Theta = V @ np.diag(Sigma_squared) @ V.T
        self.V = self.algorithm_2(self.r, self.X, Theta)
        self.W = (np.linalg.inv(self.V.T @ Theta @ self.V) @ self.V.T @ Theta).T

    def compute_reduction_bases(self):
        self.compute_reduction_basis()

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
        X_bar = square_root_Theta @ X
        U, S, VT = WeightedPODModelReduction.compute_truncated_svd(X_bar, rank, compute_full=True)
        SinvV = np.divide(VT.T, S)
        return X @ SinvV
