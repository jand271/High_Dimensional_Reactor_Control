import tensorflow as tf
from model_reduction.pod_model_reduction import PODModelReduction
import numpy as np


class WeightedPODModelReduction(PODModelReduction):

    def __init__(self, *args, ridge_regularization=1, iteration_count=100):
        """
        Constructor for weighted POD model reduction
        :param args: See PODModelReduction parameters
        :param ridge_regularization: ridge regression regularization term scalar
        :param iteration_count: number of descent iterations
        """

        """ Check constructor inputs """
        assert isinstance(ridge_regularization, int) or isinstance(ridge_regularization, float), \
            "ridge_regularization must be an int or float"
        assert isinstance(iteration_count, int), "iteration_count must be an int"

        self.ridge_regularization = ridge_regularization
        self.iteration_count = iteration_count

        super().__init__(*args)

    def compute_reduction_basis(self):
        """
        Completes the following minimization problem to compute the reduction basis
        \begin{equation}
            \arg \min_V
                \left|\left|C\left(I - VV^T\right)X\right|\right|_{||} + \beta \left|\left| V \right|\right|_{||} \\
        \end{equation}
        Where $\left|\left|\cdot \right|\right|_{||}$ is the Frobenius normal, additionaly normalized by the number of
        elements within the norm, additionally returns the closest orthogonal matrix to V via Orthogonal Procrustes
        Problem.
        """

        super().compute_reduction_basis()  # compute vanilla POD basis and assign to self.V
        V = tf.Variable(self.V, name='V', dtype=tf.float32)  # let V starting point be vanilla POD basis
        # V = tf.Variable(np.random.normal(size=(self.nx, self.r)), name='V', dtype=tf.float32)

        X = tf.convert_to_tensor(self.X, dtype=tf.float32)

        """ compute cost function normalization constants nx, ny, r, and ns """
        (ny, nx1) = self.C_full.shape
        (nx2, r) = V.shape
        (nx3, ns) = X.shape
        assert nx1 == nx2 and nx2 == nx3
        nx = nx1

        opt = tf.optimizers.Adadelta()  # select an optimizer

        def loss():
            """ Weighted SVD loss function with ridge regularization,
            normalized by the number of elements within the norm """
            Q = self.C_full @ (tf.eye(self.nx) - V @ tf.transpose(V)) @ X
            return tf.norm(Q) / (ny * ns) + self.ridge_regularization * tf.norm(V) / (nx * r)

        for i in range(self.iteration_count):
            # if i % (self.iteration_count // 10) == 0:
            #     print(loss().numpy())
            opt.minimize(loss, [V])

        """ Compute closest orthonormal matrix to V via the Orthogonal Procrustes Problem """
        V = V.value().numpy()
        U, S, ZT = np.linalg.svd(V)
        I = np.eye(*V.shape)
        self.V = U @ I @ ZT
