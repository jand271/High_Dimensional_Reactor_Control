from scipy.linalg import eigh
from control import lqr


def perform_model_reduction(phi, A, B, f, Q=None):
    """
    Performs model reduction phi on dynamic system and state cost matrix

    x^{full} = \Phi x^{reduced}

    :param A: state transition matrix
    :param B: control matrix
    :param f: affine term
    :param Q: quadratic cost matrix associated with state
    :return: reduced system matrices
    """
    print("Model Reduction dimension: {0} -> {1}".format(phi.shape[0], phi.shape[1]))
    Ar = phi.T @ A @ phi
    Br = phi.T @ B
    fr = phi.T @ f
    if Q is not None:
        Q = phi.T @ Q @ phi
    return phi, Ar, Br, fr, Q


def proper_orthogonal_decomposition_model_reduction(A, B, f, dimension, Q=None):
    """
    Performs Proper Orthogonal Decomposition Model Reduction on dynamic system and state cost matrix

    x_{t} = Ax_{t-1} + Bu_t + f

    :param A: state transition matrix
    :param B: control matrix
    :param f: affine term
    :param dimension: number of dimensions of reduced system
    :param Q: quadratic cost matrix associated with state
    :return: reduced system matrices
    """

    # eigen vector diagonalization
    # matrix is square to this is equivalent to SVD; however, method only grabs some of the eigen vectors saving time
    S, phi = eigh(A, eigvals=(A.shape[0] - dimension, A.shape[0] - 1))

    return perform_model_reduction(phi, A, B, f, Q=Q)

def lqr_proper_orthogonal_decomposition_model_reduction(A, B, f, dimension, Q=None):
    """


    :param A: state transition matrix
    :param B: control matrix
    :param f: affine term
    :param dimension: number of dimensions of reduced system
    :param Q: quadratic cost matrix associated with state
    :return: reduced system matrices
    """

    # eigen vector diagonalization
    # matrix is square to this is equivalent to SVD; however, method only grabs some of the eigen vectors saving time
    S, phi = eigh(F, eigvals=(A.shape[0] - dimension, A.shape[0] - 1))

    return perform_model_reduction(phi, A, B, f, Q=Q)