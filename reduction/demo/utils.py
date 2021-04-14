import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from os.path import join
from reduction import *


def compute_and_save_rom(reduction, *args, **kwargs):
    """ compute and save rom with reduction strategy with solving time """
    begin_time = time.time()
    r = reduction(*args, **kwargs)
    end_time = time.time()
    r.reduction_time = end_time - begin_time
    r.save_reduction_basis("output")


def compute_and_save_roms(A, B, C, X, r):
    """ compute and save all roms """

    reduction_list = [
        BalancedTruncation,
        Carlberg,
        GradientDescentWeightedPODModelReduction,
        MDModelReduction,
        PODModelReduction,
    ]
    args_list = [(A, B, C, r), (X, r, C), (X, r, C), (A, r), (X, r)]

    for reduction, args in zip(reduction_list, args_list):
        compute_and_save_rom(reduction, *args)


def projector(W, V):
    return V @ W.T
    # return V @ np.linalg.inv(W.T @ V) @ W.T


def orthogonal_error(X, C, W, V):
    Pi = projector(W, V)
    I = np.eye(Pi.shape[0])
    return np.sum(np.linalg.norm(C @ (I - Pi) @ X, axis=0))


def reduction_error(X, U, W, V, A, B, C, f=None):
    (nx, ns) = X.shape
    (_, nu) = B.shape
    nxr = V.shape[1]

    if f is None:
        f = np.zeros((nx,))

    Ar = W.T @ A @ V
    Br = W.T @ B
    fr = W.T @ f

    Q = np.zeros((nxr, ns))

    Q[:, 0] = W.T @ X[:, 0]
    error = 0
    for i in range(1, ns):
        Q[:, i] = Ar @ Q[:, i - 1] + Br @ U[:, i - 1] + fr
        error += np.linalg.norm(C @ (X[:, i] - V @ Q[:, i]))

    return error


def plot_orthogonal_error(X, C, reduction_list, rank_list, max_rank_buithanh=None):
    if max_rank_buithanh is None:
        max_rank_buithanh = max(rank_list)

    error_per_rank_per_method = {}
    unweighted_error_per_rank_per_method = {}

    for reduction in reduction_list:

        error_per_rank_per_method[reduction] = []
        unweighted_error_per_rank_per_method[reduction] = []

        for rank in rank_list:

            if reduction == "buithanh" and rank > max_rank_buithanh:
                continue

            directory = join("output", reduction + "_rank_" + str(rank))

            data = loadmat(directory)
            W = data["W"]
            V = data["V"]

            if reduction != "buithanh":
                t = float(data["compute_time_s"])

            error_per_rank_per_method[reduction].append(orthogonal_error(X, C, W, V))
            unweighted_error_per_rank_per_method[reduction].append(orthogonal_error(X, np.eye(X.shape[0]), W, V))

    plt.title("Orthogonal Error Per Rank Per Rom")
    for reduction in reduction_list:
        if reduction == "buithanh":
            plt.plot(
                np.arange(1, max_rank_buithanh + 1),
                np.array(unweighted_error_per_rank_per_method[reduction]),
                label=reduction,
            )
        else:
            plt.plot(np.array(rank_list), np.array(unweighted_error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc="best")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("unweighted_orthogonal_error.png")
    plt.show()

    plt.clf()

    plt.title("Goal-oriented Orthogonal Error Per Rank Per Rom")
    for reduction in reduction_list:
        if reduction == "buithanh":
            plt.plot(
                np.arange(1, max_rank_buithanh + 1), np.array(error_per_rank_per_method[reduction]), label=reduction
            )
        else:
            plt.plot(np.array(rank_list), np.array(error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc="best")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("weighted_orthogonal_error.png")
    plt.show()


def plot_reduction_error(X, U, A, B, C, reduction_list, rank_list, f=None, max_rank_buithanh=None):
    if max_rank_buithanh is None:
        max_rank_buithanh = max(rank_list)

    error_per_rank_per_method = {}
    unweighted_error_per_rank_per_method = {}
    time_per_rank_per_method = {}

    for reduction in reduction_list:

        error_per_rank_per_method[reduction] = []
        unweighted_error_per_rank_per_method[reduction] = []
        time_per_rank_per_method[reduction] = []

        for rank in rank_list:

            if reduction == "buithanh" and rank > max_rank_buithanh:
                continue

            directory = join("output", reduction + "_rank_" + str(rank))

            data = loadmat(directory)
            W = data["W"]
            V = data["V"]

            error_per_rank_per_method[reduction].append(reduction_error(X, U, W, V, A, B, C, f=f))
            unweighted_error_per_rank_per_method[reduction].append(
                reduction_error(X, U, W, V, A, B, np.eye(X.shape[0]), f=f)
            )

    plt.title("Reduction Error Per Rank Per Rom")
    for reduction in reduction_list:
        if reduction == "buithanh":
            plt.plot(
                np.arange(1, max_rank_buithanh + 1),
                np.array(unweighted_error_per_rank_per_method[reduction]),
                label=reduction,
            )
        else:
            plt.plot(np.array(rank_list), np.array(unweighted_error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc="best")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("unweighted_reduction_error.png")
    plt.show()

    plt.clf()

    plt.title("Goal-oriented Reduction Error Per Rank Per Rom")
    for reduction in reduction_list:
        if reduction == "buithanh":
            plt.plot(
                np.arange(1, max_rank_buithanh + 1), np.array(error_per_rank_per_method[reduction]), label=reduction
            )
        else:
            plt.plot(np.array(rank_list), np.array(error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc="best")
    plt.xlabel("rank")
    plt.ylabel("error")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("weighted_reduction_error.png")
    plt.show()


def compute_controller_cost(controller, W, V, x0, A, B, Q, R=None, f=None, xbar=None, ubar=None, time_steps=100):
    """
    Computes the LQR controller cost over a number of time steps
    :param controller: the controller to measure cost
    :param W: left reduction basis
    :param V: right reduction basis
    :param x0: starting state
    :param A: state transition matrix
    :param B: input to state matrix
    :param Q: state cost
    :param R: control cost
    :param f: affine dyanmic term
    :param xbar: state driven
    :param ubar: corresponding ubar to xbar
    :param time_steps: number of time steps to consider
    :return: total cost over time
    """

    if R is None:
        R = np.eye(B.shape[1])

    if f is None:
        f = np.zeros((A.shape[0],))

    if xbar is None:
        xbar = np.zeros((A.shape[0],))

    if ubar is None:
        ubar = np.zeros((R.shape[1],))

    J = np.zeros((time_steps - 1,))  # cost over time
    X = np.zeros((A.shape[0], time_steps))  # state over time

    X[:, 0] = x0  # set starting state

    for i in range(1, time_steps):
        u = controller.update_then_calculate_optimal_actuation(W.T @ X[:, i - 1])  # compute controller actuation
        X[:, i] = A @ X[:, i - 1] + B @ u + f  # apply actuation to compute next state
        delx = X[:, i] - xbar
        delu = u - ubar
        J[i - 1] = delx.T @ Q @ delx + delu.T @ R @ delu  # compute cost of state and actuation

    return np.sum(J)  # return sum of costs
