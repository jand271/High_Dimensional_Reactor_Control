import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
from os.path import join
from reduction import *


def compute_and_save_rom(reduction, *args, **kwargs):
    """ compute and save rom with reduction strategy with solving time """
    begin_time = time.time()
    r = reduction(*args, **kwargs)
    end_time = time.time()
    r.reduction_time = end_time - begin_time
    r.save_reduction_basis('output')


def compute_and_save_roms(A, B, C, X, r, skip=None):
    """ compute and save all roms """

    reduction_list = [BalancedTruncation,
                      Carlberg,
                      GradientDescentWeightedPODModelReduction,
                      MDModelReduction,
                      PODModelReduction]
    args_list = [
        (A, B, C, r),
        (X, r, C),
        (X, r, C),
        (A, r),
        (X, r)]

    if skip is not None:
        reduction_list.pop(skip)
        args_list.pop(skip)

    for reduction, args in zip(reduction_list, args_list):
        compute_and_save_rom(reduction, *args)


def projector(W, V):
    return V @ np.linalg.inv(W.T @ V) @ W.T


def orthogonal_error(X, C, W, V):
    Pi = projector(W, V)
    I = np.eye(Pi.shape[0])
    return np.sum(np.linalg.norm(C @ (I - Pi) @ X, axis=0))


def compute_first_metrics(X, C, reduction_list, rank_list, max_rank_buithanh=None):
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

            if reduction == 'buithanh' and rank > max_rank_buithanh:
                continue

            directory = join('output', reduction + '_rank_' + str(rank))

            data = loadmat(directory)
            W = data['W']
            V = data['V']

            if reduction != 'buithanh':
                t = float(data['compute_time_s'])

            error_per_rank_per_method[reduction].append(orthogonal_error(X, C, W, V))
            unweighted_error_per_rank_per_method[reduction].append(orthogonal_error(X, np.eye(X.shape[0]), W, V))
            time_per_rank_per_method[reduction].append(t)

    plt.title('Unweighted Orthogonal Error Per Rank Per Rom')
    for reduction in reduction_list:
        if reduction == 'buithanh':
            plt.plot(np.arange(1, max_rank_buithanh + 1), np.array(unweighted_error_per_rank_per_method[reduction]),
                     label=reduction)
        else:
            plt.plot(np.array(rank_list), np.array(unweighted_error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc='best')
    plt.xlabel('rank')
    plt.ylabel('error')
    plt.yscale('log')
    plt.savefig('unweighted_orthogonal_error.png')
    plt.show()

    plt.clf()

    plt.title('Weighted Orthogonal Error Per Rank Per Rom')
    for reduction in reduction_list:
        if reduction == 'buithanh':
            plt.plot(np.arange(1, max_rank_buithanh + 1), np.array(error_per_rank_per_method[reduction]),
                     label=reduction)
        else:
            plt.plot(np.array(rank_list), np.array(error_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc='best')
    plt.xlabel('rank')
    plt.ylabel('error')
    plt.yscale('log')
    plt.savefig('weighted_orthogonal_error.png')
    plt.show()

    plt.title('Time Per Rank Per Rom')
    for reduction in reduction_list:
        if reduction == 'buithanh':
            plt.plot(np.arange(1, max_rank_buithanh + 1), np.array(time_per_rank_per_method[reduction]),
                     label=reduction)
        else:
            plt.plot(np.array(rank_list), np.array(time_per_rank_per_method[reduction]), label=reduction)

    plt.legend(loc='best')
    plt.xlabel('rank')
    plt.ylabel('Time [s]')
    plt.savefig('time.png')
    plt.show()
