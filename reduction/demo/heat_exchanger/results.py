from scipy.io import loadmat

from reduction.demo.utils import compute_first_metrics

if __name__ == "__main__":
    model = loadmat('heat_exchanger_model.mat')
    C = model["C"]

    snapshots = loadmat('snapshots_heat_exchanger_model.mat')
    X = snapshots["X"]

    reduction_list = [
        'buithanh',
        'carlberg',
        'grad_descent',
        'modal',
        'pod']

    rank_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    max_rank_buithanh = 0

    compute_first_metrics(X, C, reduction_list, rank_list, max_rank_buithanh=max_rank_buithanh)
