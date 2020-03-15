from scipy.io import loadmat

from reduction.demo.utils import plot_orthogonal_error, plot_reduction_error

if __name__ == "__main__":
    model = loadmat('heat_exchanger_model.mat')
    A = model["A"]
    B = model["B"]
    C = model["C"]
    f = model["f"].flatten()

    snapshots = loadmat('snapshots_heat_exchanger_model.mat')
    X = snapshots["X"]
    U = snapshots["U"]

    reduction_list = [
        'pod',
        'carlberg',
        'buithanh']

    rank_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    max_rank_buithanh = 0

    plot_orthogonal_error(X, C, reduction_list, rank_list, max_rank_buithanh=max_rank_buithanh)
    plot_reduction_error(X, U, A, B, C, reduction_list, rank_list, max_rank_buithanh=max_rank_buithanh, f=f)
