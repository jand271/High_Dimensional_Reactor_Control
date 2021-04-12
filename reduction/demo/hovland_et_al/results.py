from scipy.io import loadmat

from reduction.demo.utils import plot_orthogonal_error, plot_reduction_error

if __name__ == "__main__":
    model = loadmat("hovland_et_al_model.mat")
    A = model["Ap"]
    B = model["Bp"]
    C = model["Cp"]

    snapshots = loadmat("snapshots_of_hovland_el_al.mat")
    X = snapshots["X"]
    U = snapshots["U"]

    reduction_list = ["pod", "carlberg", "buithanh"]

    rank_list = [1, 2, 3, 4, 5]

    plot_orthogonal_error(X, C, reduction_list, rank_list)
    plot_reduction_error(X, U, A, B, C, reduction_list, rank_list)
