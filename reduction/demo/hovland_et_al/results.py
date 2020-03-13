from scipy.io import loadmat

from reduction.demo.utils import plot_orthogonal_error, plot_reduction_error

if __name__ == "__main__":
    model = loadmat('hovland_et_al_model.mat')
    C = model["Cp"]

    snapshots = loadmat('snapshots_of_hovland_el_al.mat')
    X = snapshots["X"]

    reduction_list = [
        'balanced_truncation',
        'buithanh',
        'carlberg',
        'grad_descent',
        'modal',
        'pod']

    rank_list = [1, 2, 3, 4, 5]

    plot_orthogonal_error(X, C, reduction_list, rank_list)
