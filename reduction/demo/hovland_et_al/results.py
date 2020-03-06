from scipy.io import loadmat

from reduction.demo.utils import compute_first_metrics

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

    compute_first_metrics(X, C, reduction_list, rank_list)
