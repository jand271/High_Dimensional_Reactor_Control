from scipy.io import loadmat

from model_reduction.demo.utils import compute_first_metrics

if __name__ == "__main__":
    model = loadmat('hovland_et_al_model.mat')
    C = model["Cp"]

    snapshots = loadmat('snapshots_of_hovland_el_al.mat')
    X = snapshots["X"]

    rank_list = [1, 2, 3, 4, 5, 6]
    max_rank_buithanh = 2

    compute_first_metrics(X, C, rank_list, max_rank_buithanh=2)
