from scipy.io import loadmat
from reduction.demo.utils import compute_and_save_roms

if __name__ == "__main__":

    model = loadmat('heat_exchanger_model.mat')
    snapshots = loadmat('snapshots_heat_exchanger_model.mat')

    A = model["A"]
    B = model["B"]
    C = model["C"]

    X = snapshots["X"]

    for r in range(20):
        compute_and_save_roms(A, B, C, X, r + 1, skip=0)
