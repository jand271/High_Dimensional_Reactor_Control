from scipy.io import loadmat
from model_reduction.demo.utils import compute_and_save_roms

if __name__ == "__main__":

    model = loadmat('hovland_et_al_model.mat')
    snapshots = loadmat('snapshots_of_hovland_el_al.mat')

    A = model["Ap"]
    B = model["Bp"]
    C = model["Cp"]

    X = snapshots["X"]

    for r in range(6):
        compute_and_save_roms(A, B, C, X, r + 1)
