from scipy.io import loadmat
from reduction.demo.utils import compute_and_save_roms
from controllers.dlqr import DLQR

if __name__ == "__main__":

    model = loadmat('hovland_et_al_model.mat')
    snapshots = loadmat('snapshots_of_hovland_el_al.mat')

    A = model["Ap"]
    B = model["Bp"]
    C = model["Cp"]
    Q = model["Q"]
    R = model["R"]

    X = snapshots["X"]

    F = DLQR(A, B, Q=Q, R=R).F

    for r in range(6):
        compute_and_save_roms(A, B, F, X, r + 1)
