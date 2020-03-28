import numpy as np
from scipy.io import loadmat
from reduction.carlberg import Carlberg

if __name__ == "__main__":
    model = loadmat('hovland_et_al_model.mat')
    snapshots = loadmat('snapshots_of_hovland_el_al.mat')

    rank = 4
    multiplier = 10

    A = model["Ap"]
    B = model["Bp"]

    X = snapshots["X"]

    for emphasized_state in range(6):
        C = np.zeros((6,6)) # np.eye(6)
        C[emphasized_state, emphasized_state] = multiplier

        reduction = Carlberg(X, rank, C)

        reduction.save_reduction_basis(
            'focused/carlberg_focused_on_' + str(emphasized_state + 1) + '.mat', use_obj_str=False)

    C = np.zeros((6,6)) # np.eye(6)
    C[0, 0] = multiplier
    C[5, 5] = multiplier

    reduction = Carlberg(X, rank, C)

    reduction.save_reduction_basis(
        'focused/carlberg_focused_on_1_and_6.mat', use_obj_str=False)

    C = np.zeros((6,6)) # np.eye(6)

    reduction = Carlberg(X, rank, C)

    reduction.save_reduction_basis(
        'focused/carlberg_none.mat', use_obj_str=False)