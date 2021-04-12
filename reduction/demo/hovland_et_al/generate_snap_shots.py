import numpy as np
from scipy.io import loadmat, savemat

if __name__ == "__main__":
    model = loadmat("hovland_et_al_model.mat")

    A = model["Ap"]
    B = model["Bp"]
    C = model["Cp"]

    nx = A.shape[0]
    nu = B.shape[1]

    ns = 40

    X = np.zeros((nx, ns))
    X[:, 0] = np.array([-0.9044, -9.1380, -2.5036, 0.6696, -0.0821, -4.0350])

    U = np.ones((nu, ns - 1))

    for i in range(1, ns):
        X[:, i] = A @ X[:, i - 1] + B @ U[:, i - 1]

    savemat("snapshots_of_hovland_el_al.mat", {"X": X, "U": U})
