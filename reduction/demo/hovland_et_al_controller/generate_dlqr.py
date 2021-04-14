from scipy.io import loadmat, savemat
from controllers.dlqr import DLQR

if __name__ == "__main__":
    model = loadmat("hovland_et_al_model.mat")

    Ap = model["Ap"]
    Bp = model["Bp"]
    Q = model["Q"]
    R = model["R"]

    F = DLQR(Ap, Bp, Q=Q, R=R).F

    savemat("hovland_et_al_model.mat", {"Ap": Ap, "Bp": Bp, "Q": Q, "R": R, "Cp": F})
