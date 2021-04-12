import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from os.path import join
from reduction.model_reduction import ModelReduction
from controllers.dlqr import DLQR
from reduction.demo.utils import compute_controller_cost


def main(max_rank_buithanh=None):
    model = loadmat("hovland_et_al_model.mat")
    A = model["Ap"]
    B = model["Bp"]
    Q = model["Q"]
    R = model["R"]

    snapshots = loadmat("snapshots_of_hovland_el_al.mat")
    X = snapshots["X"]
    x0 = X[:, 20]

    reduction_list = ["pod", "carlberg", "buithanh"]

    rank_list = [1, 2, 3, 4, 5]

    if max_rank_buithanh is None:
        max_rank_buithanh = max(rank_list)

    controller_cost_per_rank = {}
    rank_domain = {}

    for reduction in reduction_list:
        controller_cost_per_rank[reduction] = []
        rank_domain[reduction] = []

        for rank in rank_list:

            if reduction == "buithanh" and rank > max_rank_buithanh:
                continue

            data = loadmat(join("output", reduction + "_rank_" + str(rank)))

            W = data["W"]
            V = data["V"]

            (Ar, Br, fr, Cr) = ModelReduction.compute_state_space_reduction(A, B, None, None, V, W)

            Qr = V.T @ Q @ V

            controller = DLQR(Ar, Br, Q=Qr, R=R)

            cost = compute_controller_cost(controller, W, V, x0, A, B, Q, R)

            controller_cost_per_rank[reduction].append(cost)
            rank_domain[reduction].append(rank)

    for reduction in reduction_list:
        plt.plot(rank_domain[reduction], controller_cost_per_rank[reduction], label=reduction)

    controller = DLQR(A, B, Q=Q, R=R)
    I = np.eye(A.shape[0])
    cost = compute_controller_cost(controller, I, I, x0, A, B, Q, R)
    plt.axhline(y=cost, label="full-order model", color="k", linestyle="--")

    plt.title("Cost per strategy per rank")
    plt.legend(loc="best")
    plt.xlabel("rank")
    plt.ylabel("cost")
    plt.yscale("log")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("DLQR_Cost.png")
    plt.show()


if __name__ == "__main__":
    main()
