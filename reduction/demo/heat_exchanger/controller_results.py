import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join
from reduction.model_reduction import ModelReduction
from controllers.dlqr import TrackingAffineDLQR
from reduction.demo.utils import compute_controller_cost


def main():
    model = loadmat('heat_exchanger_model.mat')
    A = model["A"]
    B = model["B"]
    f = model["f"].flatten()
    Q = np.eye(A.shape[0])

    xbar = 500 * np.ones((A.shape[0]))

    snapshots = loadmat('snapshots_heat_exchanger_model.mat')
    X = snapshots["X"]
    x0 = X[:, 0]

    reduction_list = [
        'carlberg',
        'grad_descent',
        'modal',
        'pod']

    rank_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    full_controller = TrackingAffineDLQR(A, B, f, xbar=xbar, Q=Q)
    #ubar = full_controller.update_then_calculate_optimal_actuation(xbar)
    ubar = np.linalg.lstsq(B, (np.eye(*A.shape) - A) @ xbar - f)[0]
    cost = compute_controller_cost(full_controller, np.eye(A.shape[0]), x0, A, B, Q, xbar=xbar, ubar=ubar)
    plt.axhline(y=cost, label='full-order model', color='k', linestyle='--')

    controller_cost_per_rank = {}
    rank_domain = {}

    for reduction in reduction_list:
        controller_cost_per_rank[reduction] = []
        rank_domain[reduction] = []

        for rank in rank_list:
            data = loadmat(join('output', reduction + '_rank_' + str(rank)))

            W = data['W']
            V = data['V']

            (Ar, Br, fr, Cr) = ModelReduction.compute_state_space_reduction(A, B, f, None, V, W)

            Qr = V.T @ Q @ V

            controller = TrackingAffineDLQR(Ar, Br, fr, Q=Qr, xbar=V.T @ xbar)

            cost = compute_controller_cost(controller, V, x0, A, B, Q, xbar=xbar, ubar=ubar)

            controller_cost_per_rank[reduction].append(cost)
            rank_domain[reduction].append(rank)

    for reduction in reduction_list:
        plt.plot(rank_domain[reduction], controller_cost_per_rank[reduction], label=reduction)

    plt.title('Cost per strategy per rank')
    plt.legend(loc='best')
    plt.xlabel('rank')
    plt.ylabel('cost')
    plt.yscale('log')
    plt.savefig('DLQR_Cost.png')
    plt.show()


if __name__ == "__main__":
    main()
