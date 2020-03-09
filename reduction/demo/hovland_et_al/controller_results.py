import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join
from reduction.model_reduction import ModelReduction
from controllers.dlqr import DLQR
from reduction.demo.utils import compute_controller_cost


def main():
    model = loadmat('hovland_et_al_model.mat')
    A = model["Ap"]
    B = model["Bp"]
    Q = model["Q"]
    R = model["R"]

    snapshots = loadmat('snapshots_of_hovland_el_al.mat')
    X = snapshots["X"]
    x0 = X[:, 0]

    reduction_list = [
        'balanced_truncation',
        'carlberg',
        'grad_descent',
        'modal',
        'pod']

    rank_list = [1, 2, 3, 4, 5]

    controller_cost_per_rank = {}
    rank_domain = {}

    for reduction in reduction_list:
        controller_cost_per_rank[reduction] = []
        rank_domain[reduction] = []

        for rank in rank_list:
            data = loadmat(join('output', reduction + '_rank_' + str(rank)))

            W = data['W']
            V = data['V']

            (Ar, Br, fr, Cr) = ModelReduction.compute_state_space_reduction(A, B, None, None, V, W)

            Qr = V.T @ Q @ V

            controller = DLQR(Ar, Br, Q=Qr, R=R)

            cost = compute_controller_cost(controller, V, x0, A, B, Q, R)

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
