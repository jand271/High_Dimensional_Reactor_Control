from assembly_construction.premade_fuel_assemblies import *
from modeling.temperature_fem_model import HeatExchangerFEMModel
from controllers.heat_exchanger_mpc_controller import HeatExchangerMPCController
from fenics import *

if __name__ == "__main__":

    import time

    start_time = time.time()

    fa = ComponentAssemblyB()

    dt = 1  # works with dt = 1!!!!!
    model = HeatExchangerFEMModel(fa, dt, nx=30, ny=30)

    Q_diagonal = np.ones((model.get_number_of_vertices(),))
    # for rod in heating_rods and cooling_rods:
    for rod in fa:
        Q_diagonal[model._vertex_hash_map[rod]] = 0

    A, B, f = model.state_transition_model()
    from controllers.dlqr import *

    controller = TrackingAffineDLQR(A, B, f, xbar=500 * np.ones((A.shape[0],)), max_iter=15)
    # controller = AffineDLQR(A, B, f)

    # controller = HeatExchangerMPCController(
    #     model,
    #     500,
    #     max_removal_power_density=1000,
    #     Q=np.diag(Q_diagonal))

    fa.plot()
    model.step_time()
    model._component_hash_map.plot()
    plt.title("Fuel Assembly and Mesh")
    plt.savefig("Fuel_Assembly_and_Mesh.png")
    plt.clf()

    t = 0
    for i in range(10):
        t += dt

        T = model.step_time()
        q_dots = controller.update_then_calculate_optimal_actuation(T.vector().get_local())

        for q_dot, component in zip(q_dots, fa.get_component_set("controllable_q_dot")):
            component.set_volumetric_power_density(q_dot)

    print("--- %s seconds ---" % (time.time() - start_time))

    p = plot(T)
    plt.colorbar(p, format="%.1f K")
    for component in fa.get_component_set("controllable_q_dot"):
        plt.annotate("{0:.1f}".format(component.get_volumetric_power_density()), component.get_position())

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Fuel Assembly Steady State Temperature at t={:.2f}s".format(t))
    plt.savefig("test.png")
