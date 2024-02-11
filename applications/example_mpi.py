import os

import numpy as np
from mpi4py import MPI

from pysde import integrators, schemes, stochastic_integrals, storages


def main() -> None:
    # Settings for MPI
    mpi_communicator = MPI.COMM_WORLD
    local_rank = mpi_communicator.Get_rank()

    # Settings for SDE model
    variable_dim = 2
    noise_dim = 2

    def drift(current_state: np.ndarray, current_time: float):
        return -current_state

    def diffusion(current_state: np.ndarray, current_time: float):
        return np.ones((current_state.shape[0], current_state.shape[0], current_state.shape[1]))

    # Settings for integration
    start_time = 0
    step_size = 1e-3
    num_steps = 50001
    initial_state = 0.1 * local_rank * np.ones((2, 10))
    show_progressbar = True

    # Settings for stochastic integral
    seed = local_rank

    # Settings for storage
    save_directory = os.path.join("result_data_numpy", f"process_{local_rank}")

    # Execute Simulation
    stochastic_integral = stochastic_integrals.ItoStochasticIntegral(seed)
    scheme = schemes.ExplicitEulerMaruyamaScheme(
        variable_dim, noise_dim, drift, diffusion, stochastic_integral
    )
    storage = storages.NumpyStorage(save_directory)
    integrator = integrators.StaticIntegrator(scheme, storage, show_progressbar)
    integrator.run(initial_state, start_time, step_size, num_steps)


if __name__ == "__main__":
    main()
