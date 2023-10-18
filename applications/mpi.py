import os

import numpy as np
from mpi4py import MPI

from pysde import integrators, schemes, stochastic_integrals, storages

def main() -> None:
    # Settings for MPI
    mpi_communicator = MPI.COMM_WORLD
    local_rank = mpi_communicator.Get_rank()

    # Settings for integration
    start_time=0
    step_size=1e-3
    num_steps=500001
    initial_state = 0.1 * local_rank * np.zeros((2, 10))
    show_progressbar = True

    # Settings for stochastic integral
    noise_dim = 2
    seed = local_rank

    # Settings for storage
    save_directory = os.path.join("result_data_numpy", f"process_{local_rank}")

    # Drift function
    def drift(current_state: np.ndarray, current_time: float):
        return -current_state

    # Diffusion function
    def diffusion(current_state: np.ndarray, current_time: float):
        return np.ones_like(current_state)
    
    # Execute Simulation
    stochastic_integral = stochastic_integrals.ItoStochasticIntegral(noise_dim, seed)
    scheme = schemes.ExplicitEulerMaruyamaScheme(drift, diffusion, stochastic_integral)
    storage = storages.NumpyStorage(save_directory)
    integrator = integrators.StaticIntegrator(scheme, storage, show_progressbar)
    integrator.run(start_time, step_size, num_steps, initial_state)

if __name__ == "__main__":
    main()