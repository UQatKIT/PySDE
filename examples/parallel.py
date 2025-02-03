# Run with: mpirun -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallel_runner.py

from pathlib import Path

import numpy as np

from pysde import increments, runner, schemes, storages


# ==================================================================================================
def drift(x, t):
    return -x


def diffusion(x, t):
    return 1 * np.identity(1)


x0 = np.ones((1, 100000))
t0 = 0.0
dt = 0.01
num_steps = 1000


sde_runner = runner.ParallelRunner(
    drift_function=drift,
    diffusion_function=diffusion,
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.BrownianIncrement,
    storage_type=storages.NumpyStorage,
    seed=0,
    stride=100,
    save_directory=Path("../example_results/data"),
)
result = sde_runner.run(x0, t0, dt, num_steps, progress_bar=True)
