# Run mpiexec -n 2 --map-by slot:PE=4 python -m mpi4py parallel_runner.py

from pathlib import Path

import numpy as np

from pysde import increments, runner, schemes, storages


# ==================================================================================================
def drift(x, _t):
    return -x


def diffusion(_x, _t):
    return 1 * np.identity(1)


x0 = np.ones((1, 100000))
t0 = 0.0
dt = 0.01
num_steps = 1000

settings = runner.Settings(
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.BrownianIncrement,
    increment_seed=0,
    storage_type=storages.NumpyStorage,
    storage_stride=100,
    storage_save_directory=Path("data"),
)

sde_runner = runner.ParallelRunner(settings, drift, diffusion)
result = sde_runner.run(x0, t0, dt, num_steps, progress_bar=True)
