# ==================================================================================================
# Run with:
# mpirun -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallels.py
# ==================================================================================================


from pathlib import Path

import numpy as np

from pysde import increments, runner, schemes, storages


# ==================================================================================================
def drift(x, t):
    return -2 * np.power(x, 3) + 3 * x


def diffusion(x, t):
    return np.sqrt(np.power(x, 2) + 2)


rng = np.random.default_rng(seed=0)
x0 = rng.normal(loc=0, scale=0.5, size=(1, 100000))
t0 = 0.0
dt = 0.01
num_steps = 101


sde_runner = runner.ParallelRunner(
    drift_function=drift,
    diffusion_function=diffusion,
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.BrownianIncrement,
    storage_type=storages.NumpyStorage,
    seed=0,
    stride=100,
    save_directory=Path("../results_example_parallel/result"),
    avoid_race_condition=True,
)
storage = sde_runner.run(x0, t0, dt, num_steps, progress_bar=True)
storage.save()
