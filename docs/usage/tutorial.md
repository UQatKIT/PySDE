# Input Data

```python
import numpy as np

def drift(x, t):
    return -x

def diffusion(x, t):
    return 1 * np.identity(1)
```

```python
x0 = np.ones((1, 100000))
t0 = 0.0
dt = 0.01
num_steps = 1000
```

<br>
# Modular

```python
storage = storages.NumpyStorage(stride=100)
brownian_increment = increments.WienerIncrement(seed=0)
scheme = schemes.ExplicitEulerMaruyamaScheme(drift, diffusion, brownian_increment)
sde_integrator = integrator.SDEIntegrator(scheme, storage)
```

```python
result = sde_integrator.run(
    initial_state=x0, initial_time=t0, step_size=dt, num_steps=num_steps, progress_bar=True
)
```

<br>
# Builder

```python
from pysde import runner

sde_integrator = runner.IntegratorBuilder.build_integrator(
    drift_function=drift,
    diffusion_function=diffusion,
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.WienerIncrement,
    storage_type=storages.NumpyStorage,
    seed=0,
    stride=100,
    save_directory=Path("data"),
)
result = sde_integrator.run(x0, t0, dt, num_steps, progress_bar=True)
```

<br>
# Parallel Runner

```python
sde_runner = runner.ParallelRunner(
    drift_function=drift,
    diffusion_function=diffusion,
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.WienerIncrement,
    storage_type=storages.NumpyStorage,
    seed=0,
    stride=100,
    save_directory=Path("data"),
)
result = sde_runner.run(x0, t0, dt, num_steps, progress_bar=True)
```

```bash
mpirun -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallel_runner.py
```