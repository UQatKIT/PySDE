# Modular

```python
import numpy as np

def drift(x, _t):
    return -x

def diffusion(_x, _t):
    return 1 * np.identity(1)
```

```python
x0 = np.ones((1, 100000))
t0 = 0.0
dt = 0.01
num_steps = 1000
```

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

# Serial Runner

```python

```

```python
from pysde import runner

settings = runner.Settings(
    scheme_type=schemes.ExplicitEulerMaruyamaScheme,
    increment_type=increments.WienerIncrement,
    increment_seed=0,
    storage_type=storages.NumpyStorage,
    storage_stride=100,
    storage_save_directory=Path("data"),
)
```


# Parallel Runner