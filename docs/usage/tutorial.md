# Input Data

In this tutorial, we go through a simple workflow for generating an ensembles of trajectories with 
PySDE. As an example, we consider a one-dimensional autonomous diffusion process with drift
$b(x)=-2x^3 + 3$ and diffusion $\sigma(x)=\sqrt{x^2+2}$. We can simply define these quantities as
python/numpy functions:

```python
import numpy as np

def drift(x, t):
    return -2 * np.power(x, 3) + 3 * x


def diffusion(x, t):
    return np.sqrt(np.power(x, 2) + 2)
```

Note that both functions take a position and time argument, although the latter is not used. This
is to conform to the interface of the integrator. Furthermore, PySDE requires proper function
definitions with the `def` keyword, lambda functions are not allowed.

In addition to the properties of the SDE, we need to define settings for the integrator itself. As
an initial condition for the trajectory ensemble, we draw 100000 samples from $\mathcal{N}(0,\frac{1}{2})$.
We integrate the 100000 trajectories for 100 time steps, with a step size $\Delta t = 0.01$ and starting
at $t_0=0$,

```python
rng = np.random.default_rng(seed=0)
x0 = rng.normal(loc=0, scale=0.5, size=(1, 100000)).astype(np.float32)
t0 = 0.0
dt = 0.01
num_steps = 101
```
This is everything we need to perform integration with PySDE.

!!! note

    PySDE is floating-point conforming, meaning that the data type of the input array is preserved.
    This can be useful to further reduce the memory load during computations.

<br>
# Modular

For conventional usage, PySDE integrators are built up in a highly modular fashion. To begin with,
we create an intelligent storage object, in this case a [`NumpyStorage`][pysde.storages.NumpyStorage],
```python
storage = storages.NumpyStorage(stride=10)
```
The storage takes a stride for sample storage, and optionally a path for storage on disk. Moving on,
we initialize a standard [`BrownianIncrement`][pysde.increments.BrownianIncrement] for the diffusion
term,
```python
brownian_increment = increments.BrownianIncrement(seed=1)
```
Next, we build an integration scheme with drift, diffusion, and Brownian increment. We employ
the [`ExplicitEulerMaruyamaScheme`][pysde.schemes.ExplicitEulerMaruyamaScheme],
```python
scheme = schemes.ExplicitEulerMaruyamaScheme(drift, diffusion, brownian_increment)
```
Finally, we initialize an [`SDEIntegrator`][pysde.integrator.SDEIntegrator] object with the scheme
and the storage,
```python
sde_integrator = integrator.SDEIntegrator(scheme, storage)
```

Integration is invoked with the integrators `run` method,
```python
storage = sde_integrator.run(
    initial_state=x0, initial_time=t0, step_size=dt, num_steps=num_steps, progress_bar=True
)
```
A run returns the filled storage object, which can be used for further data processing. The `values`
property returns a numpy-like handle to a `time` and a `data` array. Time is one-dimensional,
whereas the data has dimension
$\texttt{physical_dimension} \times \texttt{number_trajectories} \times \texttt{number_timesteps}$.

We can visualize the results for the initial and final states of the ensemble in a histogram:

```python
times, data = storage.values
_, ax = plt.subplots(figsize=(5, 5), layout="constrained")
ax.hist(data[0, :, 0], bins=30, density=True, alpha=0.75, label=rf"$t={times[0]:.1f}$")
ax.hist(data[0, :, -1], bins=30, density=True, alpha=0.75, label=rf"$t={times[-1]:.1f}$")
ax.set_xlim((-3, 3))
ax.set_ylim((0, 0.8))
ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8))
ax.legend()
```

<figure markdown>
![samples](../images/tutorial_result.png){ width="500" style="display: inline-block" }
</figure>


<br>
# Parallel Runner

The above code can simply be run in an MPI environment using, for instance [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/).
PySDE provides a convenience wrapper to automatically perform integration on distributed memory
architectures. The [`ParallelRunner`][pysde.runner.ParallelRunner] facade encapsulates all necessary
MPI routines (along with the storage objects that are safe for parallel execution). The parallel
runner takes in drift and diffusion functions, as well as integration schemes, increment, and 
storage types. In addition, all constructor arguments for these components need to be provided. The parallel
runner will automatically match the arguments with the respective components and initialize them
internally via the [`IntegratorBuilder`][pysde.runner.IntegratorBuilder] class. Before, the runner
modifies the seeds and save directory arguments depending on the Id of the invoking process. This
ensures that each process samples different random trajectories.

The only new input argument compared to the serial integration is the `avoid_race_condition` flag.
If this flag is set, only process 0 will attempt to generate a file hierarchy for data storage.
This avoids race conditions when all processes write to the same parent directory, as done with the `ParallelRunner`.

!!! important
    When using the `ParallelRunner` wrapper, all arguments for the different components of the
    integrator need to be supplied explicitly, even the optional ones. Moreover, the arguments need to be
    provided as key-value pairs, with the key exactly matching the argument name for the component
    they are used for.

```python
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
```

The runner is invoked analogously to the serial integrator:
```python
storage = sde_runner.run(x0, t0, dt, num_steps, progress_bar=True)
storage.save()
```
Importantly, before conducting the integration, the parallel runner chunks the input trajectory
ensemble and equally distributes it among the active workers. Each worker integrates a sub-ensemble
and writes into its own storage file, corresponding to the process Id.


Assuming that the above code is located in a file `parallel.py`, we can start a parallel run
(assuming that MPI  is available) with:
```bash
mpirun -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallel.py
```
Two things are important here. Firstly, the special directive `python -m mpi4py` ensures that 
`MPI_INIT` and `MPI_FINALZE` are automatically executed at the appropriate locations in the Python
script. Secondly, we do not only specify the number of processes `<NUM_PROCS>`, but also assign
a fixed number of threads `<NUM_THREADS_PER_PROC>`to each process. This is necessary for Numba to
actually utilize the available system threads in parallel for loops.

!!! warning

    Process 0 reserves two extra threads for system management, so the actually available number
    of threads per process is $(\texttt{number_overall_threads} - 2) / \texttt{number_processes}$.