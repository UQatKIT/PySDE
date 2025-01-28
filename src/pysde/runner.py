"""Builder and Runner for serial and parallel SDE integrator runs.

This modules implements fassades for the convenient execution of SDE runs. This is particularly
useful for parallel runs based on MPI.
The [ParallelRunner][pysde.runner.ParallelRunner] automatically takes care of all the necessary
steps to integrate a large ensemble of trajectories in parallel.

!!! note: kwargs for the Runners

    The [`IntegratorBuilder`][pysde.runner.IntegratorBuilder] provides a very flexible mechanism to
    provide all necessary arguments for the SDE integrator via `**kwargs`. These arguments are
    dynamically scanned for matching patterns within the respective components. This matching
    excludes the integration schemes, as it is assumed that the scheme is initialized from drift,
    diffusion, and random increment only. Furthermore, the `IntegratorBuilder` requires all
    arguments to be specified, as kwargs, even the optional ones.

Classes:
    IntegratorBuilder: Builder for the SDE integrator.
    ParallelRunner: Runner for hybrid parallelism with MPI.
"""

import inspect
from collections.abc import Callable
from numbers import Real
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from beartype.vale import Is
from mpi4py import MPI

from pysde import increments, integrator, schemes, storages

try:
    from mpi4py import MPI

    MPI_LOADED = True
except ImportError:
    MPI_LOADED = False


# ==================================================================================================
class IntegratorBuilder:
    """Builder for the SDE integrator.

    The `IntegratorBuilder` is a simple wrapper plugging together parameter and components for an
    [`SDEIntegrator`][pysde.integrator.SDEIntegrator] object.
    """

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def build_integrator(
        cls,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        scheme_type: type[schemes.BaseScheme],
        increment_type: type[increments.BaseRandomIncrement],
        storage_type: type[storages.BaseStorage],
        **kwargs: Any,  # noqa: ANN401
    ) -> integrator.SDEIntegrator:
        """Assemble an [`SDEIntegrator`][pysde.integrator.SDEIntegrator] object.

        The builder automatically checks that the provided kwargs contain all necessary
        parameters to initialize the scheme, increment, and storage objects.

        Args:
            drift_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Drift function of the SDE (c.f. [`schemes`][pysde.schemes])
            diffusion_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Diffusion function of the SDE (c.f. [`schemes`][pysde.schemes])
            scheme_type (type[schemes.BaseScheme]): Type of the integration scheme
            increment_type (type[increments.BaseRandomIncrement]): Type of the random increment
            storage_type (type[storages.BaseStorage]): Type of the storage object
            **kwargs (dict[str, Any]): Arguments for initializing scheme, increment, and storage
        """
        storage = cls._init_from_kwargs(storage_type, **kwargs)
        increment = cls._init_from_kwargs(increment_type, **kwargs)
        scheme = scheme_type(drift_function, diffusion_function, increment)
        sde_integrator = integrator.SDEIntegrator(scheme, storage)
        return sde_integrator

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def _init_from_kwargs[cotype](
        cls,
        component_type: cotype,
        **kwargs: Any,  # noqa: ANN401
    ) -> cotype:
        """Initialize a component from matching parameters in `**kwargs dict`."""
        init_signature = inspect.signature(component_type.__init__)
        init_args = list(init_signature.parameters.keys())
        init_args.remove("self")
        try:
            component_kwargs = {arg: kwargs[arg] for arg in init_args}
        except KeyError as missing_arg:
            raise KeyError(
                f"Missing argument {missing_arg} for component {component_type}."
            ) from missing_arg
        component = component_type(**component_kwargs)
        return component


# ==================================================================================================
class ParallelRunner:
    """Runner for parallel integration with MPI.

    The `ParallelRunner` object initiates process-parallel integration based on MPI. It is only
    avaailable if PySDE has been installed with the `mpi` option, and an MPI executable is available
    on the system path. The runner utilizes two-level hybrid parallelism. On the process level, it
    partitions the ensemble of trajectories across the available processes. On the thread level, it
    executes thread-parallel for loops over the locally assigned trajectories. This makes scaling
    to large-scale compute clusters easy.

    Internally, the parallel runner adjusts the parameter of the integrator run depending on the
    invoking MPI rank. It then constructs an SDE integrator with the
    [IntegratorBuilder][pysde.runner.IntegratorBuilder] component from the adjusted parameters.
    The adjustments are the following:
    1. Partition ensemble of initial states equally across processes.
    2. Adjust the random seed for the random increment object (add the rank number).
    3. Adjust the save directory for the storage object (add the rank number).

    !!! note: Execution from the command line

        Thread-parallel numba loops do not intelligently recognize the number of available threads
        in an MPI environment. Available threads have therefore by mapped to the MPI processes
        explicitly in the invoking `mpirun` command:
        ```bash
            mpirun -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallel_runner.py
        ```

    !!! warning: Superfluous threads

        The MPI schedular might assign to extra threads to the MPI process of rank 0. The
        NUM_THREADS_PER_PROC parameter should be adjusted accordingly.

    Methods:
        run: Run the SDE integration process, return storage object on given MPI rank.
    """  # noqa: E501

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        scheme_type: type[schemes.BaseScheme],
        increment_type: type[increments.BaseRandomIncrement],
        storage_type: type[storages.BaseStorage],
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the parallel runner.

        Args:
            drift_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Drift function of the SDE (c.f. [`schemes`][pysde.schemes])
            diffusion_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Diffusion function of the SDE (c.f. [`schemes`][pysde.schemes])
            scheme_type (type[schemes.BaseScheme]): Type of the integration scheme
            increment_type (type[increments.BaseRandomIncrement]): Type of the random increment
            storage_type (type[storages.BaseStorage]): Type of the storage object
            **kwargs (dict[str, Any]): Arguments for initializing scheme, increment, and storage

        Raises:
            ImportError: Exception indicating that MPI is required for parallel execution
        """
        if not MPI_LOADED:
            raise ImportError("MPI is required for parallel execution.")
        mpi_communicator = MPI.COMM_WORLD
        self._local_rank = mpi_communicator.Get_rank()
        self._num_processes = mpi_communicator.Get_size()
        kwargs = self._adjust_process_dependent_parameters(**kwargs)
        self._sde_integrator = IntegratorBuilder.build_integrator(
            drift_function, diffusion_function, scheme_type, increment_type, storage_type, **kwargs
        )

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_state: Real | npt.NDArray,
        initial_time: Real,
        step_size: Real,
        num_steps: Annotated[int, Is[lambda x: x > 0]],
        progress_bar: bool = False,
    ) -> storages.BaseStorage:
        r"""Run the SDE integration process, return storage object on given MPI rank.

        Args:
            initial_state (Real | npt.NDArray): Initial state of the system, given for all
                trajectories with shape $d_X \times N$
            initial_time (Real): Initial time $t_0$ of the stochastic process
            step_size (Real): Discrete step size $\delta t$
            num_steps (int): Number of steps to integrate
            progress_bar (bool): Whether to display a progress bar

        Returns:
            storages.BaseStorage: Storage object containing the SDE trajectory data for the
                ensemble subset assigned to the local MPI rank
        """
        local_initial_state = self._partition_initial_state(initial_state)
        result_storage = self._sde_integrator.run(
            local_initial_state, initial_time, step_size, num_steps, progress_bar
        )
        return result_storage

    # ----------------------------------------------------------------------------------------------
    def _adjust_process_dependent_parameters(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Adjust parameters of the integrator run depending on the MPI rank."""
        adjusted_kwargs = kwargs.copy()
        try:
            adjusted_kwargs["seed"] += self._local_rank
            adjusted_kwargs["save_directory"] = kwargs["save_directory"].with_name(
                f"{kwargs['save_directory'].stem}_p{self._local_rank}"
                f"{kwargs['save_directory'].suffix}"
            )
        except KeyError as missing_arg:
            raise KeyError(f"Missing argument {missing_arg}.") from missing_arg
        return adjusted_kwargs

    # ----------------------------------------------------------------------------------------------
    def _partition_initial_state(
        self, initial_state: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Partition trajectory ensemble equally among MPI processes."""
        num_trajectories = initial_state.shape[1]
        partition_size = num_trajectories // self._num_processes
        start_ind = self._local_rank * partition_size
        if self._local_rank == self._num_processes - 1:
            end_ind = num_trajectories
        else:
            end_ind = start_ind + partition_size
        local_initial_state = initial_state[:, start_ind:end_ind]
        return local_initial_state
