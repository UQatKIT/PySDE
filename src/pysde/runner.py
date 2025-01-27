"""Wrappers for serial and parallel SDE integrator runs.

This modules implements fassades for the convenient execution of SDE runs. The runners are
parameterized by a single dataclass. This is particularly useful for parallel runs based on MPI.
The [ParallelRunner][pysde.runner.ParallelRunner] automatacally takes care of all the necessary
steps to integrate a large ensemble of trajectories in parallel.

!!! info
    The wrapper objects are relatively high level, and therefore less stable with respect to changes
    in the interface of low-level components.

Classes:
    Settings: Settings for the setup of a runner object.
    SerialRunner: Runner for serial execution (with thread-parallel for loops over trajectories).
    ParallelRunner: Runner for hybrid parallelism with MPI.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Annotated

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
@dataclass
class Settings:
    """Settings for the setup of a runner object.

    Attributes:
        scheme_type (type[schemes.BaseScheme]): Type of the SDE integration scheme, needs to be
            subclassed from [`BaseScheme`][pysde.schemes.BaseScheme].
        increment_type (type[increments.BaseRandomIncrement]): Type of the random increment, needs
            to be subclassed from [`BaseRandomIncrement`][pysde.increments.BaseRandomIncrement].
        increment_seed (int): Seed for the random increment object
        storage_type (type[storages.BaseStorage]): Type of the storage object, needs to be
            subclassed from [`BaseStorage`][pysde.storages.BaseStorage].
        storage_stride (int): Stride of the storage object.
        storage_save_directory (Path | None): Directory to save the storage object.
    """

    scheme_type: type[schemes.BaseScheme]
    increment_type: type[increments.BaseRandomIncrement]
    increment_seed: int
    storage_type: type[storages.BaseStorage]
    storage_stride: Annotated[int, Is[lambda x: x > 0]]
    storage_save_directory: Path | None = None


# ==================================================================================================
class SerialRunner:
    """Runner for serial execution (with thread-parallel for loops over trajectories).

    The `SerialRunner` is a simple wrapper plugging together parameter and components for an
    [`SDEIntegrator`][pysde.integrator.SDEIntegrator] object.

    Methods:
        run: Run the SDE integration process, return storage object.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: Settings,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
    ) -> None:
        """Assemble an [`SDEIntegrator`][pysde.integrator.SDEIntegrator] object.

        Args:
            settings (Settings): Parameters and component types for the assembly
            drift_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Drift function of the SDE (c.f. [`schemes`][pysde.schemes])
            diffusion_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Diffusion function of the SDE (c.f. [`schemes`][pysde.schemes])
        """
        storage = settings.storage_type(
            stride=settings.storage_stride,
            save_directory=settings.storage_save_directory,
        )
        increment = settings.increment_type(settings.increment_seed)
        scheme = settings.scheme_type(drift_function, diffusion_function, increment)
        self._integrator = integrator.SDEIntegrator(scheme, storage)

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_state: Real | npt.NDArray,
        initial_time: Real,
        step_size: Real,
        num_steps: Annotated[int, Is[lambda x: x > 0]],
        progress_bar: bool = False,
    ) -> storages.BaseStorage:
        r"""Run the SDE integration process, return storage object.

        Args:
            initial_state (Real | npt.NDArray): Initial state of the system, given for all
                trajectories with shape $d_X \times N$
            initial_time (Real): Initial time $t_0$ of the stochastic process
            step_size (Real): Discrete step size $\delta t$
            num_steps (int): Number of steps to integrate
            progress_bar (bool): Whether to display a progress bar
        """
        result_storage = self._integrator.run(
            initial_state, initial_time, step_size, num_steps, progress_bar
        )
        return result_storage


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
    invoking MPI rank. It then spawns a [SerialRunner][pysde.runner.SerialRunner] object with the
    adjusted parameters. The adjustments are the following:
    1. Partition ensemble of initial states equally across processes.
    2. Adjust the random seed for the random increment object (add the rank number).
    3. Adjust the save directory for the storage object (add the rank number).

    !!! note: Execution from the command line

        Thread-parallel numba loops do not intelligently recognize the number of available threads
        in an MPI environment. Available threads have therefore by mapped to the MPI processes
        explicitly in the invoking `mpirun` command:
        ```bash
            mpiexec -n <NUM_RPOCS> --map-by slot:PE=<NUM_THREADS_PER_PROC> python -m mpi4py parallel_runner.py
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
        settings: Settings,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
    ) -> None:
        """Initialize the parallel runner.

        Args:
            settings (Settings): Parameters and component types for the assembly
            drift_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Drift function of the SDE (c.f. [`schemes`][pysde.schemes])
            diffusion_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray]):
                Diffusion function of the SDE (c.f. [`schemes`][pysde.schemes])

        Raises:
            ImportError: Exception indicating that MPI is required for parallel execution
        """
        if not MPI_LOADED:
            raise ImportError("MPI is required for parallel execution.")
        mpi_communicator = MPI.COMM_WORLD
        self._local_rank = mpi_communicator.Get_rank()
        self._num_processes = mpi_communicator.Get_size()

        settings.storage_save_directory = settings.storage_save_directory.with_name(
            f"{settings.storage_save_directory.stem}_p{self._local_rank}"
            f"{settings.storage_save_directory.suffix}"
        )
        settings.increment_seed += self._local_rank
        self._serial_runner = SerialRunner(settings, drift_function, diffusion_function)

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
        result_storage = self._serial_runner.run(
            local_initial_state, initial_time, step_size, num_steps, progress_bar
        )
        return result_storage

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
