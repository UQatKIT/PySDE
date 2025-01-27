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
    scheme_type: type[schemes.BaseScheme]
    increment_type: type[increments.BaseRandomIncrement]
    increment_seed: int
    storage_type: type[storages.BaseStorage]
    storage_stride: Annotated[int, Is[lambda x: x > 0]]
    storage_save_directory: Path | None = None


# ==================================================================================================
class SerialRunner:

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: Settings,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
    ) -> None:
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
    ):
        result_storage = self._integrator.run(
            initial_state, initial_time, step_size, num_steps, progress_bar
        )
        return result_storage


# ==================================================================================================
class ParallelRunner:

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        settings: Settings,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
    ) -> None:
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
    ):
        local_initial_state = self._partition_initial_state(initial_state)
        result_storage = self._serial_runner.run(
            local_initial_state, initial_time, step_size, num_steps, progress_bar
        )
        return result_storage

    # ----------------------------------------------------------------------------------------------
    def _partition_initial_state(self, initial_state: npt.NDArray[np.floating]):
        num_trajectories = initial_state.shape[1]
        partition_size = num_trajectories // self._num_processes
        start_ind = self._local_rank * partition_size
        if self._local_rank == self._num_processes - 1:
            end_ind = num_trajectories
        else:
            end_ind = start_ind + partition_size
        local_initial_state = initial_state[:, start_ind:end_ind]
        return local_initial_state

