# ==================================================================================================
import functools
import multiprocessing
import traceback
from numbers import Real
from pathlib import Path
from typing import Annotated

import numba
import numpy as np
import numpy.typing as npt
from beartype.vale import Is

from pysde import schemes, storages


# ==================================================================================================
class SDEIntegrator:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, scheme: schemes.BaseScheme, storage: storages.BaseStorage) -> None:
        self._scheme = scheme
        self._storage = storage

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        num_processes: Annotated[int, Is[lambda x: x > 0]],
        initial_state: npt.NDArray[np.floating],
        initial_time: Real,
        step_size: Real,
        num_steps: Annotated[int, Is[lambda x: x > 0]],
    ):
        partial_run_function = functools.partial(
            self._run,
            num_processes=num_processes,
            initial_state=initial_state,
            initial_time=initial_time,
            step_size=step_size,
            num_steps=num_steps,
        )
        process_ids = range(num_processes)
        with multiprocessing.Pool(processes=num_processes) as process_pool:
            process_pool.map(partial_run_function, process_ids)

    # ----------------------------------------------------------------------------------------------
    def _run(
        self,
        process_id: Annotated[int, Is[lambda x: x > 0]],
        num_processes: Annotated[int, Is[lambda x: x > 0]],
        initial_state: npt.NDArray[np.floating],
        initial_time: Real,
        step_size: Real,
        num_steps: Annotated[int, Is[lambda x: x > 0]],
    ) -> None:
        self._scheme.setup_parallel(process_id)
        self._storage.setup_parallel(process_id)
        if not (isinstance(initial_state, npt.NDArray) and initial_state.ndim == 2):
            initial_state = np.atleast_2d(initial_state).T
        local_initial_state = self._get_data_portion(initial_state, process_id, num_processes)
        current_state = local_initial_state
        current_time = initial_time

        with Path.open(f"error_p{process_id}.log", "w") as error_log:
            try:
                for i in range(num_steps):
                    self._storage.store(current_time, current_state, i)
                    next_state = self._vectorized_step(
                        self._scheme, current_state, current_time, step_size
                    )
                    current_time += step_size
                    current_state = next_state
            except BaseException:  # noqa: BLE001
                traceback.print_exc(error_log)
            finally:
                self._storage.save()

    # ----------------------------------------------------------------------------------------------
    def _get_data_portion(
        self,
        initial_state: npt.NDArray[np.floating],
        process_id: Annotated[int, Is[lambda x: x > 0]],
        num_processes: Annotated[int, Is[lambda x: x > 0]],
    ) -> npt.NDArray[np.floating]:
        num_trajectories = initial_state.shape[1]
        chunk_size = num_trajectories // num_processes
        start_ind = process_id * chunk_size
        end_ind = num_trajectories if process_id == num_processes - 1 else start_ind + chunk_size
        local_initial_state = initial_state[:, start_ind:end_ind]
        return local_initial_state

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @numba.njit
    def _vectorized_step(
        scheme: schemes.BaseScheme,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        num_local_trajectories = current_state.shape[1]
        new_state = np.empty_like(current_state)
        for i in range(num_local_trajectories):
            new_state[:, i] = scheme.step(current_state[:, i], current_time, step_size)
        return new_state
