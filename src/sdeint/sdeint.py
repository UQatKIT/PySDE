# =================================== Imports and Configuration ====================================
from numbers import Number
from collections.abc import Callable
from typing import Any

import numpy as np
import zarr


# ================================= Stochastic Integral Generator ==================================
class StochasticIntegralGenerator:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, noise_dim: int, seed: int) -> None:
        self._noise_dim = noise_dim
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    def compute_single(self, step_size: float, num_trajectories: int) -> np.ndarray:
        single_integral = self._rng.normal(
            loc=0.0, scale=np.sqrt(step_size), size=(self._noise_dim, num_trajectories)
        )
        return single_integral

    # ----------------------------------------------------------------------------------------------
    def compute_double(self, step_size: float, num_trajectories: int) -> np.ndarray:
        raise NotImplementedError("Double integral is not implemented for this class")


# ===================================== Simple Result Storage ======================================
class SimpleStorage:
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._result_array = None

    # ----------------------------------------------------------------------------------------------
    def set_up_storage(self, num_components: int, num_trajectories: int, num_steps: int) -> None:
        self._result_array = np.zeros((num_components, num_trajectories, num_steps))

    # ----------------------------------------------------------------------------------------------
    def store_result(self, result: np.ndarray, time_index: int) -> None:
        self._result_array[:, :, time_index] = result

    # ----------------------------------------------------------------------------------------------
    def get_result(self, time_index: int) -> np.ndarray:
        result = self._result_array[:, :, time_index]
        return result

    # ----------------------------------------------------------------------------------------------
    def return_result_data(self):
        return self._result_array


# ================================= Persistent Chunkwise Storage ===================================
class PersistentChunkwiseStorage:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, chunk_size: int, save_directory: str) -> None:
        self._chunk_size = chunk_size
        self._save_directory = save_directory
        self._num_components = None
        self._num_trajectories = None
        self._num_steps = None
        self._result_array = None
        self._zarr_storage = None
        self._storage_chunk_sizes = None

    # ----------------------------------------------------------------------------------------------
    def set_up_storage(self, num_components: int, num_trajectories: int, num_steps: int) -> None:
        self._num_components = num_components
        self._num_trajectories = num_trajectories
        self._num_steps = num_steps
        self._zarr_storage = zarr.open(
            f"{self._save_directory}.zarr",
            mode="w",
            shape=(num_components, num_trajectories, num_steps),
        )
        self._storage_chunk_sizes = self._define_chunks(num_steps)

    # ----------------------------------------------------------------------------------------------
    def store_result(self, result: np.ndarray, time_index: int) -> None:
        chunk_num = int(np.floor(time_index / self._chunk_size))
        time_index_chunk = time_index % self._chunk_size
        if time_index_chunk == 0:
            self._result_array = np.zeros(
                (self._num_components, self._num_trajectories, self._storage_chunk_sizes[chunk_num])
            )
        self._result_array[:, :, time_index_chunk] = result
        if time_index_chunk == (self._storage_chunk_sizes[chunk_num] - 1):
            self._save_result_to_file(self._result_array, chunk_num)

    # ----------------------------------------------------------------------------------------------
    def get_result(self, time_index: int) -> np.ndarray:
        time_index = time_index % self._chunk_size
        result = self._result_array[:, :, time_index]
        return result

    # ----------------------------------------------------------------------------------------------
    def return_result_data(self):
        return self._zarr_storage

    # ----------------------------------------------------------------------------------------------
    def _define_chunks(self, num_steps: int) -> tuple[int]:
        num_chunks = int(np.floor(num_steps / self._chunk_size))
        if not (trailing_size := num_steps % self._chunk_size) == 0:
            num_chunks += 1
            chunk_sizes = (self._chunk_size,) * (num_chunks - 1) + (trailing_size,)
        else:
            chunk_sizes = (self._chunk_size,) * num_chunks
        return chunk_sizes

    # ----------------------------------------------------------------------------------------------
    def _save_result_to_file(self, result_array: np.ndarray, chunk_num: int) -> None:
        start_index = chunk_num * self._chunk_size
        end_index = np.min((start_index + self._chunk_size, self._num_steps))
        self._zarr_storage[..., start_index:end_index] = result_array


# ===================================== Euler-Maruyama Scheme ======================================
class EulerMaruyamaScheme:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        drift_function: Callable,
        diffusion_function: Callable,
        stochastic_integral_generator: Any,
    ) -> None:
        self._drift = drift_function
        self._diffusion = diffusion_function
        self._si_generator = stochastic_integral_generator

    # ----------------------------------------------------------------------------------------------
    def compute_step(self, current_state: np.ndarray, current_time: float, step_size: float):
        num_trajectories = current_state.shape[1]
        random_incr = self._si_generator.compute_single(step_size, num_trajectories)
        current_drift = self._drift(current_state, current_time)
        current_diffusion = self._diffusion(current_state, current_time)
        next_state = current_state + current_drift * step_size + current_diffusion * random_incr
        return next_state


# ================================= Integrator with Fixed Stepsize =================================
class StaticIntegrator:
    
    # ----------------------------------------------------------------------------------------------
    def __init__(self, scheme: Any, result_storage: Any) -> None:
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    def run(
        self, start_time: float, step_size: float, num_steps: int, initial_state: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        initial_state = reshape_initial_state(initial_state)
        num_components, num_trajectories = initial_state.shape
        time_array = np.linspace(start_time, start_time + (num_steps - 1) * step_size, num_steps)
        self._storage.set_up_storage(num_components, num_trajectories, num_steps)
        self._storage.store_result(initial_state, 0)

        for i, current_time in enumerate(time_array[:-1]):
            current_state = self._storage.get_result(i)
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            self._storage.store_result(next_state, i + 1)
        return time_array, self._storage.return_result_data()


# =========================================== Utilities ============================================
def reshape_initial_state(initial_state: Number | np.ndarray) -> np.ndarray:
    if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
        initial_state = np.atleast_2d(initial_state).T
    return initial_state
