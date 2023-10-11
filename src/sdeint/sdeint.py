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


# ===================================== Simple Result Storage ======================================
class SimpleStorage:
    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    def get(self) -> np.ndarray:
        result_array = np.stack(self._result_list, axis=2)
        return result_array


# ================================= Persistent Chunkwise Storage ===================================
class PersistentChunkwiseStorage:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, chunk_size: int, save_directory: str) -> None:
        self._chunk_size = chunk_size
        self._save_directory = save_directory
        self._result_list = []
        self._zarr_storage = None

    # ----------------------------------------------------------------------------------------------
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)
        if len(self._result_list) == self._chunk_size:
            self._save_to_file()
            self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def get(self) -> zarr.Array:
        if self._result_list:
            self._save_to_file()
        return self._zarr_storage

    # ----------------------------------------------------------------------------------------------
    def _save_to_file(self) -> None:
        result_array = np.stack(self._result_list, axis=2)
        if self._zarr_storage is None:
            self._zarr_storage = zarr.array(
                result_array, store=f"{self._save_directory}.zarr", overwrite=True
            )
        else:
            self._zarr_storage.append(result_array, axis=2)


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
        time_array = np.linspace(start_time, start_time + (num_steps - 1) * step_size, num_steps)
        self._storage.append(initial_state)

        current_state = initial_state
        for current_time in time_array[:-1]:
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            self._storage.append(next_state)
            current_state = next_state
        return time_array, self._storage.get()


# =========================================== Utilities ============================================
def reshape_initial_state(initial_state: Number | np.ndarray) -> np.ndarray:
    if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
        initial_state = np.atleast_2d(initial_state).T
    return initial_state