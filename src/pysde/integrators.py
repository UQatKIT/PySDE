# =================================== Imports and Configuration ====================================
import traceback
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Collection
from numbers import Number
from typing import final

import numpy as np
import tqdm
from typeguard import typechecked

from pysde import schemes, storages


# ===================================== Integrator Base Class ======================================
class BaseIntegrator(ABC):
    is_static = None

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, scheme: schemes.BaseScheme, result_storage: storages.BaseStorage) -> None:
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        if not getattr(cls, "is_static"):
            raise AttributeError("Classes derived from BaseIntegrator need to implement the "
                                 "class attribute 'is_static'.")

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def run(
        self, initial_state: Number | np.ndarray, *args, **kwargs
    ) -> tuple[np.ndarray, Collection]:
        self._storage.reset()
        initial_state = self._reshape_initial_state(initial_state)
        self._scheme.check_sde_model(initial_state, self.is_static)

        try:
            self._execute_integration(initial_state, *args, **kwargs)
        except BaseException:
            print(5 * "=")
            print(
                f"Exception has occured during integration with {self.__class__.__name__}. "
                "Integrator will be terminated normally and available results returned."
            )
            print(traceback.format_exc())
            print(5 * "=")
        finally:
            time_array, result_array = self._storage.get()
            return time_array, result_array

    # ----------------------------------------------------------------------------------------------
    def _reshape_initial_state(self, initial_state: Number | np.ndarray) -> np.ndarray:
        if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
            initial_state = np.atleast_2d(initial_state).T
        if initial_state.ndim > 2:
            raise ValueError(
                "Initial state can be at most 2D, " f"but has shape {initial_state.shape}"
            )
        return initial_state

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _execute_integration(self, initial_state: np.ndarray, *args, **kwargs) -> None:
        pass


# ================================= Integrator with Fixed Stepsize =================================
@final
class StaticIntegrator(BaseIntegrator):
    is_static = True

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(
        self,
        scheme: schemes.BaseScheme,
        result_storage: storages.BaseStorage,
        show_progressbar: bool = False,
    ) -> None:
        super().__init__(scheme, result_storage)
        self._disable_progressbar = not show_progressbar

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def _execute_integration(
        self, initial_state: np.ndarray, start_time: float, step_size: float, num_steps: int
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"Step size ({step_size}) must be positive.")
        if num_steps <= 0:
            raise ValueError(f"Number of steps ({num_steps}) must be positive.")

        self._storage.append(start_time, initial_state)
        current_state = initial_state
        time_array = np.linspace(start_time, start_time + (num_steps - 1) * step_size, num_steps)

        for i, current_time in enumerate(
            tqdm.tqdm(time_array[:-1], disable=self._disable_progressbar)
        ):
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            next_time = time_array[i + 1]
            self._storage.append(next_time, next_state)
            current_state = next_state
