# =================================== Imports and Configuration ====================================
import traceback
from abc import ABC, abstractmethod
from collections.abc import Collection
from numbers import Number
from typing import Any, final

import numpy as np
import tqdm
from typeguard import typechecked

from .schemes import BaseScheme


# ===================================== Integrator Base Class ======================================
class BaseIntegrator(ABC):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, scheme: BaseScheme, result_storage: Any) -> None:
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def run(self,
            initial_state: Number | np.ndarray,
            *args,
            **kwargs) -> tuple[np.ndarray, Collection]:
        
        self._time_array = None
        self._storage.reset()
        initial_state = self._reshape_initial_state(initial_state)
        self._storage.append(initial_state)

        try:
            self._execute_integration(initial_state, *args, **kwargs)
        except BaseException:
            print(5*"=")
            print(f"Exception has occured during integration with {self.__class__.__name__}. "
                  "Integrator will be terminated normally and available results returned.")
            print(traceback.format_exc())
            print(5*"=")
        finally:
            time_array = np.array(self._time_array)
            result_array = self._storage.get()
            return time_array, result_array

    # ----------------------------------------------------------------------------------------------
    def _reshape_initial_state(self, initial_state: Number | np.ndarray) -> np.ndarray:
        if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
            initial_state = np.atleast_2d(initial_state).T
        return initial_state
    
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def _execute_integration(self, initial_state: np.ndarray, *args, **kwargs) -> None:
        pass


# ================================= Integrator with Fixed Stepsize =================================
@final
class StaticIntegrator(BaseIntegrator):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, scheme: Any, result_storage: Any, show_progressbar: bool = False) -> None:
        super().__init__(scheme, result_storage)
        self._disable_progressbar = not show_progressbar

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def _execute_integration(self,
                             initial_state: np.ndarray,
                             start_time: float,
                             step_size: float,
                             num_steps: int) -> None:

        self._time_array = np.linspace(start_time,
                                       start_time + (num_steps - 1) * step_size,
                                       num_steps)
        current_state = initial_state
        for current_time in tqdm.tqdm(self._time_array[:-1], disable=self._disable_progressbar):
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            self._storage.append(next_state)
            current_state = next_state
    