# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from collections.abc import Sequence
from numbers import Number
from typing import Any, final

import numpy as np
import tqdm


# ===================================== Integrator Base Class ======================================
class BaseIntegrator(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, scheme: Any, result_storage: Any) -> None:
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def run(
        self, start_time: float, step_size: float, num_steps: int, initial_state: np.ndarray
    ) -> tuple[np.ndarray, Sequence]:
        pass


# ================================= Integrator with Fixed Stepsize =================================
@final
class StaticIntegrator(BaseIntegrator):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, scheme: Any, result_storage: Any, show_progressbar: bool = False) -> None:
        super().__init__(scheme, result_storage)
        self._disable_progressbar = not show_progressbar

    # ----------------------------------------------------------------------------------------------
    def run(
        self, start_time: float, step_size: float, num_steps: int, initial_state: np.ndarray
    ) -> tuple[np.ndarray, Any]:
        initial_state = reshape_initial_state(initial_state)
        time_array = np.linspace(start_time, start_time + (num_steps - 1) * step_size, num_steps)
        self._storage.append(initial_state)

        current_state = initial_state
        for current_time in tqdm.tqdm(time_array[:-1], disable=self._disable_progressbar):
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            self._storage.append(next_state)
            current_state = next_state
        return time_array, self._storage.get()


# =========================================== Utilities ============================================
def reshape_initial_state(initial_state: Number | np.ndarray) -> np.ndarray:
    if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
        initial_state = np.atleast_2d(initial_state).T
    return initial_state
    