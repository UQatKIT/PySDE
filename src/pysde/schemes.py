# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, final
import numpy as np


# ======================================== Scheme Base Class =======================================
class BaseScheme(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        drift_function: Callable,
        diffusion_function: Callable,
        stochastic_integral: Any,
    ) -> None:
        self._drift = drift_function
        self._diffusion = diffusion_function
        self._stochastic_integral = stochastic_integral

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_step(self, current_state: np.ndarray, current_time: float, step_size: float):
        pass 


# ================================= Explicit Euler-Maruyama Scheme =================================
@final
class ExplicitEulerMaruyamaScheme(BaseScheme):
    # ----------------------------------------------------------------------------------------------
    def compute_step(self, current_state: np.ndarray, current_time: float, step_size: float):
        num_trajectories = current_state.shape[1]
        random_incr = self._stochastic_integral.compute_single(step_size, num_trajectories)
        current_drift = self._drift(current_state, current_time)
        current_diffusion = self._diffusion(current_state, current_time)
        next_state = current_state + current_drift * step_size + current_diffusion * random_incr
        return next_state