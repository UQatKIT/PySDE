# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from typing import final
import numpy as np


# ================================ Stochastic Integral Base Class ==================================
class BaseStochasticIntegral(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, noise_dim: int, seed: int) -> None:
        self._noise_dim = noise_dim
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_single(self, step_size: float, num_trajectories: int) -> np.ndarray:
        pass


# ================================= Simple Default Implementation ==================================
@final
class DefaultStochasticIntegral(BaseStochasticIntegral):
    # ----------------------------------------------------------------------------------------------
    def compute_single(self, step_size: float, num_trajectories: int) -> np.ndarray:
        single_integral = self._rng.normal(
            loc=0.0, scale=np.sqrt(step_size), size=(self._noise_dim, num_trajectories)
        )
        return single_integral
