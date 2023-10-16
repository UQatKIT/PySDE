# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from typeguard import typechecked


# ================================ Stochastic Integral Base Class ==================================
class BaseStochasticIntegral(ABC):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_single(
        self, noise_dim: int, step_size: float | np.ndarray, num_trajectories: int
    ) -> np.ndarray:
        pass

    # ----------------------------------------------------------------------------------------------
    def _check_input(
        self, noise_dim: int, step_size: float | np.ndarray, num_trajectories: int
    ) -> None:
        if noise_dim <= 0:
            raise ValueError(f"Noise dimension ({noise_dim}) must be positive.")
        if np.any(np.array(step_size)) <= 0:
            raise ValueError("Step size must be positive.")
        if num_trajectories <= 0:
            raise ValueError(f"Number of trajectories ({num_trajectories}) must be positive.")


# ================================= Simple Default Implementation ==================================
@final
class DefaultStochasticIntegral(BaseStochasticIntegral):
    # ----------------------------------------------------------------------------------------------
    def compute_single(self, noise_dim: int, step_size: float, num_trajectories: int) -> np.ndarray:
        self._check_input(noise_dim, step_size, num_trajectories)
        single_integral = np.sqrt(step_size) * self._rng.normal(size=(noise_dim, num_trajectories))
        return single_integral
