"""_summary_."""
# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from typeguard import typechecked


# ================================ Stochastic Integral Base Class ==================================
class BaseStochasticIntegral(ABC):
    """Abstract base class for computing stochastic integrals.

    Args:
        seed (int): Seed for the random number generator.

    Attributes:
        _rng (numpy.random.Generator): Random number generator.

    """
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, seed: int) -> None:
        """Initializes an instance of the BaseStochasticIntegral class.

        Args:
            seed (int): Seed for the random number generator.

        """
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_single(
        self, noise_dim: int, step_size: float | np.ndarray, num_trajectories: int
    ) -> np.ndarray:
        """Computes a single stochastic integral.

        Args:
            noise_dim (int): Dimension of the noise.
            step_size (float or numpy.ndarray): Step size.
            num_trajectories (int): Number of trajectories.

        Returns:
            numpy.ndarray: Computed stochastic integral.

        """
        pass


# ================================= Simple Default Implementation ==================================
@final
class DefaultStochasticIntegral(BaseStochasticIntegral):
    """A class that computes a single stochastic integral using a random number generator.

    Inherits from the BaseStochasticIntegral abstract base class.

    Methods:
    - compute_single(noise_dim: int, step_size: float, num_trajectories: int) -> np.ndarray: 
        Computes a single stochastic integral using the provided dimensions, step size,
        and number of trajectories.
    """
    # ----------------------------------------------------------------------------------------------
    def compute_single(self, noise_dim: int, step_size: float, num_trajectories: int) -> np.ndarray:
        """Computes a single stochastic integral.

        Args:
            noise_dim (int): The dimension of the noise.
            step_size (float): The step size for the stochastic integral.
            num_trajectories (int): The number of trajectories.

        Returns:
        - np.ndarray: A numpy array representing the computed stochastic integral.
        """
        single_integral = np.sqrt(step_size) * self._rng.normal(size=(noise_dim, num_trajectories))
        return single_integral
