"""Stochastic Integrals for the Wiener Process Portion of an SDE.

This module implements the functionality for integration over Wiener processes, as necessary for the
random portion in SDEs. The integral objects are initialized with a seed for their intrinsic PRNG.

Classes:
    `BaseStochasticIntegral`: Abstract base class for a common interface
    `ItoStochasticIntegral`: Implementation of Ito stochastic integrals
"""

# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from typing import final

import numpy as np
from typeguard import typechecked


# ================================ Stochastic Integral Base Class ==================================
class BaseStochasticIntegral(ABC):
    """Abstract base class for computing stochastic integrals.

    This classes provided the interface and base functionality for all stochastic integral classes.
    It requires that they implement a `compute_single` method. This method computes single
    stochastic integrals, which is the minimal requirement of any integration scheme. Of course,
    higher order integrals can be implemented by the derived classes as well.

    Attributes:
        _rng (numpy.random.Generator): Random number generator.

    Methods:
        __init__(): Base class constructor.
        compute_single(): Interface for computing a single stochastic integral.
    """
    
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, seed: int) -> None:
        """Initializes an instance of the BaseStochasticIntegral class.

        Basically initialized a numpy PRNG with the provided seed.

        This method employs run-time type checking.

        Args:
            seed (int): Seed for the random number generator.
        """
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_single(
        self, noise_dim: int, step_size: float | np.ndarray, num_trajectories: int
    ) -> np.ndarray:
        """Interface for computing a single stochastic integral.

        The RNG process needs to be vectorized over the different trajectories.

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
class ItoStochasticIntegral(BaseStochasticIntegral):
    """Implementation for Ito stochastic integrals."""

    # ----------------------------------------------------------------------------------------------
    def compute_single(self, noise_dim: int, step_size: float, num_trajectories: int) -> np.ndarray:
        """Computes a single stochastic integral of appropriate dimension for the given step size.

        The RNG process is vectorized over the different trajectories.

        Args:
            noise_dim (int): The dimension of the noise.
            step_size (float): The step size for the stochastic integral.
            num_trajectories (int): The number of trajectories.

        Returns:
            np.ndarray: A numpy array representing the computed stochastic integral.
        """
        single_integral = np.sqrt(step_size) * self._rng.normal(size=(noise_dim, num_trajectories))
        return single_integral
