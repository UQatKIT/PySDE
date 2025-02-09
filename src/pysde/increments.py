r"""Stochastic integrals for diffusion tems in SDEs.

For a typical SDE in differential form,
$d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t,t)dt + \mathbf{\Sigma}(X_t,t)d\mathbf{W}_t$, objects
derived from [`BaseRandomIncrement`][pysde.increments.BaseRandomIncrement] generate the Wiener
process increments $d\mathbf{W}_t$. In the base class, we assume basically nothing about what this
class actually returns in view of the numerical integrator. This means that an implementation could
return a simple random increment vector for first order schemes, as well as more complex
combinations of Lévy areas. The ABC simply enforces an interface for the
[`sample`][pysde.increments.BaseRandomIncrement.sample] method.

classes:
    BaseRandomIncrement: ABC of stochastic integrals for diffusion tems in SDEs.
    BrownianIncrement: Simple implementation of a Wiener process increment for first order schemes.
"""

# ==================================================================================================
from abc import ABC, abstractmethod
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is


# ==================================================================================================
class BaseRandomIncrement(ABC):
    r"""ABC of stochastic integrals for diffusion tems in SDEs.

    Implements a minimal interface for stochastic integrals in numerical integration schemes.

    Methods:
        sample: Generate a random increment.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, seed: int = 0) -> None:
        """Initialize a random increment with at least a seed.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to 0.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def sample(
        self,
        _dimension: Annotated[int, Is[lambda x: x > 0]],
        _num_trajectories: Annotated[int, Is[lambda x: x > 0]],
        _step_size: Real,
        _dtype: np.float32 | np.float64,
    ) -> object:
        r"""Generate a random increment.

        Stochastic integrals take at least a step size $\Delta t$, a physical dimension $d_W$, and
        the number of trajectories $N$. Generation of the random increment has to be vectorized over
        $N$.

        !!! note "Make sure to use the right dtype"
            For the SDE sample to be floating point agnostic, the `sample` method in all derived
            classes needs to return arrays of the correct dtype, as provided through the function
            argument.

        Args:
            _dimension (int): Physical dimension $d_W$ of the increment
            _num_trajectories (int): Number of trajectories $N$
            _step_size (Real): Discrete step size $\nabla t$ of the integrator
            _dtype (np.float32 | np.float64): Data type of the random increment, can be 32 or 64
                bits
        Raises:
            NotImplementedError: Exception indicating that the method needs to be implemented in
                derived classes.

        Returns:
            object: Random increment
        """
        raise NotImplementedError


# ==================================================================================================
class BrownianIncrement(BaseRandomIncrement):
    r"""Simple implementation of a Wiener process increment for first order schemes.

    For a step size $\Delta t$, we approximate $dW_t$ simply as $\sqrt{\Delta t}\xi$, where $\xi$ is
    a vector of i.i.d. unit normal samples.
    The sample is implemented to also work with $\Delta t < 0$ for integration
    backwards in time.
    """

    def sample(
        self,
        dimension: Annotated[int, Is[lambda x: x > 0]],
        num_trajectories: Annotated[int, Is[lambda x: x > 0]],
        step_size: Real,
        dtype: object,
    ) -> npt.NDArray[np.floating]:
        r"""Generate a random increment $dW_t$.

        Args:
            dimension (int): Physical dimension $d_W$ of the increment
            num_trajectories (int): Number of trajectories $N$
            step_size (Real): Discrete step size $\nabla t$ of the integrator
            dtype (np.float32 | np.float64): Data type of the random increment, can be 32 or 64 bits
        Returns:
            npt.NDArray[np.floating]: Sample of dimension $d_W\times N$
        """
        increment = np.sqrt(np.abs(step_size), dtype=dtype) * self._rng.standard_normal(
            size=(dimension, num_trajectories), dtype=dtype
        )
        return increment
