"""SDE Integration schemes.

This module facilitates the implementation of SDE integration schemes according to an ABC contract.
Similar to the [`BaseRandomIncrement`][pysde.increments.BaseRandomIncrement] class, the
[`BaseScheme`][pysde.schemes.BaseScheme] class assumes basicially nothing about the properties of
the integration algorithm, ensuring flexibility for a broad range of options. It enforces a simple
interface via the [`step`][pysde.schemes.BaseScheme.step] method. The actual implementation of the
user schemes determines their user-friendliness. The recommended workflow can be seen in the
[`ExplicitEulerMaruyamaScheme`][pysde.schemes.ExplicitEulerMaruyamaScheme] class. Here, the user has
to provide drift and diffusion functions only for a single trajectory. Broadcasting over
trajectories is not necessary. This is taken care of by a jit-compiled, parallel for loop using
`numba`. Other algorithms should follow this pattern, which can be realized by a simple closure
around the step function.

Classes:
    BaseScheme: ABC for SDE integration schemes.
    ExplicitEulerMaruyamaScheme: Implementation of the Explicit Euler-Maruyama scheme.
"""

# ==================================================================================================
from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Real

import numba
import numpy as np
import numpy.typing as npt
from beartype import BeartypeConf, BeartypeStrategy, beartype

from pysde import increments

nobeartype = beartype(conf=BeartypeConf(strategy=BeartypeStrategy.O0))
"""Decorator to deactivate type checking."""


# ==================================================================================================
class BaseScheme(ABC):
    """ABC for SDE integration schemes.

    The base class enforces a uniform interface through the `step` method.

    Methods:
        step: Perform a single step of the integration scheme
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def step(
        self,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        r"""Implement a single step in the integration scheme.

        Args:
            current_state (npt.NDArray[np.floating]): The current state of the system,
                in vectorized form $d_X \times N$
            current_time (Real): The current time $t$ of the stochastic process.
            step_size (Real): Discrete step size $\Delta t$.

        Returns:
            npt.NDArray[np.floating]: The updated state of the system, in vectorized form
                $d_X \times N$.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


# ==================================================================================================
class ExplicitEulerMaruyamaScheme(BaseScheme):
    r"""Implementation of the Explicit Euler-Maruyama scheme.

    The explicit Euler-Maruyama scheme is a simple and efficient method for integrating SDEs. For
    a vector SDE of the form
    $d\mathbf{X}_t = \mathbf{b}(\mathbf{X}_t,t)dt + \mathbf{\Sigma}(X_t,t)d\mathbf{W}_t$,
    the scheme is given by

    $$
        X_{k+1} = X_k + \mathbf{b}(\mathbf{X}_k,t_k)\Delta t + \mathbf{\Sigma}(X_k,t_k)\Delta W_k.
    $$

    Methods:
        step: Perform a single step of the integration scheme
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]],
        random_increment: increments.BaseRandomIncrement,
    ) -> None:
        r"""Initialize scheme with drift, diffusion, and Brownian increment.

        Drift and diffusion need to be implemented as functions of the current state and time. They
        can employ standard Python and numpy functionality, to be jit-compilable with numba. These
        functions have to be provided for single trajectories only, broadcasting is taken care of
        internally.

        !!! warning
            Drift and diffusion have to be defined as proper functions with the `def` keyword,
            not as lambda functions.

        Args:
            drift_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]]):
                Callable representing the drift function $\mathbf{b}(\mathbf{X}_t,t)$.
            diffusion_function (Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]]):
                Callable representing the diffusion function $\mathbf{\Sigma}(\mathbf{X}_t,t)$.
            random_increment (increments.BaseRandomIncrement):
                [BaseRandomIncrement][pysde.increments.BaseRandomIncrement] to approximate the
                Wiener process increment $\Delta W_t$.
        """
        self._drift = numba.njit(drift_function)
        self._diffusion = numba.njit(diffusion_function)
        self._jitted_step = numba.njit(self._step, parallel=True)
        self._random_increment = random_increment

    # ----------------------------------------------------------------------------------------------
    def step(
        self,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        r"""Perform a single step of the integration scheme.

        This method is a closure to the static `_step` function, which implements the actual loop
        over trajectories. in numba.

        Args:
            current_state (npt.NDArray[np.floating]): The current state of the system,
                in vectorized form $d_X \times N$.
            current_time (Real): The current time $t$ of the stochastic process.
            step_size (Real): Discrete step size $\Delta t$.

        Returns:
            npt.NDArray[np.floating]: The updated state of the system, in vectorized
                form $d_X \times N$.
        """
        dimension, num_trajectories = current_state.shape
        vectorized_increment = self._random_increment.sample(
            dimension, num_trajectories, step_size, dtype=current_state.dtype
        )
        vectorized_step = self._jitted_step(
            current_state,
            current_time,
            step_size,
            self._drift,
            self._diffusion,
            vectorized_increment,
        )
        return vectorized_step

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    @nobeartype
    def _step(
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray[np.floating]],
        vectorized_increment: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Numba jittable step function for thread-parallel loop over trajectories.

        !!! warning
            Jitted functions with numba cannot be typ checked with beartype. The `@nobeartype`
            decorator is used to suppress the type checking for this function.
        """
        num_trajectories = current_state.shape[1]
        vectorized_step = np.empty_like(current_state)

        for i in numba.prange(num_trajectories):
            current_drift = drift_function(current_state[:, i], current_time)
            current_diffusion = diffusion_function(current_state[:, i], current_time)
            scalar_step = current_drift * step_size + current_diffusion @ vectorized_increment[:, i]
            vectorized_step[:, i] = scalar_step
        return vectorized_step
