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


# ==================================================================================================
class BaseScheme(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def step(
        self,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        raise NotImplementedError


# ==================================================================================================
class ExplicitEulerMaruyamaScheme(BaseScheme):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        random_increment: increments.BaseRandomIncrement,
    ) -> None:
        self._drift = numba.njit(drift_function)
        self._diffusion = numba.njit(diffusion_function)
        self._jitted_step = numba.njit(self._step)
        self._random_increment = random_increment

    # ----------------------------------------------------------------------------------------------
    def step(
        self,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        dimension, num_trajectories = current_state.shape
        vectorized_increment = self._random_increment.sample(step_size, dimension, num_trajectories)
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
        drift_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        diffusion_function: Callable[[npt.NDArray[np.floating], Real], npt.NDArray],
        vectorized_increment: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        num_trajectories = current_state.shape[1]
        vectorized_step = np.empty_like(current_state)

        for i in range(num_trajectories):
            current_drift = drift_function(current_state[:, i], current_time)
            current_diffusion = diffusion_function(current_state[:, i], current_time)
            scalar_step = current_drift * step_size + current_diffusion @ vectorized_increment[:, i]
            vectorized_step[:, i] = scalar_step
        return vectorized_step
