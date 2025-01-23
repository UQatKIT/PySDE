# ==================================================================================================
from abc import ABC, abstractmethod
from collections.abc import Callable
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is

from pysde import increments


# ==================================================================================================
class BaseScheme(ABC):
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def setup_parallel(self, process_id: Annotated[int, Is[lambda x: x > 0]]) -> None:
        raise NotImplementedError

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
        self._drift = drift_function
        self._diffusion = diffusion_function
        self._random_increment = random_increment

    # ----------------------------------------------------------------------------------------------
    def setup_parallel(self, process_id: Annotated[int, Is[lambda x: x > 0]]) -> None:
        self._random_increment.setup_parallel(process_id)

    # ----------------------------------------------------------------------------------------------
    def step(
        self,
        current_state: npt.NDArray[np.floating],
        current_time: Real,
        step_size: Real,
    ) -> npt.NDArray[np.floating]:
        random_increment = self._random_increment.sample(step_size)
        current_drift = self._drift(current_state, current_time)
        current_diffusion = self._diffusion(current_state, current_time)
        next_state = current_drift * step_size + current_diffusion @ random_increment
        return next_state
