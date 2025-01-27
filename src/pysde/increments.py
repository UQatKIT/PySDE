# ==================================================================================================
from abc import ABC, abstractmethod
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is

# ==================================================================================================


class BaseRandomIncrement(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def sample(
        self,
        _step_size: Real,
        _dimension: Annotated[int, Is[lambda x: x > 0]],
        _num_trajectories: Annotated[int, Is[lambda x: x > 0]],
    ) -> npt.NDArray[np.floating]:
        raise NotImplementedError


# ==================================================================================================
class BrownianIncrement(BaseRandomIncrement):
    def sample(
        self,
        step_size: Real,
        dimension: Annotated[int, Is[lambda x: x > 0]],
        num_trajectories: Annotated[int, Is[lambda x: x > 0]],
    ) -> npt.NDArray[np.floating]:
        return np.sqrt(step_size) * self._rng.standard_normal((dimension, num_trajectories))
