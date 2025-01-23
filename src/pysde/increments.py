# ==================================================================================================
from abc import ABC, abstractmethod
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is


# ==================================================================================================
class BaseRandomIncrement(ABC):
    def __init__(self, dimension: Annotated[int, Is[lambda x: x > 0]]) -> None:
        self._dimension = dimension

    # ----------------------------------------------------------------------------------------------
    def setup_parallel(self, process_id: Annotated[int, Is[lambda x: x > 0]]) -> None:
        self._rng = np.random.default_rng(process_id)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def sample(self, step_size: Real) -> npt.NDArray[np.floating]:
        pass


# ==================================================================================================
class BrownianIncrement(BaseRandomIncrement):
    def sample(self, step_size: Real) -> npt.NDArray[np.floating]:
        return np.sqrt(step_size) * self._rng.standard_normal(self._dimension)
