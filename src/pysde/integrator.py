"""SDE Integrator.

The SDE integrator stands at the top of the composition hierarchy in PySDE. It is composed of a
scheme and a storage object, and drives the SDE integration process. To stick to the overall design
goal of simplicity, PySDE only supports integration with constant step size. Due to its modularity,
however, adaptive schemes can easily implemented with the lower-level components.

Classes:
    SDEIntegrator: Static step size SDE integrator.
"""

# ==================================================================================================
import traceback
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
from beartype.vale import Is
from tqdm import tqdm

from pysde import schemes, storages


# ==================================================================================================
class SDEIntegrator:
    """Static step size SDE integrator.

    Methods:
        run: Run the integration process.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, scheme: schemes.BaseScheme, storage: storages.BaseStorage) -> None:
        """Initialize with a scheme and a storage object.

        Args:
            scheme (schemes.BaseScheme): SDE Integration scheme
            storage (storages.BaseStorage): Data storage object
        """
        self._scheme = scheme
        self._storage = storage

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_state: Real | npt.NDArray,
        initial_time: Real,
        step_size: Real,
        num_steps: Annotated[int, Is[lambda x: x > 0]],
        progress_bar: bool = False,
    ) -> storages.BaseStorage:
        r"""Run the integration process.

        Args:
            initial_state (Real | npt.NDArray): Initial state of the system, given for all
                trajectories with shape $d_X \times N$
            initial_time (Real): Initial time $t_0$ of the stochastic process
            step_size (Real): Discrete step size $\Delta t$
            num_steps (int): Number of steps to integrate
            progress_bar (bool): Whether to display a progress bar

        Returns:
            storages.BaseStorage: Storage object containing the SDE trajectory data
        """
        if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
            initial_state = np.atleast_2d(initial_state).T
            initial_state = np.astype(initial_state, np.float64)

        current_state = initial_state
        current_time = initial_time
        disable_progress_bar = not progress_bar
        try:
            for i in tqdm(range(num_steps), disable=disable_progress_bar):
                self._storage.store(current_time, current_state, i)
                vectorized_step = self._scheme.step(current_state, current_time, step_size)
                current_state = current_state + vectorized_step
                current_time += step_size
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
        finally:
            self._storage.save()
            return self._storage  # noqa: B012
