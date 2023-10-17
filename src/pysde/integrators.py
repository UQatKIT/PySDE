"""Module: integrators.py.

This module provides integrator classes for SDEs (Stochastic Differential Equations).

- `BaseIntegrator`: The base class for all integrators.
- `StaticIntegrator`: An integrator with a fixed step size.

These integrators can be used to solve SDEs numerically.

Example usage:
    >>> integrator = StaticIntegrator()
    >>> result = integrator.run(initial_state, start_time, step_size, num_steps)
"""


# =================================== Imports and Configuration ====================================
import functools
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection
from typing import final

import numpy as np
import tqdm
from typeguard import typechecked

from pysde import schemes, storages

RunFunction = Callable[..., tuple[Collection, Collection]]


# ============================================ Utilities ===========================================
def reshape_initial_state(initial_state: float | np.ndarray) -> np.ndarray:
    """Reshape the initial state array if necessary.

    Parameters:
        initial_state (float | np.ndarray): The initial state array to be reshaped.

    Returns:
        np.ndarray: The reshaped initial state array.
    """
    if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
        initial_state = np.atleast_2d(initial_state).T
    if initial_state.ndim > 2:
        raise TypeError("Initial state can be at most 2D, " f"but has shape {initial_state.shape}")
    return initial_state


# --------------------------------------------------------------------------------------------------
def decorate_run_method(run_function: RunFunction) -> RunFunction:
    """Decorator that adds functionality to the `run` method of a subclass of the `BaseIntegrator`.
    
    Args:
        run_function (RunFunction): The original `run` method of a subclass of `BaseIntegrator`.
    
    Returns:
        RunFunction: The decorated `run` method, which performs additional pre-processing and 
                     exception handling.
    """
    @functools.wraps(run_function)
    def wrapper(self, initial_state: float | np.ndarray, *args, **kwargs):
        if not isinstance(self, BaseIntegrator):
            raise TypeError("Decorated function must be called from a subclass of BaseIntegrator.")

        initial_state = reshape_initial_state(initial_state)
        self._scheme.check_consistency(initial_state, self._is_static)
        self._storage.reset()

        try:
            time_array, result_array = run_function(self, initial_state, *args, **kwargs)
        except BaseException:
            print(f"Exception has occured during integration with {self.__class__.__name__}.")
            print("Integrator will be terminated normally and available results returned.")
            print(traceback.format_exc())
            time_array, result_array = self._storage.get()
        finally:
            return time_array, result_array

    return wrapper


# ===================================== Integrator Base Class ======================================
class BaseIntegrator(ABC):
    """Abstract base class for implementing different types of integrators.

    Attributes:
        _is_static: A class attribute that indicates whether the integrator has a fixed step size.
    """

    _is_static = None

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, scheme: schemes.BaseScheme, result_storage: storages.BaseStorage) -> None:
        """Initialize a new instance of the class.

        Args:
            scheme (schemes.BaseScheme): The scheme to use for initialization.
            result_storage (storages.BaseStorage): The storage to use for storing the results.
        """
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        """Initialize a subclass of BaseIntegrator.

        This method is called automatically when a subclass of BaseIntegrator is defined.

        Parameters:
            cls (type): The subclass being initialized.

        Raises:
            AttributeError: If the subclass does not implement the '_is_static' class attribute.
        """
        if not hasattr(cls, "_is_static"):
            raise AttributeError(
                "Classes derived from BaseIntegrator need to implement the "
                "class attribute 'is_static'."
            )

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def run(
        self, initial_state: float | np.ndarray, *args, **kwargs
    ) -> tuple[Collection, Collection]:
        """Run the integration process with the specified initial state.

        Args:
            initial_state (float | np.ndarray): The initial state for the integration process.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[np.ndarray, Collection]: A tuple containing the time array and the result array.
        """
        pass


# ================================= Integrator with Fixed Stepsize =================================
@final
class StaticIntegrator(BaseIntegrator):
    """A class representing a static integrator.

    This class inherits from the `BaseIntegrator` class and implements the integration process
    for a static integrator.

    Attributes:
        _is_static (bool): A class attribute indicating that the integrator has a fixed step size.
                           In this case, it is set to True.

    Methods:
        __init__(scheme, result_storage, show_progressbar=False):
            Initializes a new instance of the `StaticIntegrator` class.
        run(initial_state, start_time, step_size, num_steps):
            Runs the simulation for a given initial state, start time, step size, and step number.
    """

    _is_static = True

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(
        self,
        scheme: schemes.BaseScheme,
        result_storage: storages.BaseStorage,
        show_progressbar: bool = False,
    ) -> None:
        """Initialize a new instance of the class.

        Args:
            scheme (schemes.BaseScheme): The scheme object.
            result_storage (storages.BaseStorage): The result storage object.
            show_progressbar (bool, optional): Whether to show the progress bar. Defaults to False.
        """
        super().__init__(scheme, result_storage)
        self._disable_progressbar = not show_progressbar

    # ----------------------------------------------------------------------------------------------
    @decorate_run_method
    #@typechecked
    def run(
        self, initial_state: float | np.ndarray, start_time: float, step_size: float, num_steps: int
    ) -> tuple[Collection, Collection]:
        """Run the simulation for a given initial state, start time, step size, and number of steps.

        Args:
            initial_state (float | np.ndarray): The initial state of the simulation.
            start_time (float): The start time of the simulation.
            step_size (float): The size of each step in the simulation.
            num_steps (int): The number of steps to run the simulation for.

        Returns:
            tuple[Collection, Collection]: A tuple containing the collections of data
                                           generated for the simulation.
        """
        if step_size <= 0:
            raise ValueError(f"Step size ({step_size}) must be positive.")
        if num_steps <= 0:
            raise ValueError(f"Number of steps ({num_steps}) must be positive.")

        self._storage.append(start_time, initial_state)
        current_state = initial_state
        run_times = np.linspace(start_time, start_time + (num_steps - 1) * step_size, num_steps)

        for i, current_time in enumerate(
            tqdm.tqdm(run_times[:-1], disable=self._disable_progressbar)
        ):
            next_state = self._scheme.compute_step(current_state, current_time, step_size)
            next_time = run_times[i + 1]
            self._storage.append(next_time, next_state)
            current_state = next_state

        time_array, result_array = self._storage.get()
        return time_array, result_array