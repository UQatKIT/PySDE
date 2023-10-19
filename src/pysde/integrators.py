"""Integrators for SDEs.

This module implements the integrators that serve as overall wrappers for the SDE integration.
They take in a solver scheme and storage object, and execute the actual integration loops.
All integrator implementations have to inherit from the `BaseIntegrator` class.

Classes:
    BaseIntegrator: Abstract base class for all integrators.
    StaticIntegrator: Implementation with a fixed step size.

Functions:
    reshape_initial_state: Ensures consistency of the initial state provided by user. An initial
                           state has to be convertible into a 2D array, whereas the first
                           dimension accounts for the physical dimension of the solution, and the
                           second dimension for the number of trajectories to be simulated.
    decorate_run_method: Decorator that adds functionality to the `run` method of all subclasses
                         of `BaseIntegrator`. This allows derived classes to only implement the
                         integrator-specific logic in their `run` method. Initial state setup,
                         consistency checks and error handling are handled by the decorator.
"""

# =================================== Imports and Configuration ====================================
import functools
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import final

import numpy as np
import tqdm
from typeguard import typechecked

from pysde import schemes, storages

RunFunction = Callable[..., tuple[Iterable, Iterable]]


# ============================================ Utilities ===========================================
@typechecked
def reshape_initial_state(initial_state: float | np.ndarray) -> np.ndarray:
    """Reshapes the initial state array if necessary.

    Users can provide initial states as single number, 1D arrays or 2D arrays. The converted initial
    state is always a 2D array, where the first dimension accounts for the physical dimension of the
    solution, and the second dimension for the number of trajectories to be simulated. If a number
    is provided as input, it is interpreted as 1x1 array, denoting a 1D problem and a single
    trajectory. If a one-dimensional array of length N is provided, it is interpreted as a
    problem of physical dimension N with a single trajectory.

    Parameters:
        initial_state (float | np.ndarray): The initial state array to be reshaped.

    Returns:
        np.ndarray: The reshaped initial state array, always 2D.
    """
    if not (isinstance(initial_state, np.ndarray) and initial_state.ndim == 2):
        initial_state = np.atleast_2d(initial_state).T
    if initial_state.ndim > 2:
        raise ValueError(f"Initial state can be at most 2D, but has shape {initial_state.shape}")
    return initial_state


# --------------------------------------------------------------------------------------------------
def decorate_run_method(run_function: RunFunction) -> RunFunction:
    """Decorates the `run` method of a subclass of the `BaseIntegrator`.

    The general run logic of an SDE integrator comprises a series of commands that is common to all
    concrete implementations. These functionalities are:
    - Correct setup of the initial condition via `reshape_initial_state`
    - Check for consistency of the initial condition with the provided scheme
    - Reset of the result storage object
    - Guarding of the actual integration by exception handling, such that intermediate result are
      returned in the event of an error
    This decorator performs these operations on the `run` method of subclasses of `BaseIntegrator`.
    Therefore these subclasses must only implement the bare integration logic, while still exposing
    a complete interface to the user.
    
    Args:
        run_function (RunFunction): The original `run` method of a subclass of `BaseIntegrator`.
    
    Returns:
        RunFunction: The decorated `run` method, which performs additional pre-processing and 
                     exception handling.

    Raises:
        TypeError: If the decorated function is not called from a subclass of `BaseIntegrator`.
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
            self._storage.save()
            time_array, result_array = self._storage.get()
        finally:
            return time_array, result_array

    return wrapper


# ===================================== Integrator Base Class ======================================
class BaseIntegrator(ABC):
    """Abstract base class for implementing different types of integrators.

    This ABC implements the common interface for all integrator classes. It also initializes the
    integrators with a scheme and a result storage object.

    Attributes:
        _is_static: A class attribute that indicates whether the integrator has a fixed step size.
                    Subclasses are forced to implement this attribute, as it is required for
                    consistency checks with the integration scheme.

    Methods:
        __init__(): Base class constructor.
        run(): Inteface for integration runs.
    """

    _is_static = None

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, scheme: schemes.BaseScheme, result_storage: storages.BaseStorage) -> None:
        """Initializes a new instance of the class.

        The method employs run-time type checking.

        Args:
            scheme (schemes.BaseScheme): The scheme to use for initialization.
            result_storage (storages.BaseStorage): The storage to use for storing the results.
        """
        self._scheme = scheme
        self._storage = result_storage

    # ----------------------------------------------------------------------------------------------
    def __init_subclass__(cls) -> None:
        """Subclass initialization hook.

        This method is called automatically when a subclass of BaseIntegrator is defined.
        It enforces that all subclasses implement the '_is_static' class attribute.

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
    ) -> tuple[Iterable, Iterable]:
        """Runs the integration process with the specified initial state.

        This is an abstract method that defines an interface with the mose generic set of arguments.

        Args:
            initial_state (float | np.ndarray): The initial state for the integration process.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple[Iterable, Iterable]:
                A tuple containing the time array and the result array. For a static integrator, the
                time array will be one-dimensional and of size N, where N is the number of step
                sizes. For a dynamic integrator, the time array will be of two-dimensional, where
                the first dimension holds the custom time point for each trajectory simulated, and
                the second dimension is over the different time steps. The result array will always
                be three-dimensional. The first dimension corresponds to that of the physical
                problem, the second to the number of trajectories, and the third to the number of
                time steps.
        """
        pass


# ================================= Integrator with Fixed Stepsize =================================
@final
class StaticIntegrator(BaseIntegrator):
    """Integrator implementation with constant step size.

    This class implements a basic integrator with constant (scalar) step size.

    Attributes:
        _is_static (bool): A class attribute indicating that the integrator has a fixed step size.
                           In this case, it is set to True.
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
        """Initializes a new instance of the class.

        This method employs run-time type checking.

        Args:
            scheme (schemes.BaseScheme): The chosen integration scheme. Needs to be a subclass of
                                         `BaseScheme`.
            result_storage (storages.BaseStorage): The chosen storage object. Needs to be a subclass
                                                   of `BaseStorage`. 
            show_progressbar (bool, optional): Whether to show a progress bar for the integration.
                                               Defaults to False.

        Attributes:
            _disable_progressbar (bool): A class attribute that indicates whether the progress bar
        """
        super().__init__(scheme, result_storage)
        self._disable_progressbar = not show_progressbar

    # ----------------------------------------------------------------------------------------------
    @decorate_run_method
    @typechecked
    def run(
        self, initial_state: float | np.ndarray, start_time: float, step_size: float, num_steps: int
    ) -> tuple[Iterable, Iterable]:
        """Runs the simulation.

        With a constant step size, the integrator  performs a deterministic loop for the given
        number of steps and with the given step size. No error checking is performed.

        The method is decorated with the `decorate_run_method` decorator to enforce standard setup
        and error handling for the integration loop. Hav a look at the base class documentation for
        a more detailed description of the return type.

        This method uses run-time type checking.

        Args:
            initial_state (float | np.ndarray): The initial state of the simulation.
            start_time (float): The start time of the simulation.
            step_size (float): The size of each step in the simulation.
            num_steps (int): The number of steps to run the simulation for.

        Returns:
            tuple[Iterable, Iterable]: A
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

        self._storage.save()
        time_array, result_array = self._storage.get()
        return time_array, result_array