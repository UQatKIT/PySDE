"""_summary_."""
# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

import numpy as np
from typeguard import typechecked

from pysde import stochastic_integrals

sde_model_function = Callable[[np.ndarray, float | np.ndarray], np.ndarray]


# ======================================== Scheme Base Class =======================================
class BaseScheme(ABC):
    """Abstract base class for implementing schemes for solving SDEs.

    Attributes:
        _variable_dim (int): The dimension of the variables in the SDE.
        _noise_dim (int): The dimension of the noise in the SDE.
        _drift (sde_model_function): The drift function of the SDE.
        _diffusion (sde_model_function): The diffusion function of the SDE.
        _stochastic_integral (stochastic_integrals.BaseStochasticIntegral):
            The stochastic integral used in the scheme.
    """
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(
        self,
        variable_dim: int,
        noise_dim: int,
        drift_function: sde_model_function,
        diffusion_function: sde_model_function,
        stochastic_integral: stochastic_integrals.BaseStochasticIntegral,
    ) -> None:
        """Initializes the BaseScheme class.

        Args:
            variable_dim (int): The dimension of the variables in the SDE.
            noise_dim (int): The dimension of the noise in the SDE.
            drift_function (sde_model_function): The drift function of the SDE.
            diffusion_function (sde_model_function): The diffusion function of the SDE.
            stochastic_integral (stochastic_integrals.BaseStochasticIntegral):
                The stochastic integral used in the scheme.
        """
        self._variable_dim = variable_dim
        self._noise_dim = noise_dim
        self._drift = drift_function
        self._diffusion = diffusion_function
        self._stochastic_integral = stochastic_integral

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def compute_step(
        self,
        current_state: np.ndarray,
        current_time: float | np.ndarray,
        step_size: float | np.ndarray,
    ) -> np.ndarray:
        """Abstract method that computes a single step of the scheme.

        Args:
            current_state (np.ndarray): The current state of the SDE.
            current_time (float | np.ndarray): The current time of the SDE.
            step_size (float | np.ndarray): The step size of the scheme.

        Returns:
            np.ndarray: The next state of the SDE.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    def check_consistency(self, initial_state: np.ndarray, static_steps: bool=True) -> None:
        """Checks the consistency of the scheme, initial state, and model functions.

        Args:
            initial_state (np.ndarray): The initial state of the SDE.
            static_steps (bool, optional): Whether the steps are static or dynamic.
                                           Defaults to True.

        Raises:
            ValueError: If the initial state is not two-dimensional or has an incorrect size.
            ValueError: If the output of the drift function has an incorrect shape.
            ValueError: If the output of the diffusion function has an incorrect shape.
        """
        if not initial_state.ndim == 2:
            raise ValueError(
                "Initial condition needs to be two-dimensional, "
                f"but has shape {initial_state.shape}."
            )
        if not initial_state.shape[0] == self._variable_dim:
            raise ValueError(
                f"Initial condition needs to have size {self._variable_dim} in first dimension, "
                f"but has size {initial_state.shape[0]}."
            )

        num_trajectories = initial_state.shape[1]
        if static_steps:
            test_time = 0
        else:
            test_time = np.zeros((1, num_trajectories))
        output_drift = self._drift(initial_state, test_time)
        output_diffusion = self._diffusion(initial_state, test_time)

        if not output_drift.shape == (self._variable_dim, num_trajectories):
            raise ValueError(
                f"Drift function needs to return array of shape {initial_state.shape}, "
                f"but the output has shape {output_drift.shape}. "
                "Is the drift function correctly vectorized?"
            )
        if not output_diffusion.shape == (self._variable_dim, self._noise_dim, num_trajectories):
            raise ValueError(
                "Diffusion function needs to return array of shape "
                f"{(self._variable_dim, self._noise_dim, num_trajectories)}, "
                f"but the output has shape {output_diffusion.shape}. "
                "Is the diffusion function correctly vectorized?"
            )


# ================================= Explicit Euler-Maruyama Scheme =================================
@final
class ExplicitEulerMaruyamaScheme(BaseScheme):
    """A class that implements the explicit Euler-Maruyama scheme.

    Args:
        variable_dim (int): The dimension of the variables in the SDE.
        noise_dim (int): The dimension of the noise in the SDE.
        drift_function (callable): The drift function of the SDE.
        diffusion_function (callable): The diffusion function of the SDE.
        stochastic_integral (StochasticIntegral): The stochastic integral used in the scheme.

    Example Usage:
        # Create an instance of the ExplicitEulerMaruyamaScheme class
        scheme = ExplicitEulerMaruyamaScheme(variable_dim=2,
                                             noise_dim=1,
                                             drift_function=drift,
                                             diffusion_function=diffusion,
                                             stochastic_integral=integral)

        # Compute a single step of the scheme
        current_state = np.array([[1.0, 2.0]])
        current_time = 0.0
        step_size = 0.1
        next_state = scheme.compute_step(current_state, current_time, step_size)

        print(next_state)
        # Output: [[1.1, 2.2]]
    """

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def compute_step(
        self,
        current_state: np.ndarray,
        current_time: float | np.ndarray,
        step_size: float | np.ndarray,
    ) -> np.ndarray:
        """Computes a single step of the explicit Euler-Maruyama scheme for solving SDEs.

        Args:
            current_state (np.ndarray): The current state of the SDE.
            current_time (float | np.ndarray): The current time of the SDE.
            step_size (float | np.ndarray): The step size of the scheme.

        Returns:
            np.ndarray: The next state of the SDE.
        """
        num_trajectories = current_state.shape[1]
        random_incr = self._stochastic_integral.compute_single(
            self._noise_dim, step_size, num_trajectories
        )

        current_drift = self._drift(current_state, current_time)
        current_diffusion = self._diffusion(current_state, current_time)
        next_state = (
            current_state
            + current_drift * step_size
            + np.einsum("ijk,jk->ik", current_diffusion, random_incr)
        )

        assert (
            next_state.shape == current_state.shape
        ), f"Shape mismatch: {next_state.shape} != {current_state.shape}"
        return next_state