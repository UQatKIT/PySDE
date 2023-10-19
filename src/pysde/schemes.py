"""Integration Schemes for SDEs.

This module implements different integration schemes that can be employed within an integrator.
Each scheme takes in the information of an SDE model and stochastic integegral generator, and
provides a method for computing a single step in the numerical integration. All schemes have to be 
derived from the `BaseScheme` class.

Classes:
    BaseScheme: Abstract base class for all integration schemes.
    ExplicitEulerMaruyamaScheme: Explicit Euler-Maruyama scheme.
"""

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
    """Abstract base class for implementing schemes for numerical SDE integration.

    Attributes:
        _variable_dim (int): The physical dimension of the variables in the SDE.
        _noise_dim (int): The dimension of the noise in the SDE.
        _drift (sde_model_function): The drift function of the SDE.
        _diffusion (sde_model_function): The diffusion function of the SDE.
        _stochastic_integral (stochastic_integrals.BaseStochasticIntegral):
            The stochastic integral used in the scheme.

    Methods:
        __init__(): Initializes the BaseScheme class.
        compute_Step(): Abstract method defining the interface for single-step
                        numerical integration.
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

        Scheme classes take in the information of an SDE model and a 
        stochastic integegral generator. If the scheme requires higher order stochastic integral,
        it shhould implement checks if the provided integrator offers the required functionality.
        Single integrals are guaranteed to be implemented by all classes derived from
        BaseStochasticIntegral.

        This method employs run-time type checking.

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
        """Abstract method defining the interface for single-step numerical integration.

        Given a current state, time and step size, this method propagates the SDE solution one
        step forward. It has to be vectorized over different trajectories and ,for dynamics
        dynamic integrators, over different times and step sizes for the different trajectories.
        This is also part of the responsibility of the user-provided drift and diffusion functions.
        The method `check_consistency`, which should be called by every integrator before a
        simulation, tests drift/diffusion functions and initial condition for compatibility
        with each other and the invoking integrator
        `current state` is always a 2D array, where the first dimension is the physical dimension
        of the solution variables, and the second dimension accounts for different trajectories.
        For a static integrator, the current time and step size will be scalars. In the case of
        dynamics integration, both will be 1xN arrays, where N is the number of trajectories.

        Args:
            current_state (np.ndarray): The current state of the SDE.
            current_time (float | np.ndarray): The current time of the SDE.
            step_size (float | np.ndarray): The step size of the scheme.

        Returns:
            np.ndarray: The next state of the SDE.
        """
        pass


    # ----------------------------------------------------------------------------------------------
    def check_consistency(self, initial_state: np.ndarray, static_steps: bool) -> None:
        """Checks the consistency of the scheme, initial state, and model functions.

        This function tests the drift/diffusion functions and the initial condition for consistency
        with the integrator that calls it. The user provided drift and diffusion functions must
        produce output matching the variable and noise dimensions in their first dimension, and the
        number of trajectories in their second dimension. In addition, for dynami integrators, as
        indicated by the `static_steps` flag, the functions must work with 1xN time arrays, where N
        is the number of trajectories.

        Args:
            initial_state (np.ndarray): The initial state of the SDE.
            static_steps (bool, optional): Whether the steps are static or dynamic.

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
    """A class that implements the explicit Euler-Maruyama one-step scheme.

    The scheme requires the provided stochastic integral only to compute single integrals.
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

        For a detailed description of arguments and return types, see the base class docstring.

        Args:
            current_state (np.ndarray): The current state of the SDE.
            current_time (float | np.ndarray): The current time of the SDE.
            step_size (float | np.ndarray): The step size of the scheme.

        Returns:
            np.ndarray: The next state of the SDE.
        
        Raises:
            AssertionError: If the shape of the next state does not match that of the current state.
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