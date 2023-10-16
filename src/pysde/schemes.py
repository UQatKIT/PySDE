# =================================== Imports and Configuration ====================================
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

import numpy as np
from typeguard import typechecked

from pysde import stochastic_integrals


# ======================================== Scheme Base Class =======================================
class BaseScheme(ABC):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(
        self,
        variable_dim: int,
        noise_dim: int,
        drift_function: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
        diffusion_function: Callable[[np.ndarray, float | np.ndarray], np.ndarray],
        stochastic_integral: stochastic_integrals.BaseStochasticIntegral,
    ) -> None:
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
        pass

    # ----------------------------------------------------------------------------------------------
    def check_sde_model(self, initial_state: np.ndarray, static_steps: bool=True) -> None:
        if not initial_state.ndim == 2:
            raise ValueError(
                "Initial condition needs to be two-dimensional, "
                f"but has shape {initial_state.shape}."
            )
        if not initial_state.shape[0] == self._variable_dim:
            raise ValueError(
                f"Initial condition needs to have shape {self._variable_dim}, "
                f"but has shape {initial_state.shape[0]}."
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

    # ----------------------------------------------------------------------------------------------
    def _check_input(self, current_state: np.ndarray, step_size: float | np.ndarray) -> None:
        if not current_state.ndim == 2:
            raise ValueError(
                "Current state needs to be two-dimensional, "
                f"but has shape {current_state.shape}."
            )
        if not current_state.shape[0] == self._variable_dim:
            raise ValueError(
                "First dimension of the given state needs to be "
                f"{self._variable_dim}, but is {current_state.shape[0]}."
            )
        if np.any(np.array(step_size)) <= 0:
            raise ValueError("Step size must be positive.")


# ================================= Explicit Euler-Maruyama Scheme =================================
@final
class ExplicitEulerMaruyamaScheme(BaseScheme):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def compute_step(
        self,
        current_state: np.ndarray,
        current_time: float | np.ndarray,
        step_size: float | np.ndarray,
    ) -> np.ndarray:
        self._check_input(current_state, step_size)
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
