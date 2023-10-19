# =================================== Imports and Configuration ====================================
import numpy as np
import pytest

from pysde import stochastic_integrals, schemes


# ===================================== Module-Level Fixtures ======================================
@pytest.fixture(scope="module")
def correct_ornstein_uhlenbeck_model():
    def drift(current_state, current_time):
        return -current_state

    def diffusion(current_state, current_time):
        return np.ones((current_state.shape[0], current_state.shape[0], current_state.shape[1]))

    return drift, diffusion


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def incorrect_ornstein_uhlenbeck_model():
    def drift(current_state, current_time):
        return -current_state[0]

    def diffusion(current_state, current_time):
        return np.ones(current_state.shape)

    return drift, diffusion


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=[0, 1, 2, 3], scope="module")
def initial_condition(request):
    initial_condition_list = [
        np.array([[1]]),
        np.array([[1], [2]]),
        np.array([[1, 2]]),
        np.ones((4, 10)),
    ]
    return initial_condition_list[request.param]


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=[stochastic_integrals.ItoStochasticIntegral], scope="module")
def stochastic_integral(request):
    stochastic_integral = request.param(seed=0)
    return stochastic_integral


# =============================== Tests for Explicit Euler-Maruyama ================================
class TestExplicitEulerMaruyamaScheme:
    # ----------------------------------------------------------------------------------------------
    @pytest.fixture(scope="class")
    def correct_scheme(
        self,
        correct_ornstein_uhlenbeck_model,
        initial_condition,
        stochastic_integral,
    ):
        drift, diffusion = correct_ornstein_uhlenbeck_model
        variable_dimension = initial_condition.shape[0]
        noise_dimension = initial_condition.shape[0]
        scheme = schemes.ExplicitEulerMaruyamaScheme(
            variable_dimension, noise_dimension, drift, diffusion, stochastic_integral
        )
        return scheme, initial_condition

    # ----------------------------------------------------------------------------------------------
    @pytest.fixture(scope="class")
    def incorrect_scheme(
        self,
        incorrect_ornstein_uhlenbeck_model,
        initial_condition,
        stochastic_integral,
    ):
        drift, diffusion = incorrect_ornstein_uhlenbeck_model
        variable_dimension = initial_condition.shape[0]
        noise_dimension = initial_condition.shape[0]
        scheme = schemes.ExplicitEulerMaruyamaScheme(
            variable_dimension, noise_dimension, drift, diffusion, stochastic_integral
        )
        return scheme, initial_condition

    # ----------------------------------------------------------------------------------------------
    def test_init(
        self,
        correct_ornstein_uhlenbeck_model,
        initial_condition,
        stochastic_integral,
    ):
        drift, diffusion = correct_ornstein_uhlenbeck_model
        variable_dimension = initial_condition.shape[0]
        noise_dimension = initial_condition.shape[0]
        _ = schemes.ExplicitEulerMaruyamaScheme(
            variable_dimension, noise_dimension, drift, diffusion, stochastic_integral
        )

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("is_static", (True, False))
    def test_check_correct(self, correct_scheme, is_static):
        scheme, initial_condition = correct_scheme
        scheme.check_consistency(initial_condition, is_static)
        assert True

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("is_static", (True, False))
    def test_check_incorrect(self, incorrect_scheme, is_static):
        scheme, initial_condition = incorrect_scheme
        with pytest.raises(ValueError):
            scheme.check_consistency(initial_condition, is_static)

    # ----------------------------------------------------------------------------------------------
    def test_compute_step(self, correct_scheme):
        scheme, initial_condition = correct_scheme
        current_time_scalar = 0.0
        time_step_scalar = 0.1
        current_time_vector = np.zeros((1, initial_condition.shape[1]))
        time_step_vector = 0.1 * np.ones((1, initial_condition.shape[1]))

        next_state = scheme.compute_step(initial_condition, current_time_scalar, time_step_scalar)
        assert next_state.shape == initial_condition.shape

        next_state = scheme.compute_step(initial_condition, current_time_vector, time_step_vector)
        assert next_state.shape == initial_condition.shape


    
