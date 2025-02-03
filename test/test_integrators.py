# =================================== Imports and Configuration ====================================
import numpy as np
import pytest

from pysde import integrators, schemes, stochastic_integrals, storages


# ===================================== Module-Level Fixtures ======================================
@pytest.fixture(params=[(storages.NumpyStorage, None), (storages.ZarrStorage, 100)])
def storage(request, tmp_path):
    storage_type, chunk_size = request.param
    if chunk_size:
        storage = storage_type(tmp_path / "data", chunk_size)
    else:
        storage = storage_type(tmp_path / "data")
    return storage


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=[stochastic_integrals.ItoStochasticIntegral], scope="module")
def stochastic_integral(request):
    stochastic_integal = request.param(seed=0)
    return stochastic_integal


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ornstein_uhlenbeck_model():
    def drift(current_state, current_time):
        return -current_state

    def diffusion(current_state, current_time):
        return np.ones((current_state.shape[0], current_state.shape[0], current_state.shape[1]))

    return drift, diffusion


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=[schemes.ExplicitEulerMaruyamaScheme], scope="module")
def scheme(stochastic_integral, ornstein_uhlenbeck_model, request):
    variable_dim = 2
    noise_dim = 2
    drift, diffusion = ornstein_uhlenbeck_model
    scheme = request.param(variable_dim, noise_dim, drift, diffusion, stochastic_integral)
    return scheme


# --------------------------------------------------------------------------------------------------
@pytest.fixture(params=[0, 1], scope="module")
def initial_condition(request):
    initial_condition_list = [
        np.array([1, 2]),
        np.ones((2, 10)),
    ]
    return initial_condition_list[request.param]


# ================================ Tests of Stand-Alone Functions ==================================
class TestReshapeInitialState:
    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "input_array, expected_reshaped_array",
        [
            (1, np.array([[1]])),
            (np.array([1, 2]), np.array([[1], [2]])),
            (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
        ],
    )
    def test_valid_input(self, input_array, expected_reshaped_array):
        output = integrators.reshape_initial_state(input_array)
        assert np.array_equal(output, expected_reshaped_array)

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("input_array", [np.zeros((1, 2, 3, 4))])
    def test_invalid_shape(self, input_array):
        with pytest.raises(ValueError):
            _ = integrators.reshape_initial_state(input_array)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("self_arg", [None, "self", 0.0])
def test_run_decorator_rejection(self_arg):
    @integrators.decorate_run_method
    def some_random_function(self_arg, *args, **kwags):
        pass

    with pytest.raises(TypeError):
        some_random_function(self_arg)


# ================================== Tests for Static Integrator ===================================
class TestStaticIntegrator:
    # ----------------------------------------------------------------------------------------------
    def test_init(self, scheme, storage):
        print(type(scheme))
        _ = integrators.StaticIntegrator(scheme, storage)

    # ----------------------------------------------------------------------------------------------
    def test_run(self, scheme, storage, initial_condition):
        start_time = 0
        step_size = 1e-3
        num_steps = 501

        integrator = integrators.StaticIntegrator(scheme, storage)
        time_array, result_array = integrator.run(
            initial_condition, start_time, step_size, num_steps
        )
        time_array_shape = (num_steps,)
        initial_condition = integrators.reshape_initial_state(initial_condition)
        result_array_shape = initial_condition.shape + (num_steps,)

        assert time_array_shape == time_array[:].shape
        assert result_array_shape == result_array[:].shape
