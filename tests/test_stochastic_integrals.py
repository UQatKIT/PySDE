# =================================== Imports and Configuration ====================================
import numpy as np
import pytest
from typeguard import TypeCheckError

from pysde import stochastic_integrals


# ==================================== Tests for Ito Integrals =====================================
class TestItoStochasticIntegral:
    # ----------------------------------------------------------------------------------------------
    def _compute_single_ito_integral(self, seed, noise_dim, step_size, num_trajectories):
        rng = np.random.default_rng(seed)
        single_integral = np.sqrt(step_size) * rng.normal(size=(noise_dim, num_trajectories))
        return single_integral
    
    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("seed", [0, 42])
    def test_ito_valid_init(self, seed):
        _ = stochastic_integrals.ItoStochasticIntegral(seed)
        assert True

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("seed", [-8.9, 0.14, None, "bla"])
    def test_ito_invalid_init(self, seed):
        with pytest.raises(TypeCheckError):
            _ = stochastic_integrals.ItoStochasticIntegral(seed)
    
    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("noise_dim", [1, 2])
    @pytest.mark.parametrize("num_trajectories, step_size", ([1, 1], [2, np.ones((1, 2))]))
    def test_ito_single_valid_input(self, noise_dim, step_size, num_trajectories):
        seed = 42
        benchmark_result = self._compute_single_ito_integral(seed,
                                                             noise_dim,
                                                             step_size,
                                                             num_trajectories)
        ito_integral = stochastic_integrals.ItoStochasticIntegral(seed)
        test_result = ito_integral.compute_single(noise_dim, step_size, num_trajectories)
        assert np.array_equal(test_result, benchmark_result)