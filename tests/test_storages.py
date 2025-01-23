# =================================== Imports and Configuration ====================================
import numpy as np
import pytest
import zarr

from pysde import storages


# ===================================== Module-Level Fixtures ======================================
@pytest.fixture(scope="module")
def test_arrays_scalar_time():
    rng = np.random.default_rng()
    scalar_time_array_list = []
    result_array_list = []
    variable_dim = 2
    num_trajectories = 10

    for i in range(3):
        scalar_time_array_list.append(i)
        result_array_list.append(rng.uniform(-1, 1, size=(variable_dim, num_trajectories)))
    return scalar_time_array_list, result_array_list


# --------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def test_arrays_vector_time():
    rng = np.random.default_rng()
    vector_time_array_list = []
    result_array_list = []
    variable_dim = 2
    num_trajectories = 10

    for _ in range(3):
        vector_time_array_list.append(rng.uniform(-1, 1, size=(num_trajectories,)))
        result_array_list.append(rng.uniform(-1, 1, size=(variable_dim, num_trajectories)))
    return vector_time_array_list, result_array_list


# ==================================== Tests for Numpy Storage =====================================
class TestNumpyStorage:
    # ----------------------------------------------------------------------------------------------
    @pytest.fixture
    def test_storage(self, tmp_path):
        test_storage = storages.NumpyStorage(tmp_path / "data")
        return test_storage

    # ----------------------------------------------------------------------------------------------
    @pytest.fixture(params=("test_arrays_scalar_time", "test_arrays_vector_time"))
    def test_storage_filled(self, test_storage, request):
        time_array_list, result_array_list = request.getfixturevalue(request.param)
        for time, result in zip(time_array_list, result_array_list):
            test_storage.append(time, result)
        return test_storage, time_array_list, result_array_list

    # ----------------------------------------------------------------------------------------------
    def test_init_storage(self, tmp_path):
        _ = storages.NumpyStorage(tmp_path)
        _ = storages.NumpyStorage()
        assert True

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("arrays", ("test_arrays_scalar_time", "test_arrays_vector_time"))
    def test_append(self, test_storage, arrays, request):
        time_array_list, result_array_list = request.getfixturevalue(arrays)
        for time, result in zip(time_array_list, result_array_list):
            test_storage.append(time, result)

        for i, time in enumerate(time_array_list):
            assert np.array_equal(time, test_storage._time_list[i])
        for i, result in enumerate(result_array_list):
            assert np.array_equal(result, test_storage._result_list[i])

    # ----------------------------------------------------------------------------------------------
    def test_reset(self, test_storage):
        test_storage.reset()
        assert not test_storage._time_list
        assert not test_storage._result_list

    # ----------------------------------------------------------------------------------------------
    def test_save(self, test_storage_filled):
        test_storage, time_array_list, result_array_list = test_storage_filled
        save_dir = test_storage._save_directory
        test_storage.save()

        time_array = np.stack(time_array_list, axis=0)
        result_array = np.stack(result_array_list, axis=2)
        save_file = save_dir.parent / (save_dir.name + ".npz")
        assert save_file.is_file()

        with np.load(save_file) as data:
            saved_time_array = data["times"]
            saved_result_array = data["results"]
        assert np.array_equal(time_array, saved_time_array)
        assert np.array_equal(result_array, saved_result_array)

    # ----------------------------------------------------------------------------------------------
    def test_get(self, test_storage_filled):
        test_storage, time_array_list, result_array_list = test_storage_filled

        time_array = np.stack(time_array_list, axis=0)
        result_array = np.stack(result_array_list, axis=2)

        storage_time_array, storage_result_array = test_storage.get()
        assert np.array_equal(time_array, storage_time_array)
        assert np.array_equal(result_array, storage_result_array)


# =============================== Tests for Zarr Chunkwise Storage =================================
class TestZarrChunkwiseStorage:
    # ----------------------------------------------------------------------------------------------
    @pytest.fixture(params=[1, 3, 10])
    def test_storage(self, tmp_path, request):
        chunk_size = request.param
        test_storage = storages.ZarrChunkwiseStorage(tmp_path / "data", chunk_size)
        return test_storage

    # ----------------------------------------------------------------------------------------------
    @pytest.fixture(params=("test_arrays_scalar_time", "test_arrays_vector_time"))
    def test_storage_filled(self, test_storage, request):
        time_array_list, result_array_list = request.getfixturevalue(request.param)
        for time, result in zip(time_array_list, result_array_list):
            test_storage.append(time, result)
        return test_storage, time_array_list, result_array_list

    # ----------------------------------------------------------------------------------------------
    def test_init_storage(self, tmp_path):
        _ = storages.ZarrChunkwiseStorage(tmp_path, 1)
        assert True

    # ----------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("arrays", ("test_arrays_scalar_time", "test_arrays_vector_time"))
    def test_append(self, test_storage, arrays, request):
        time_array_list, result_array_list = request.getfixturevalue(arrays)
        for time, result in zip(time_array_list, result_array_list):
            test_storage.append(time, result)
        assert True

    # ----------------------------------------------------------------------------------------------
    def test_reset(self, test_storage):
        test_storage.save()
        test_storage.reset()
        assert not test_storage._time_list
        assert not test_storage._result_list
        assert not test_storage._zarr_storage_group
        assert not test_storage._zarr_storage_times
        assert not test_storage._zarr_storage_results

    # ----------------------------------------------------------------------------------------------
    def test_save(self, test_storage_filled):
        test_storage, time_array_list, result_array_list = test_storage_filled
        save_dir = test_storage._save_directory
        test_storage.save()

        time_array = np.stack(time_array_list, axis=0)
        result_array = np.stack(result_array_list, axis=2)
        save_file = save_dir.parent / (save_dir.name + ".zarr")
        assert save_file.is_dir()

        with zarr.open_group(save_file) as data:
            saved_time_array = data["times"]
            saved_result_array = data["results"]
        assert np.array_equal(time_array, saved_time_array[:])
        assert np.array_equal(result_array, saved_result_array[:])

    # ----------------------------------------------------------------------------------------------
    def test_get(self, test_storage_filled):
        test_storage, time_array_list, result_array_list = test_storage_filled

        time_array = np.stack(time_array_list, axis=0)
        result_array = np.stack(result_array_list, axis=2)

        test_storage.save()
        storage_time_array, storage_result_array = test_storage.get()
        assert np.array_equal(time_array, storage_time_array[:])
        assert np.array_equal(result_array, storage_result_array[:])
