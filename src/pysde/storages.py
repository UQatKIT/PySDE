# =================================== Imports and Configuration ====================================
import os
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import final

import numpy as np
import zarr
from typeguard import typechecked


# ======================================= Storage Base Class =======================================
class BaseStorage(ABC):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | None = None) -> None:
        self._save_directory = save_directory
        self._save_directory_times = save_directory + "_times"
        self._save_directory_results = save_directory + "_results"
        self._time_list = []
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        self._time_list.append(time)
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._time_list = []
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> Collection:
        pass

    # ----------------------------------------------------------------------------------------------
    def _make_result_directory(self) -> None:
        pathlib_dir = pathlib.Path(self._save_directory)
        os.makedirs(pathlib_dir.parent, exist_ok=True)


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def get(self) -> tuple[np.ndarray]:
        time_array = np.stack(self._time_list, axis=0)
        result_array = np.stack(self._result_list, axis=2)
        if self._save_directory:
            self._make_result_directory()
            np.save(f"{self._save_directory_times}.npy", time_array)
            np.save(f"{self._save_directory_results}.npy", result_array)
        return time_array, result_array


# ==================================== Zarr Chunkwise Storage ======================================
@final
class ZarrChunkwiseStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | pathlib.Path, chunk_size: int) -> None:
        super().__init__(save_directory)
        self._chunk_size = chunk_size
        self._zarr_storage_times = None
        self._zarr_storage_result = None

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        super().append()
        if len(self._result_list) == self._chunk_size:
            self._save_to_file()
            super().reset()

    # ----------------------------------------------------------------------------------------------
    def get(self) -> zarr.Array:
        if self._result_list:
            self._save_to_file()
        return self._zarr_storage_times, self._zarr_storage_result

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._zarr_storage_times = None
        self._zarr_storage_result = None

    # ----------------------------------------------------------------------------------------------
    def _save_to_file(self) -> None:
        time_array = np.stack(self._time_list, axis=0)
        result_array = np.stack(self._result_list, axis=2)

        if self._zarr_storage_times is None:
            self._make_result_directory()
            self._zarr_storage_times = zarr.array(
                time_array, store=f"{self._save_directory_times}.zarr", overwrite=True
            )
        else:
            self._zarr_storage_times.append(time_array, axis=0)

        if self._zarr_storage_result is None:
            self._make_result_directory()
            self._zarr_storage = zarr.array(
                result_array, store=f"{self._save_directory_results}.zarr", overwrite=True
            )
        else:
            self._zarr_storage_times.append(result_array, axis=2)
