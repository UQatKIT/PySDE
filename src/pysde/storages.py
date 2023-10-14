# =================================== Imports and Configuration ====================================
import os
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import final

import numpy as np
import zarr


# ======================================= Storage Base Class =======================================
class BaseStorage(ABC):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, save_directory: str | None=None) -> None:
        self._save_directory = save_directory
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def append(self, result: np.ndarray) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> Sequence:
        pass


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    def get(self) -> np.ndarray:
        result_array = np.stack(self._result_list, axis=2)
        if self._save_directory:
            pathlib_dir = pathlib.Path(self._save_directory)
            os.makedirs(pathlib_dir.parent, exist_ok=True)
            np.save(pathlib_dir, result_array)
        return result_array


# ==================================== Zarr Chunkwise Storage ======================================
@final
class ZarrChunkwiseStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, save_directory: str, chunk_size: int) -> None:
        super().__init__(save_directory)
        self._chunk_size = chunk_size
        self._zarr_storage = None

    # ----------------------------------------------------------------------------------------------
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)
        if len(self._result_list) == self._chunk_size:
            self._save_to_file()
            self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def get(self) -> zarr.Array:
        if self._result_list:
            self._save_to_file()
        return self._zarr_storage

    # ----------------------------------------------------------------------------------------------
    def _save_to_file(self) -> None:
        result_array = np.stack(self._result_list, axis=2)
        if self._zarr_storage is None:
            self._zarr_storage = zarr.array(
                result_array, store=f"{self._save_directory}.zarr", overwrite=True
            )
        else:
            self._zarr_storage.append(result_array, axis=2)