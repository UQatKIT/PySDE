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
    def __init__(self, save_directory: str | pathlib.Path | None=None) -> None:
        self._save_directory = save_directory
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def append(self, result: np.ndarray) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> Collection:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def reset(self) -> None:
        pass
    
    # ----------------------------------------------------------------------------------------------
    def _make_result_directory(self) -> None:
        pathlib_dir = pathlib.Path(self._save_directory)
        os.makedirs(pathlib_dir.parent, exist_ok=True)


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def get(self) -> np.ndarray:
        result_array = np.stack(self._result_list, axis=2)
        if self._save_directory:
            self._make_result_directory()
            np.save(f"{self._save_directory}.npy", result_array)
        return result_array

    # ----------------------------------------------------------------------------------------------     
    @typechecked
    def reset(self) -> None:
        self._result_list = []


# ==================================== Zarr Chunkwise Storage ======================================
@final
class ZarrChunkwiseStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | pathlib.Path, chunk_size: int) -> None:
        super().__init__(save_directory)
        self._chunk_size = chunk_size
        self._zarr_storage = None

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def append(self, result: np.ndarray) -> None:
        self._result_list.append(result)
        if len(self._result_list) == self._chunk_size:
            self._save_to_file()
            self._result_list = []

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def get(self) -> zarr.Array:
        if self._result_list:
            self._save_to_file()
        return self._zarr_storage
    
    # ----------------------------------------------------------------------------------------------
    @typechecked
    def reset(self) -> None:
        self._result_list = []
        self._zarr_storage = None

    # ----------------------------------------------------------------------------------------------
    def _save_to_file(self) -> None:
        result_array = np.stack(self._result_list, axis=2)
        if self._zarr_storage is None:
            self._make_result_directory()
            self._zarr_storage = zarr.array(
                result_array, store=f"{self._save_directory}.zarr", overwrite=True
            )
        else:
            self._zarr_storage.append(result_array, axis=2)