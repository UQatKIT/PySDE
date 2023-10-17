"""_summary_."""
# =================================== Imports and Configuration ====================================
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import final

import numpy as np
import zarr
from typeguard import typechecked


# ======================================= Storage Base Class =======================================
class BaseStorage(ABC):
    """Abstract base class that provides a common interface for storing time and result data.

    Args:
        save_directory (str | None, optional): The directory where the data will be saved.
                                               Defaults to None.

    Attributes:
        _save_directory (pathlib.Path | None): The directory where the data will be saved.
        _time_list (list): A list to store the time data.
        _result_list (list): A list to store the result data.
    """

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | None = None) -> None:
        """Initializes the storage with an optional save directory.

        Args:
            save_directory (str | None, optional): The directory where the data will be saved.
                                                   Defaults to None.
        """
        self._save_directory = pathlib.Path(save_directory) if save_directory else None
        self._time_list = []
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        """Appends a time and result data to the storage.

        Args:
            time (float | np.ndarray): The time data to be appended.
            result (np.ndarray): The result data to be appended.
        """
        self._time_list.append(time)
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Resets the stored data."""
        self._time_list.clear()
        self._result_list.clear()

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> Collection:
        """Retrieves the stored data.

        Returns:
            Collection: The stored data.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    def _make_result_directory(self) -> None:
        """Creates the result directory if it doesn't exist."""
        pathlib_dir = pathlib.Path(self._save_directory)
        pathlib_dir.parent.mkdir(parents=True, exist_ok=True)


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    """A subclass of the BaseStorage abstract base class relying on numpy storage formats.

    Methods:
    - get(): Retrieves the stored data as a tuple of NumPy arrays. If a save directory is provided,
             it also creates the result directory and saves the data as a NumPy .npz file.
    """

    # ----------------------------------------------------------------------------------------------
    def get(self) -> tuple[np.ndarray]:
        """Retrieves the stored data as a tuple of NumPy arrays.
        
        If a save directory is provided, it also creates the result directory and saves the data
        as a NumPy .npz file.

        Returns:
        - time_array (np.ndarray): A NumPy array containing the stored time data.
        - result_array (np.ndarray): A NumPy array containing the stored result data.
        """
        time_array = np.stack(self._time_list, axis=0)
        result_array = np.stack(self._result_list, axis=2)
        if self._save_directory:
            self._make_result_directory()
            np.savez(f"{self._save_directory}.npz", times=time_array, results=result_array)
        return time_array, result_array


# ==================================== Zarr Chunkwise Storage ======================================
@final
class ZarrChunkwiseStorage(BaseStorage):
    """A class for storing time and result data in a chunkwise manner using the Zarr library.

    Args:
        save_directory (str | pathlib.Path): The directory where the Zarr file will be saved.
        chunk_size (int): The maximum number of results to store before saving to a Zarr file.

    Example Usage:
        storage = ZarrChunkwiseStorage(save_directory='data', chunk_size=100)
        storage.append(time=0.1, result=np.array([1, 2, 3]))
        storage.append(time=0.2, result=np.array([4, 5, 6]))
        storage.append(time=0.3, result=np.array([7, 8, 9]))
        times, results = storage.get()
        print(times)  # Output: Zarr array containing the stored time data
        print(results)  # Output: Zarr array containing the stored result data
    """

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | pathlib.Path, chunk_size: int) -> None:
        """Initializes the ZarrChunkwiseStorage object.

        Args:
            save_directory (str | pathlib.Path): The directory where the Zarr file will be saved.
            chunk_size (int): The maximum number of results to store before saving to a Zarr file.
        """
        super().__init__(save_directory)
        self._chunk_size = chunk_size
        self._zarr_storage_group = None
        self._zarr_storage_times = None
        self._zarr_storage_results = None

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        """Appends a time and result data to the storage.
        
        Saves the data to a Zarr file when the number of stored results reaches the chunk size.

        Args:
            time (float | np.ndarray): The time data to append.
            result (np.ndarray): The result data to append.
        """
        super().append(time, result)
        if len(self._result_list) >= self._chunk_size:
            self._save_to_file()
            super().reset()

    # ----------------------------------------------------------------------------------------------
    def get(self) -> zarr.Array:
        """Retrieves the stored time and result data as Zarr arrays.
        
        Saves the data to a Zarr file if there are any remaining stored results.

        Returns:
            zarr.Array: The Zarr array containing the stored time and result data.
        """
        if self._result_list:
            self._save_to_file()
        return self._zarr_storage_times, self._zarr_storage_results

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Resets the stored data and clears the Zarr storage."""
        super().reset()
        self._zarr_storage_group = None
        self._zarr_storage_times = None
        self._zarr_storage_result = None

    # ----------------------------------------------------------------------------------------------
    def _save_to_file(self) -> None:
        """Saves the stored time and result data to a Zarr file."""
        time_array = np.stack(self._time_list, axis=0)
        result_array = np.stack(self._result_list, axis=2)

        if self._zarr_storage_group is None:
            self._make_result_directory()
            self._zarr_storage_group = zarr.group(
                store=f"{self._save_directory}.zarr", overwrite=True
            )
            self._zarr_storage_times = self._zarr_storage_group.create_dataset(
                "times", shape=time_array.shape
            )
            self._zarr_storage_results = self._zarr_storage_group.create_dataset(
                "results", shape=result_array.shape
            )
            self._zarr_storage_times[:] = time_array
            self._zarr_storage_results[:] = result_array
        else:
            self._zarr_storage_times.append(time_array, axis=0)
            self._zarr_storage_results.append(result_array, axis=2)
