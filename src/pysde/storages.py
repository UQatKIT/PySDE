"""Storage Objects for SDE Integration Results.

Description:
- Custom storage objects for storing SDE integration results.
- Handle time and result data.
- Common interface that can be leveraged by integrator classes, abstracting away different 
  storage mechanisms

Classes:
    `BaseStorage`: Abstract base class that provides a common interface for storage objects.
    `NumpyStorage`: Storage relying on numpy file formats.
    `ZarrChunkwiseStorage`: Storage relying on zarr file formats, allowing automatic chunkwise
                            storage to disk for large simulations.
"""
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

    This base class enforces a uniform interface on all storage objects, regardless of the 
    respective storage mechanism. As a consequence, all storage objects can be used by any
    integrator, inclusing dynamic integrators where the form of the result arrays is not known
    a-priori.

    Methods:
        __init__(): Base class constructor.
        append(): Appends a time and result data to the storage.
        get(): Saves data to file and returns a numpy-like handle.
        reset(): Resets the stored data.

    Attributes:
        _save_directory (pathlib.Path | None): The directory where the data will be saved.
        _time_list (list): A list to temporarily store the time data during integration.
        _result_list (list): A list to temporarily store the result data during integration.
    """

    # ----------------------------------------------------------------------------------------------
    @typechecked
    def __init__(self, save_directory: str | None = None) -> None:
        """Initializes the storage with an optional save directory."""
        self._save_directory = pathlib.Path(save_directory) if save_directory else None
        self._time_list = []
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        """Appends a time and result data to the storage.

        By utilizing an append mechanism, storage objects do not need to know the size of the 
        result arrays upfront.This makes them suitable for static as well as dynamic integrators.

        Args:
            time (float | np.ndarray): The current time data to be appended. For a static
                                       integrator, this is a scalar. For a dynamic integrator,
                                       this is a 1xN array, where N is the number of
                                       trajectories.
            result (np.ndarray): The current result data to be appended. This is always a
                                 two-dimensional array of size M x N, where M is the physical
                                 problem dimension and N is the number of trajectories.
        """
        self._time_list.append(time)
        self._result_list.append(result)

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Resets the stored data.
        
        This method is important to re-utilize storage objects for successive runs. It should be
        called before integration to delete data from previous runs if existing.
        """
        self._time_list.clear()
        self._result_list.clear()

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> Collection:
        """Retrieves the stored data and saves to file.

        This abstract method defines the interface for retrieving the data from a storage object.
        Precisely, it performance two-different steps. Firstly, it saves the data to a file in
        some format, depending on if a save directory is provided. Secondly, it returns handles
        to the stored time and result data that behave very similar to numpy arrays.

        Returns:
            Collection: The stored time and result data. These might be actual numpy arrays or file
                        handles that implement a similar interface.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    def _make_result_directory(self) -> None:
        """Creates the provided result directory if it doesn't exist."""
        pathlib_dir = pathlib.Path(self._save_directory)
        pathlib_dir.parent.mkdir(parents=True, exist_ok=True)


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    """A subclass of the BaseStorage abstract base class relying on numpy file formats.

    This storage format should be utilized if the entire integration data fits into RAM.
    """

    # ----------------------------------------------------------------------------------------------
    def get(self) -> tuple[np.ndarray]:
        """Retrieves the stored data as a tuple of NumPy arrays.
        
        If a save directory is provided, it also creates the result directory and saves the data
        as a NumPy .npz file.

        Returns:
            time_array (np.ndarray): A NumPy array containing the stored time data.
            result_array (np.ndarray): A NumPy array containing the stored result data.
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

    This storage format can be used if the simulation data does not fit into RAM. The storage
    object will automatically only store a limited number of integration steps at once. When that
    number is exceeded, the data held in memory is flushed to a Zarr data storage and the RAM
    storage is reset. Note that providing a save doirectory for this class ism mandatory.

    Attributes:
        _chunk_size (int): The maximum number of results to store before saving to a Zarr file.
        _zarr_storage_group (zarr.Group): The Zarr group where the data will be stored.
        _zarr_storage_times (zarr.Array): The Zarr array containing the time data.
        _zarr_storage_results (zarr.Array): The Zarr array containing the result data.
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
        
        Additionally saves the data to a Zarr file when the number of stored results reaches the
        chunk size.

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
        
        Saves all current data to Zarr storage and returns handles to the stored time and result.
        These handles behave very similar to numpy arrays. The only pitfall is that when accessing
        the entire array, one has to call zarr_array[:] instead of just zarr_array.

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
        """Saves the stored time and result data to a Zarr group."""
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
