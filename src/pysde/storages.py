"""Storage Objects for SDE Integration Results.

This module provides custom storage objects for SDE integration results. They handle time and
result arrays. All storage objects provide a common interface that can be leveraged by integrator
objects, abstracting away different storage mechanisms within the integration procedure.

Classes:
    BaseStorage: Abstract base class that provides a common interface for storage objects.
    NumpyStorage: Storage relying on numpy file formats.
    ZarrChunkwiseStorage: Storage relying on zarr file formats, allowing automatic chunkwise
                          storage to disk for large simulations.
"""

# =================================== Imports and Configuration ====================================
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

import numpy as np
import zarr


# ======================================= Storage Base Class =======================================
class BaseStorage(ABC):
    """Abstract base class that provides a common interface for storing time and result data.

    This base class enforces a uniform interface on all storage objects, regardless of the
    respective storage mechanism. As a consequence, all storage objects can be used by any
    integrator, including dynamic integrators where the shape of the result arrays is not known
    a-priori.

    Methods:
        __init__(): Base class constructor.
        append(): Appends time and result data to the storage.
        reset(): Resets the data storage.
        save(): Save the stored data to disk.
        get(): Returns a numpy-like handle.

    Attributes:
        _save_directory (pathlib.Path | None): The directory where the data will be saved.
        _time_list (list): A list to temporarily store the time data during integration.
        _result_list (list): A list to temporarily store the result data during integration.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, save_directory: str | pathlib.Path | None = None) -> None:
        """Initializes the storage with an optional save directory."""
        self._save_directory = pathlib.Path(save_directory) if save_directory else None
        self._time_list = []
        self._result_list = []

    # ----------------------------------------------------------------------------------------------
    def append(self, time: float | np.ndarray, result: np.ndarray) -> None:
        """Appends time and result data to the storage.

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
        self._reset_storage_lists()

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self) -> None:
        """Saves the internal data to disk, depending on the implemented storage format."""
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get(self) -> tuple[Iterable, None]:
        """Retrieves the stored data in form of numpy-like handles.

        Returns:
            Iterable: The stored time and result data. These might be actual numpy arrays or file
                        handles that implement a similar interface.
        """
        pass

    # ----------------------------------------------------------------------------------------------
    def _make_result_directory(self) -> None:
        """Creates the requested result directory if it doesn't exist."""
        self._save_directory.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    def _prepare_output_arrays(self) -> tuple[np.ndarray]:
        """Stacks time and result data into numpy arrays for further processing."""
        time_array = np.stack(self._time_list, axis=0)
        result_array = np.stack(self._result_list, axis=2)
        return time_array, result_array

    # ----------------------------------------------------------------------------------------------
    def _reset_storage_lists(self) -> tuple[np.ndarray]:
        """Reset the lists that store the data during integration."""
        self._time_list.clear()
        self._result_list.clear()


# ====================================== Numpy Result Storage ======================================
@final
class NumpyStorage(BaseStorage):
    """A subclass of the BaseStorage abstract base class relying on numpy file formats.

    This storage format should be utilized if the entire integration data fits into RAM.
    """

    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        """Saves data to a npz file, if a save direcrtory has been provided.

        Times are stored in an array named "times", results in an array named "results".
        """
        if self._time_list and self._result_list:
            if self._save_directory:
                self._make_result_directory()
                time_array, result_array = self._prepare_output_arrays()
                np.savez(f"{self._save_directory}.npz", times=time_array, results=result_array)

    # ----------------------------------------------------------------------------------------------
    def get(self) -> tuple[np.ndarray | None]:
        """Retrieves the stored data as a tuple of NumPy arrays.

        Returns:
            time_array (np.ndarray): A NumPy array containing the stored time data.
            result_array (np.ndarray): A NumPy array containing the stored result data.
        """
        if self._time_list and self._result_list:
            time_array, result_array = self._prepare_output_arrays()
            return time_array, result_array
        else:
            return None, None


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
            self.save()
            self._reset_storage_lists()

    # ----------------------------------------------------------------------------------------------
    def get(self) -> tuple[zarr.Array, None]:
        """Retrieves the stored time and result data as Zarr arrays.

        Saves all current data to Zarr storage and returns handles to the stored time and result.
        These handles behave very similar to numpy arrays. The only pitfall is that when accessing
        the entire array, one has to call zarr_array[:] instead of just zarr_array.

        Returns:
            zarr.Array: The Zarr array containing the stored time and result data.
        """
        return self._zarr_storage_times, self._zarr_storage_results

    # ----------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Resets the stored data and clears the Zarr storage."""
        super().reset()
        self._zarr_storage_group = None
        self._zarr_storage_times = None
        self._zarr_storage_results = None

    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        """Saves the stored time and result data to a Zarr group."""
        if self._time_list and self._result_list:
            time_array, result_array = self._prepare_output_arrays()

            if self._zarr_storage_group is None:
                self._make_result_directory()
                self._create_zarr_storage(time_array.shape, result_array.shape)
                self._zarr_storage_times[:] = time_array
                self._zarr_storage_results[:] = result_array
            else:
                self._zarr_storage_times.append(time_array, axis=0)
                self._zarr_storage_results.append(result_array, axis=2)

    # ----------------------------------------------------------------------------------------------
    def _create_zarr_storage(
        self, time_array_shape, result_array_shape
    ) -> tuple[zarr.Group, zarr.Array]:
        """Initializes Zarr group when `save()` is called for the first time."""
        self._zarr_storage_group = zarr.group(store=f"{self._save_directory}.zarr", overwrite=True)
        self._zarr_storage_times = self._zarr_storage_group.create_dataset(
            "times", shape=time_array_shape
        )
        self._zarr_storage_results = self._zarr_storage_group.create_dataset(
            "results", shape=result_array_shape
        )
