"""Storage objects for large data processing.

For SDE simulations with large numbers of trajectories and time steps, the resulting data often
does not fit into RAM. A possible solution is to only store statistics or implement strides. In
PySDE, we provide a simple option for actually storing all the data, and process it a-posteriori.
This is realized by data containers that automatically flush data from memory to disk. The data
storage backend we employ is [Zarr](https://zarr.readthedocs.io/en/stable/), but any other backend can easily be implemented by deriving from
the [`BaseStorage`][pysde.storages.BaseStorage] class.

classes:
    BaseStorage: ABC for data storage objects.
    NumpyStorage: Storage object with simple Numpy backend.
    ZarrStorage: Storage object with Zarr backend and chunkwise saving.
"""

# ==================================================================================================
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from numbers import Real
from typing import Annotated

import numpy as np
import numpy.typing as npt
import zarr
from beartype.vale import Is


# ==================================================================================================
class BaseStorage(ABC):
    """ABC for data storage objects.

    This class provides the minimal interface for storage objects. This includes a stride for sample
    saves and a path to save data on disk to. Automatic flushing to disk is not enforced on the
    ABC level, but only included in backend-specific implementations. The internal structure of
    storage objects is very simplistic. It consists of a list of scalars (for time) and a list of
    numpy arrays (for trajectory data). These lists is appended to when the
    [`store`][pysde.storages.BaseStorage.store] method is called. Further processing of the data is
    backend-specific.

    Methods:
        store: Store current time and trajectory ensemble in memory.
        save: Manual call to save in-memory data to disk.
        values: Return numpy-like handles to time and trajectory data.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        stride: Annotated[int, Is[lambda x: x > 0]],
        save_directory: pathlib.Path | None = None,
    ) -> None:
        r"""Initialize storage object with a stride and a save directory.

        Args:
            stride (int): Stride for saving every $n-th$ sample
            save_directory (pathlib.Path | None, optional): Path to save data to, on manual call or
                potentially automatically. Defaults to None.
        """
        self._stride = stride
        self._save_directory = save_directory
        self._time_list = []
        self._data_list = []

    # ----------------------------------------------------------------------------------------------
    def store(
        self,
        time: Real,
        data: npt.NDArray[np.floating],
        iteration_number: Annotated[int, Is[lambda x: x >= 0]],
    ) -> None:
        r"""Store current time and trajectory ensenmble in memory, if stride is reached.

        Args:
            time (Real): Current value of time variable of the stochastic process
            data (npt.NDArray[np.floating]): Trajectory data of size $d_X \times N$
            iteration_number (int): Iteration number of the calling integrator.
        """
        if iteration_number % self._stride == 0:
            self._time_list.append(time)
            self._data_list.append(data)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self) -> None:
        """Manual call to save in-memory data to disk.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def values(self) -> Iterable:
        """Return numpy-like handles to time and trajectory data.

        Raises:
            NotImplementedError: Needs to be implemented by subclasses.
        """
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(BaseStorage):
    """Storage object with simple Numpy backend.

    Data in this storage object is not automatically flushed to disk, but can be stored manually
    in `npz` format using the [`save`][pysde.storages.NumpyStorage.save] method.

    Methods:
        save: Stack internal lists to numpy arrays and save to disk in compressed `npz` format.
        values: Stack internal lists to numpy arrays and return time and trajectory arrays.
    """

    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        """Stack internal lists to numpy arrays and save to disk in compressed `npz` format.

        For $M$ saved snapshots, the time array has shape $(M,)$ and the data array has shape
        $(d_X, N, M)$.
        """
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        if self._save_directory is not None:
            np.savez_compressed(self._save_directory, time=time_array, data=data_array)

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        """Stack internal lists to numpy arrays and return time and trajectory arrays.

        For $M$ saved snapshots, return time array of shape $(M,)$ and data array of shape
        $(d_X, N, M)$.
        """
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        return time_array, data_array


# ==================================================================================================
class ZarrStorage(BaseStorage):
    """Storage object with [Zarr](https://zarr.readthedocs.io/en/stable/) backend and chunkwise saving.

    Zarr is a powerful storage backend inspired by the HDF5 format. It provides a numpy-like API
    while storing data on disk in a compressed and chunked format. This storage object saves data
    automatically to disk in regular intervals, making it suitable for SDE runs with large ensembles
    and/or long integration times. The data is stored in a
    [Zarr group](https://zarr.readthedocs.io/en/stable/user-guide/groups.html).
    """  # noqa: E501

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        stride: Annotated[int, Is[lambda x: x > 0]],
        chunk_size: Annotated[int, Is[lambda x: x > 0]],
        save_directory: pathlib.Path,
    ) -> None:
        """Initialize Zarr storage object with a stride, chunk size, and save directory.

        Args:
            stride (int): Stride for saving every $n-th$ sample
            chunk_size (int): Number of snapshots to store in memory before flushing to disk
            save_directory (pathlib.Path): Path to save data to
        """
        super().__init__(stride, save_directory)
        self._chunk_size = chunk_size
        self._zarr_storage_group = None
        self._zarr_storage_time = None
        self._zarr_storage_data = None

    # ----------------------------------------------------------------------------------------------
    def store(
        self,
        time: Real,
        data: npt.NDArray[np.floating],
        iteration_number: Annotated[int, Is[lambda x: x >= 0]],
    ) -> None:
        r"""Store current time and trajectory ensemble in memory, if stride is reached.

        If the local buffer is full, as defined by `chunk_size`, the data is flushed to disk.

        Args:
            time (Real): Current value of time variable of the stochastic process
            data (npt.NDArray[np.floating]): Trajectory data of size $d_X \times N$
            iteration_number (int): Iteration number of the calling integrator.
        """
        if len(self._time_list) >= self._chunk_size:
            self._save_to_disk()
            self._time_list.clear()
            self._data_list.clear()
        super().store(time, data, iteration_number)

    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        """Flush remaining data to disk and clear the local buffer.

        For $M$ saved snapshots, the time array has shape $(M,)$ and the data array has shape
        $(d_X, N, M)$.
        """
        self._save_to_disk()
        self._time_list.clear()
        self._data_list.clear()

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> tuple[zarr.Array, zarr.Array]:
        """Return Zarr handles to time and trajectory data.

        Numpy-like handles to time and trajectory data. For $M$ saved snapshots, return time array
        of shape $(M,)$ and data array of shape $(d_X, N, M)$.

        Raises:
            ValueError: Checks that internal storage has been initialized. This is done
                automatically when the first chunk of data is flushed to disk.
        """
        if (self._zarr_storage_times is None) or (self._zarr_storage_data is None):
            raise ValueError("No data has been saved to disk yet.")
        return self._zarr_storage_time, self._zarr_storage_data

    # ----------------------------------------------------------------------------------------------
    def _save_to_disk(self) -> None:
        """Stack internal lists to numpy arrays and save to Zarr storages on disk."""
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        if (self._zarr_storage_times is None) or (self._zarr_storage_data is None):
            self._init_storage_and_fill(time_array, data_array)
        else:
            self._zarr_storage_time.append(time_array, axis=0)
            self._zarr_storage_data.append(data_array, axis=0)

    # ----------------------------------------------------------------------------------------------
    def _init_storage_and_fill(
        self, time_array: npt.NDArray[np.floating], data_array: npt.NDArray[np.floating]
    ) -> None:
        """Init Zarr storage and fill with initial time and trajectory data."""
        self._zarr_storage_group = zarr.group(store=self._save_directory, overwrite=True)
        self._zarr_storage_time = self._zarr_storage_group.create_array(
            name="time", shape=time_array.shape, dtype=time_array.dtype, chunks=time_array.shape
        )
        self._zarr_storage_data = self._zarr_storage_group.create_array(
            name="data", shape=data_array.shape, dtype=data_array.dtype, chunks=data_array.shape
        )
        self._zarr_storage_time[:] = time_array
        self._zarr_storage_data[:] = data_array
