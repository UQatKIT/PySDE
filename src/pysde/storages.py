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
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        stride: Annotated[int, Is[lambda x: x > 0]],
        save_directory: pathlib.Path | None = None,
    ) -> None:
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
        if iteration_number % self._stride == 0:
            self._time_list.append(time)
            self._data_list.append(data)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def values(self) -> Iterable:
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        if self._save_directory is not None:
            np.savez_compressed(self._save_directory, time=time_array, data=data_array)

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        return time_array, data_array


# ==================================================================================================
class ZarrChunkwiseStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        stride: Annotated[int, Is[lambda x: x > 0]],
        chunk_size: Annotated[int, Is[lambda x: x > 0]],
        save_directory: pathlib.Path,
    ) -> None:
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
        if len(self._time_list) >= self._chunk_size:
            self._save_to_disk()
            self._time_list.clear()
            self._data_list.clear()
        super().store(time, data, iteration_number)

    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        self._save_to_disk()
        self._time_list.clear()
        self._data_list.clear()

    # ----------------------------------------------------------------------------------------------
    @property
    def values(self) -> tuple[zarr.Array, zarr.Array]:
        if (self._zarr_storage_times is None) or (self._zarr_storage_data is None):
            raise ValueError("No data has been saved to disk yet.")
        return self._zarr_storage_time, self._zarr_storage_data

    # ----------------------------------------------------------------------------------------------
    def _save_to_disk(self) -> None:
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
        self._zarr_storage_group = zarr.group(store=self._save_directory, overwrite=True)
        self._zarr_storage_time = self._zarr_storage_group.create_array(
            name="time", shape=time_array.shape, dtype=time_array.dtype, chunks=time_array.shape
        )
        self._zarr_storage_data = self._zarr_storage_group.create_array(
            name="data", shape=data_array.shape, dtype=data_array.dtype, chunks=data_array.shape
        )
        self._zarr_storage_time[:] = time_array
        self._zarr_storage_data[:] = data_array
