# ==================================================================================================
import pathlib
from abc import ABC, abstractmethod
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
        self, stride: Annotated[int, Is[lambda x: x > 0]], save_directory: pathlib.Path
    ) -> None:
        self._stride = stride
        self._save_directory = save_directory
        self._process_specific_directory = None
        self._time_list = []
        self._data_list = []

    # ----------------------------------------------------------------------------------------------
    def setup_parallel(self, process_id: Annotated[int, Is[lambda x: x > 0]]):
        self._process_specific_directory = self._save_directory.with_name(
            f"{self._save_directory.name}_p{process_id}"
        )
        self._process_specific_directory.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    def store(
        self,
        time: Real,
        data: npt.NDArray[np.floating],
        iteration_number: Annotated[int, Is[lambda x: x > 0]],
    ) -> None:
        if iteration_number % self._stride == 0:
            self._time_list.append(time)
            self._data_list.append(data)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError


# ==================================================================================================
class NumpyStorage(BaseStorage):
    # ----------------------------------------------------------------------------------------------
    def save(self) -> None:
        time_array = np.stack(self._time_list, axis=-1)
        data_array = np.stack(self._data_list, axis=-1)
        np.savez_compressed(self._process_specific_directory, time=time_array, data=data_array)
        self._time_list.clear()
        self._data_list.clear()


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
        iteration_number: Annotated[int, Is[lambda x: x > 0]],
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
        self._zarr_storage_group = zarr.group(
            store=self._process_specific_directory, overwrite=True
        )
        self._zarr_storage_time = self._zarr_storage_group.create_array(
            name="time", shape=time_array.shape, dtype=time_array.dtype, chunks=time_array.shape
        )
        self._zarr_storage_data = self._zarr_storage_group.create_array(
            name="data", shape=data_array.shape, dtype=data_array.dtype, chunks=data_array.shape
        )
        self._zarr_storage_time[:] = time_array
        self._zarr_storage_data[:] = data_array
