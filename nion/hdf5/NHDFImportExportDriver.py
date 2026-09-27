from __future__ import annotations

import json
import logging
import pathlib
import time
import typing
import uuid

import h5py
import hdf5plugin
import numpy
import numpy.typing

from nion.swift.model import StorageHandler
from nion.utils import DateTime
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.data import DataAndMetadata

PersistentDictType = typing.Dict[str, typing.Any]


def _create_dataset(data_group: typing.Any, dataset_id: str, data: numpy.typing.NDArray[typing.Any], filter: h5py.filters.FilterRefBase | None = None) -> typing.Any:
    chunks: tuple[int, ...] | None | bool = getattr(data, "chunks", True)
    data_shape = data.shape

    # create the dataset, preallocate space.
    ds = data_group.create_dataset(dataset_id, shape=data_shape, dtype=data.dtype, compression=filter, chunks=chunks)

    if callable(getattr(data, "iter_chunks", None)):
        # if the data has an iter_chunks method, use it to write the data in chunks.
        for selection in getattr(data, "iter_chunks")():
            ds[selection] = data[selection]
        return ds

    # search for the chunk size by iterating backwards through the shape and finding the
    # largest chunk size that is less than 64MB.
    index_count = 0
    chunk_size = 1
    for n in reversed(data_shape):
        if chunk_size * n > 64 * 1024 * 1024:
            break
        index_count += 1
        chunk_size *= n

    # iterate over the remaining dimensions so that we can write the data in chunks.
    for index in numpy.ndindex(*data_shape[:len(data_shape) - index_count]):
        selection = tuple(index) + (slice(None),) * index_count
        ds[selection] = data[selection]

    return ds


class HDFImportExportDriver:
    def __init__(self, filter: h5py.filters.FilterRefBase | None = None) -> None:
        self.__filter = filter

    def read_data(self, file_path: pathlib.Path, storage_handler_provider: StorageHandler.StorageHandlerProvider) -> StorageHandler.StorageHandlerImportData:
        t0 = time.time()
        storage_handlers = list[StorageHandler.StorageHandler]()
        uuid_map = dict[uuid.UUID, uuid.UUID]()
        items = list[PersistentDictType]()
        fp = h5py.File(file_path, "r")
        if "data" in fp:
            data_group = fp["data"]
            for key in sorted(data_group.keys()):
                ds = data_group[key]
                data_item_uuid = uuid.uuid4()
                data_item_properties = json.loads(ds.attrs["properties"])
                data_item_data = ds
                if "uuid" in data_item_properties:
                    uuid_map[uuid.UUID(data_item_properties["uuid"])] = data_item_uuid
                data_item_properties["uuid"] = str(data_item_uuid)
                is_sequence = data_item_properties.get("is_sequence", False)
                collection_dimension_count = data_item_properties.get("collection_dimension_count", 0)
                datum_dimension_count = data_item_properties.get("datum_dimension_count", 0)
                data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
                assert data_descriptor.expected_dimension_count == len(ds.shape)
                storage_handler_attributes = StorageHandler.StorageHandlerAttributes(
                    data_item_uuid,
                    DateTime.utcnow(),
                    data_item_properties.get("session_id", None),
                    ds.nbytes
                )
                storage_handler = storage_handler_provider.make_storage_handler(storage_handler_attributes)
                storage_handler.write_data(data_item_data, data_descriptor, storage_handler_attributes.created_local)
                storage_handler.write_properties(data_item_properties, storage_handler_attributes.created_local)
                storage_handlers.append(storage_handler)
        if "index" in fp:
            index_group = fp["index"]
            for key in sorted(index_group.attrs.keys()):
                items.append(json.loads(index_group.attrs[key]))
        fp.close()
        logging.getLogger("import").info(f"Import of {file_path} took {time.time() - t0:.2f} seconds")
        return StorageHandler.StorageHandlerImportData(storage_handlers, uuid_map, items)

    def write_display_item(self, path: pathlib.Path, items: typing.Sequence[StorageHandler.StorageHandlerExportItem]) -> None:
        path.unlink(missing_ok=True)
        fp = h5py.File(path, "a")
        index_group = fp.create_group("index")
        data_group = fp.create_group("data")
        data_index = 0
        for index, item in enumerate(items):
            index_group.attrs["1"] = json.dumps(item.write_to_dict())
            for data_item in item.data_items:
                ds = _create_dataset(data_group, str(data_index), data_item.data, self.__filter)
                ds.attrs["properties"] = json.dumps(data_item.write_to_dict())
                data_index += 1
        fp.close()


class CompressedHDFImportExportDriverFactory:
    driver_id = "nhdf-io-handler-compressed"
    title = "NData HDF (lz4 compressed)"
    extensions = ["nhdf"]

    def make_import_export_driver(self) -> HDFImportExportDriver:
        return HDFImportExportDriver(filter=hdf5plugin.Blosc(cname='lz4', clevel=9, shuffle=hdf5plugin.Blosc.BITSHUFFLE))


Registry.register_component(CompressedHDFImportExportDriverFactory(), {"import-export-driver-factory"})
