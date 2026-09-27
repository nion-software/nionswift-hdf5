import unittest

import h5py
import numpy

from nion.hdf5 import NHDFImportExportDriver

class TestLibrary(unittest.TestCase):

    def setUp(self) -> None:
        """Common code for all tests can go here."""
        pass

    def tearDown(self) -> None:
        """Common code for all tests can go here."""
        pass

    def test_import_export_driver(self) -> None:
        pass

    def test_create_dataset_from_unchunked_h5py_source(self) -> None:
        # regression test: an h5py Dataset that is not chunked (contiguous storage) does not
        # support iter_chunks, so writing from such a source must not call it.
        source_fp = h5py.File("source.h5", "w", driver="core", backing_store=False)
        source_data = numpy.arange(24, dtype=numpy.float32).reshape(4, 6)
        source_ds = source_fp.create_dataset("data", data=source_data, chunks=None)
        self.assertIsNone(source_ds.chunks)
        try:
            dest_fp = h5py.File("dest.h5", "w", driver="core", backing_store=False)
            try:
                ds = NHDFImportExportDriver._create_dataset(dest_fp, "0", source_ds)
                numpy.testing.assert_array_equal(ds[...], source_data)
            finally:
                dest_fp.close()
        finally:
            source_fp.close()

    def test_create_dataset_from_chunked_h5py_source(self) -> None:
        source_fp = h5py.File("source.h5", "w", driver="core", backing_store=False)
        source_data = numpy.arange(24, dtype=numpy.float32).reshape(4, 6)
        source_ds = source_fp.create_dataset("data", data=source_data, chunks=(2, 3))
        try:
            dest_fp = h5py.File("dest.h5", "w", driver="core", backing_store=False)
            try:
                ds = NHDFImportExportDriver._create_dataset(dest_fp, "0", source_ds)
                numpy.testing.assert_array_equal(ds[...], source_data)
            finally:
                dest_fp.close()
        finally:
            source_fp.close()

    def test_create_dataset_from_numpy_array(self) -> None:
        data = numpy.arange(24, dtype=numpy.float32).reshape(4, 6)
        fp = h5py.File("dest.h5", "w", driver="core", backing_store=False)
        try:
            ds = NHDFImportExportDriver._create_dataset(fp, "0", data)
            numpy.testing.assert_array_equal(ds[...], data)
        finally:
            fp.close()
