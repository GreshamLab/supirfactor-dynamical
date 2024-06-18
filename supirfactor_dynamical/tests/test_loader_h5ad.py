import unittest
import anndata as ad
import numpy as np
import numpy.testing as npt
import pandas.testing as pdt
import scipy.sparse as sps
import torch
import tempfile
import os

from torch.utils.data import DataLoader

from ._stubs import (
    X,
    T
)

from supirfactor_dynamical.datasets import (
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified,
    H5ADDatasetObsStratified,
    StackIterableDataset
)
from supirfactor_dynamical.datasets.stratified_file_dataset import (
    StratifySingleFileDataset
)

from supirfactor_dynamical.datasets.anndata_backed_dataset import (
    _batched_len,
    _batched_n
)


class TestBatchers(unittest.TestCase):

    def test_batched_len(self):

        for b, t in zip(
            _batched_len(np.arange(100), 25),
            [
                np.arange(25),
                np.arange(25, 50),
                np.arange(50, 75),
                np.arange(75, 100)
            ]
        ):
            npt.assert_equal(b, t)

        for b, t in zip(
            _batched_len(np.arange(27), 25),
            [
                np.arange(25),
                np.arange(25, 27)
            ]
        ):
            npt.assert_equal(b, t)

    def test_batched_n(self):
        for b, t in zip(
            _batched_n(np.arange(100), 2),
            [
                np.arange(50),
                np.arange(50, 100)
            ]
        ):
            npt.assert_equal(b, t)

        for b, t in zip(
            _batched_n(np.arange(27), 2),
            [
                np.arange(14),
                np.arange(14, 27)
            ]
        ):
            npt.assert_equal(b, t)


class TestADBacked(unittest.TestCase):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)

    def tearDown(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

        return super().tearDown()

    def test_load_h5(self):

        dataset = H5ADDataset(self.filename)
        self.dataset = dataset

        self.assertEqual(
            len(dataset),
            100
        )

        dataloader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=self.num_workers
        )

        lens = [d.shape[0] for d in dataloader]

        self.assertEqual(
            len(lens),
            10
        )
        self.assertListEqual(
            lens,
            [10] * 10
        )

    def test_h5_mask(self):

        dataset = H5ADDataset(
            self.filename,
            obs_include_mask=np.arange(50, 100)
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset),
            50
        )

        dataloader = DataLoader(
            dataset,
            batch_size=50,
            shuffle=False,
            num_workers=self.num_workers
        )

        data = [d for d in dataloader]

        self.assertEqual(
            len(data),
            1
        )
        self.assertEqual(
            data[0].shape[0],
            50
        )

        npt.assert_almost_equal(
            X[np.arange(50, 100), :],
            data[0].numpy()
        )

    def test_load_h5_layer(self):

        dataset = H5ADDataset(self.filename, layer='tst')
        self.dataset = dataset

        self.assertEqual(
            len(dataset),
            100
        )

        dataloader = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            num_workers=self.num_workers
        )

        lens = [d.shape[0] for d in dataloader]

        self.assertEqual(
            len(lens),
            10
        )
        self.assertListEqual(
            lens,
            [10] * 10
        )


class TestADBackedOneWorker(TestADBacked):
    num_workers = 1


class TestADBackedTwoWorker(TestADBacked):
    num_workers = 2


class TestADBackedSparse(TestADBacked):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)


class TestADBackedSparseOneWorker(TestADBackedSparse):
    num_workers = 1


class TestADBackedSparseTwoWorker(TestADBackedSparse):
    num_workers = 2


class TestADBackedChunk(unittest.TestCase):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)

    def tearDown(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

        return super().tearDown()

    def test_load_h5(self):

        dataset = H5ADDatasetIterable(
            self.filename,
            file_chunk_size=50
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.file_chunks),
            2
        )

        npt.assert_array_equal(
            dataset.file_chunks[0],
            np.arange(50)
        )

        npt.assert_array_equal(
            dataset.file_chunks[1],
            np.arange(50, 100)
        )

        dataset.load_chunk(0)

        npt.assert_array_equal(
            dataset._data_loaded_chunk.numpy(),
            X[0:50, :]
        )

        dataset.load_chunk(1)

        npt.assert_array_equal(
            dataset._data_loaded_chunk.numpy(),
            X[50:100, :]
        )

        dataloader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=self.num_workers
        )

        lens = [d.shape[0] for d in dataloader]

        self.assertEqual(
            len(lens),
            10
        )
        self.assertListEqual(
            lens,
            [10] * 10
        )

    def test_h5_mask(self):

        dataset = H5ADDatasetIterable(
            self.filename,
            obs_include_mask=np.arange(50, 100),
            file_chunk_size=27
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.file_chunks),
            2
        )

        npt.assert_array_equal(
            dataset.file_chunks[0],
            np.arange(50, 77)
        )

        npt.assert_array_equal(
            dataset.file_chunks[1],
            np.arange(77, 100)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=50,
            num_workers=self.num_workers
        )

        data = [d for d in dataloader]

        self.assertEqual(
            len(data),
            max(1, self.num_workers)
        )

        self.assertEqual(
            torch.cat(data).shape[0],
            50
        )

    def test_load_h5_big_chunk(self):

        dataset = H5ADDatasetIterable(
            self.filename,
            file_chunk_size=5000
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.file_chunks),
            1
        )

        npt.assert_array_equal(
            dataset.file_chunks[0],
            np.arange(100)
        )

        dataset.load_chunk(0)

        npt.assert_array_equal(
            dataset._data_loaded_chunk.numpy(),
            X
        )

        dataloader = DataLoader(
            dataset,
            batch_size=10,
            num_workers=self.num_workers
        )

        lens = [d.shape[0] for d in dataloader]

        self.assertEqual(
            len(lens),
            10
        )
        self.assertListEqual(
            lens,
            [10] * 10
        )


class TestADBackedChunkOneWorker(TestADBackedChunk):
    num_workers = 1


class TestADBackedChunkTwoWorker(TestADBackedChunk):
    num_workers = 2


class TestADBackedChunkSparse(TestADBackedChunk):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)


class TestADBackedSparseChunkOneWorker(TestADBackedChunkSparse):
    num_workers = 1


class TestADBackedSparseChunkTwoWorker(TestADBackedChunkSparse):
    num_workers = 2


class TestADBackedStratified(unittest.TestCase):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X
        )
        self.adata.obs['time'] = T

        _strats = np.tile(["A", "B", "C"], 34)[0:100]
        _strats[[3, 6, 10]] = 'D'
        self.adata.obs['strat'] = _strats
        self.adata.obs['other_strat'] = ['F'] * 49 + ['G'] * 50 + ['F']
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)

    def tearDown(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

        return super().tearDown()

    def test_load_h5(self):

        dataset = H5ADDatasetStratified(
            self.filename,
            'strat',
            file_chunk_size=50
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.file_chunks),
            2
        )

        pdt.assert_series_equal(
            self.adata.obs['strat'],
            dataset.stratification_grouping['strat']
        )

        npt.assert_array_equal(
            dataset.file_chunks[0],
            np.arange(50)
        )

        npt.assert_array_equal(
            dataset.file_chunks[1],
            np.arange(50, 100)
        )

        for c, s in enumerate([slice(0, 50), slice(50, 100)]):
            dataset.load_chunk(c)

            npt.assert_array_equal(
                dataset._data_loaded_chunk.numpy(),
                X[s, :]
            )
            self.assertEqual(
                len(dataset._data_loaded_stratification),
                4
            )
            for i in range(4):
                _idx = np.nonzero(
                    self.adata.obs['strat'].iloc[s].cat.codes == i
                )[0]
                npt.assert_array_equal(
                    _idx,
                    np.intersect1d(
                        _idx,
                        dataset._data_loaded_stratification[i]
                    )
                )

    def test_h5_mask(self):

        dataset = H5ADDatasetStratified(
            self.filename,
            'strat',
            obs_include_mask=np.arange(0, 50),
            file_chunk_size=27
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.file_chunks),
            2
        )

        npt.assert_array_equal(
            dataset.file_chunks[0],
            np.arange(27)
        )

        npt.assert_array_equal(
            dataset.file_chunks[1],
            np.arange(27, 50)
        )

        dataloader = DataLoader(
            dataset,
            batch_size=50,
            num_workers=self.num_workers
        )

        data = [d for d in dataloader]

        self.assertEqual(
            len(data),
            1
        )
        self.assertEqual(
            data[0].shape[0],
            33
        )

    def test_dual_strat(self):

        dataset = H5ADDatasetStratified(
            self.filename,
            ['strat', 'other_strat'],
            discard_categories=['D'],
            file_chunk_size=50
        )

        pdt.assert_series_equal(
            self.adata.obs['strat'],
            dataset.stratification_grouping['strat']
        )

        pdt.assert_series_equal(
            self.adata.obs['other_strat'],
            dataset.stratification_grouping['other_strat']
        )

        dataset.load_chunk(0)

        npt.assert_array_equal(
            dataset._data_loaded_chunk.numpy(),
            X[0:50, :]
        )

        dataset.get_chunk_order(0)

        self.assertEqual(
            len(dataset._data_loaded_stratification),
            4
        )


class TestADObsBacked(unittest.TestCase):

    dataset = None
    obs_dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

        x = X.copy()
        x[:, 0] = np.arange(x.shape[0])
        cls.adata = ad.AnnData(
            x
        )
        cls.adata.obs['time'] = T.astype(str)
        cls.adata.obs['strat'] = np.tile(["A", "B", "C"], 34)[0:100]
        cls.adata.write(cls.filename)

    def tearDown(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

        return super().tearDown()

    def test_obs_one_hot(self):

        obs_dataset = H5ADDatasetObsStratified(
            self.filename,
            'strat',
            file_chunk_size=27,
            obs_columns=['time', 'strat'],
            random_seed=876,
            one_hot=True
        )

        self.assertEqual(
            torch.stack([x for x in obs_dataset]).shape,
            (99, 7)
        )

    def test_obs_classnum_discard(self):

        obs_dataset = H5ADDatasetObsStratified(
            self.filename,
            'strat',
            file_chunk_size=27,
            obs_columns='strat',
            random_seed=876,
            one_hot=False,
            discard_categories=['B']
        )

        self.assertEqual(
            torch.stack([x for x in obs_dataset]).shape,
            (66, 1)
        )

        obs_dataset.format_data(True)

        self.assertEqual(
            torch.stack([x for x in obs_dataset]).shape,
            (66, 3)
        )

        torch.testing.assert_close(
            torch.stack([x for x in obs_dataset]).sum(axis=0),
            torch.Tensor([33, 0, 33])
        )

    def test_obs_classnum(self):

        obs_dataset = H5ADDatasetObsStratified(
            self.filename,
            'strat',
            file_chunk_size=27,
            obs_columns='strat',
            random_seed=876,
            one_hot=False
        )

        self.assertEqual(
            torch.stack([x for x in obs_dataset]).shape,
            (99, 1)
        )

        obs_dataset.format_data(True)

        self.assertEqual(
            torch.stack([x for x in obs_dataset]).shape,
            (99, 3)
        )

    def test_obs_alignment(self):

        dataset = H5ADDatasetStratified(
            self.filename,
            'strat',
            file_chunk_size=27,
            random_seed=876
        )
        self.dataset = dataset

        obs_dataset = H5ADDatasetObsStratified(
            self.filename,
            'strat',
            file_chunk_size=27,
            obs_columns=['time', 'strat'],
            random_seed=876
        )

        self.assertEqual(
            obs_dataset._data_reference.shape,
            (100, 7)
        )

        obs_dataset._data_reference[:, 0] = torch.LongTensor(
            np.arange(100)
        )

        for i, j in zip(
            DataLoader(dataset, num_workers=self.num_workers),
            DataLoader(obs_dataset, num_workers=self.num_workers)
        ):
            self.assertEqual(
                i[0][0], j[0][0]
            )

        for data in DataLoader(
            StackIterableDataset(
                dataset,
                obs_dataset
            ),
            batch_size=2,
            drop_last=True,
            num_workers=self.num_workers
        ):
            self.assertEqual(
                data[0][0][0], data[1][0][0]
            )
            self.assertEqual(
                data[0].shape,
                (2, 4)
            )
            self.assertEqual(
                data[1].shape,
                (2, 7)
            )


class TestMemoryDenseStratified(unittest.TestCase):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X
        )
        self.adata.obs['time'] = T

        _strats = np.tile(["A", "B", "C"], 34)[0:100]
        _strats[[3, 6, 10]] = 'D'
        self.adata.obs['strat'] = _strats
        self.adata.obs['strat'] = self.adata.obs['strat'].astype('category')
        self.adata.obs['other_strat'] = ['F'] * 49 + ['G'] * 50 + ['F']
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)

    def test_load_h5(self):

        dataset = StratifySingleFileDataset(
            self.filename,
            ['strat'],
            yield_obs_cats=['strat']
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.loaded_data),
            2
        )

        self.assertEqual(
            len(dataset.loaded_data[0]),
            100
        )

        self.assertEqual(
            self.adata.obs['strat'].value_counts().reindex(
                ['A', 'B', 'C', 'D']
            ).values.tolist(),
            list(map(len, dataset.stratification_group_indexes))
        )

        _classes = []

        for i, (v, c) in enumerate(dataset):
            self.assertEqual(
                v.shape,
                (4,)
            )
            self.assertEqual(
                c.shape,
                (4,)
            )
            _classes.append(c)

        _classes = np.vstack(_classes).sum(0).astype(int)
        self.assertEqual(i, 11)

        npt.assert_equal(
            _classes,
            [3, 3, 3, 3]
        )

    def test_h5_mask(self):

        dataset = StratifySingleFileDataset(
            self.filename,
            ['strat'],
            yield_obs_cats=['strat'],
            obs_include_mask=np.arange(50)
        )
        self.dataset = dataset

        self.assertEqual(
            len(dataset.loaded_data),
            2
        )

        self.assertEqual(
            len(dataset.loaded_data[0]),
            50
        )

        self.assertEqual(
            [15, 16, 16, 3],
            list(map(len, dataset.stratification_group_indexes))
        )

        _classes = []

        for i, (v, c) in enumerate(dataset):
            self.assertEqual(
                v.shape,
                (4,)
            )
            self.assertEqual(
                c.shape,
                (4,)
            )
            _classes.append(c)

        _classes = np.vstack(_classes).sum(0).astype(int)
        self.assertEqual(i, 11)

        npt.assert_equal(
            _classes,
            [3, 3, 3, 3]
        )

    def test_dual_strat(self):

        dataset = StratifySingleFileDataset(
            self.filename,
            ['strat', 'other_strat'],
            discard_categories=['D'],
            yield_obs_cats=['strat', 'other_strat']
        )

        self.assertEqual(
            len(dataset.loaded_data),
            3
        )

        self.assertEqual(
            len(dataset.loaded_data[0]),
            100
        )

        self.assertEqual(
            [16, 16, 15, 17, 16, 17],
            list(map(len, dataset.stratification_group_indexes))
        )

        self.assertEqual(
            97,
            sum(map(len, dataset.stratification_group_indexes))
        )

        _classes = []

        for i, (v, c, c2) in enumerate(dataset):
            self.assertEqual(
                v.shape,
                (4,)
            )
            self.assertEqual(
                c.shape,
                (4,)
            )
            self.assertEqual(
                c2.shape,
                (2,)
            )
            _classes.append(c)

        _classes = np.vstack(_classes).sum(0).astype(int)

        npt.assert_equal(
            _classes,
            [30, 30, 30, 0]
        )


class TestMemoryCSRStratified(TestMemoryDenseStratified):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T

        _strats = np.tile(["A", "B", "C"], 34)[0:100]
        _strats[[3, 6, 10]] = 'D'
        self.adata.obs['strat'] = _strats
        self.adata.obs['strat'] = self.adata.obs['strat'].astype('category')
        self.adata.obs['other_strat'] = ['F'] * 49 + ['G'] * 50 + ['F']
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)


class TestMemoryCSCStratified(TestMemoryDenseStratified):

    dataset = None
    num_workers = 0

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csc_matrix(X)
        )
        self.adata.obs['time'] = T

        _strats = np.tile(["A", "B", "C"], 34)[0:100]
        _strats[[3, 6, 10]] = 'D'
        self.adata.obs['strat'] = _strats
        self.adata.obs['strat'] = self.adata.obs['strat'].astype('category')
        self.adata.obs['other_strat'] = ['F'] * 49 + ['G'] * 50 + ['F']
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)
