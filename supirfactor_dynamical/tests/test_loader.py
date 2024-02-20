import unittest
import anndata as ad
import numpy as np
import numpy.testing as npt
import pandas as pd
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
    TimeDataset,
    MultimodalDataLoader,
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified,
    H5ADDatasetObsStratified
)


class TestTimeDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            X
        )
        self.adata.obs['time'] = T

    def test_stratified_sampling(self):
        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1
        )

        self.assertEqual(len(td), 25)

        self.assertListEqual(
            [len(i) for i in td.strat_idxes],
            [25, 25, 25, 25]
        )

        self.assertListEqual(
            [len(i) for i in td.shuffle_idxes],
            [4] * 25
        )

        npt.assert_equal(
            td.strat_idxes[0],
            np.arange(25)
        )

        npt.assert_equal(
            td.strat_idxes[1],
            np.arange(25, 50)
        )

        npt.assert_equal(
            td.strat_idxes[2],
            np.arange(50, 75)
        )

        npt.assert_equal(
            td.strat_idxes[3],
            np.arange(75, 100)
        )

        self.assertTrue(
            all(
                self.adata.obs['time'].iloc[idx].is_monotonic_increasing
                for idx in td.shuffle_idxes
            )
        )

        with self.assertRaises(AssertionError):
            npt.assert_equal(
                td.shuffle_idxes[0],
                np.array([0, 25, 50, 75])
            )

    def test_bulk_sampling(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            2,
        )

        self.assertEqual(len(td), 50)
        self.assertIsNone(td.strat_idxes)

        try:
            x = self.adata.X.A
        except AttributeError:
            x = self.adata.X

        for i in range(50):

            npt.assert_equal(
                td[i],
                x[i, :]
            )

    def test_stratified_with_dataloader(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1
        )

        dl = DataLoader(
            td,
            batch_size=5
        )

        self.assertEqual(len(td), 25)
        self.assertEqual(len(dl), 5)

        x_dl = [i for i in dl]
        self.assertEqual(len(x_dl), 5)

        td.shuffle()
        x_dl2 = [j for j in dl]

        for k in range(5):
            with self.assertRaises(AssertionError):
                npt.assert_almost_equal(
                    x_dl[k].numpy(), x_dl2[k].numpy(), decimal=4
                )

    def test_bulk_with_dataloader(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            2
        )

        dl = DataLoader(
            td,
            batch_size=10
        )

        self.assertEqual(len(td), 50)
        self.assertEqual(len(dl), 5)

        x_dl = [i for i in dl]
        self.assertEqual(len(x_dl), 5)

        x_dl2 = [j for j in dl]

        for k in range(5):
            npt.assert_almost_equal(
                x_dl[k].numpy(), x_dl2[k].numpy(), decimal=4
            )

    def test_short_sequence(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=2
        )

        self.assertEqual(
            len(td),
            50
        )

        self.assertTrue(
            np.max([
                np.abs(
                    np.diff(x)
                )
                for x in td.shuffle_idxes
            ]) < 50
        )

    def test_get_all_data(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=2
        )

        self.assertEqual(
            td.get_data_time().shape,
            (100, 4)
        )

        self.assertEqual(
            td.get_data_time(0, 1).shape,
            (25, 4)
        )

        self.assertEqual(
            td.get_data_time(3, 5).shape,
            (25, 4)
        )

    def test_stratified_aggregation(self):
        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1
        )

        agg = td.get_aggregated_times()

        self.assertEqual(
            agg.shape,
            (4, 4)
        )

        _agg_comp = pd.DataFrame(X)
        _agg_comp['Time'] = T

        npt.assert_almost_equal(
            _agg_comp.groupby('Time').agg('mean').values,
            agg
        )

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=2
        )

        agg = td.get_aggregated_times()

        self.assertEqual(
            len(agg),
            3
        )

        _agg_comp = pd.DataFrame(X)
        _agg_comp['Time'] = T
        _agg_comp = _agg_comp.groupby('Time').agg('mean').values

        for i in range(len(agg)):
            print(agg[i])
            npt.assert_almost_equal(
                agg[i],
                _agg_comp[i:i + 2]
            )

    def test_stratified_inorder(self):
        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1
        )

        ordergen = td.get_times_in_order()

        for i, a in enumerate(ordergen):
            self.assertEqual(
                a.shape,
                (25, 4, 4)
            )

        self.assertEqual(i, 0)

    def test_multiple_sequence_lengths(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=[1, 2, 3],
            random_seed=1
        )

        self.assertEqual(
            td.sequence_length,
            np.random.default_rng(1).choice([1, 2, 3])
        )

        s = set()

        for _ in range(20):
            td.shuffle()

            s.add(td.sequence_length)

            self.assertEqual(
                td.sequence_length,
                len(td.shuffle_idxes[0])
            )

        self.assertEqual(
            len(s),
            3
        )

    def test_time_randomize(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            shuffle_time_vector=[0, 2],
            random_seed=1
        )

        npt.assert_equal(
            self.adata.obs['time'].values,
            td._base_time_vector
        )

        _last_iteration = self.adata.obs['time'].values
        _bincount = np.bincount(_last_iteration)

        for i in range(10):
            with self.assertRaises(AssertionError):

                npt.assert_equal(
                    _last_iteration,
                    td.time_vector
                )

            npt.assert_equal(
                _bincount,
                np.bincount(td.time_vector)
            )

            _last_iteration = td.time_vector
            td.shuffle()

    def test_time_seq_randomize(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            shuffle_time_vector=[0, 2],
            sequence_length=[2, 3],
            random_seed=1
        )

        npt.assert_equal(
            self.adata.obs['time'].values,
            td._base_time_vector
        )

        _last_iteration = self.adata.obs['time'].values
        _bincount = np.bincount(_last_iteration)

        for i in range(10):
            with self.assertRaises(AssertionError):

                npt.assert_equal(
                    _last_iteration,
                    td.time_vector
                )

            npt.assert_equal(
                _bincount,
                np.bincount(td.time_vector)
            )

            self.assertGreater(len(td), 0)

            dl = DataLoader(td, 2, drop_last=True)

            for data in dl:

                self.assertEqual(data.shape[0], 2)
                self.assertEqual(data.shape[1], td.sequence_length)
                self.assertEqual(data.shape[2], 4)

            _last_iteration = td.time_vector
            td.shuffle()

    def test_time_seq_randomize_bad(self):

        with self.assertRaises(ValueError):
            TimeDataset(
                self.adata.X,
                self.adata.obs['time'],
                0,
                4,
                t_step=1,
                shuffle_time_vector=[0, 2],
                sequence_length=10,
                random_seed=1
            )

    def test_time_wrap(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=3,
            random_seed=1,
            wrap_times=True
        )

        self.assertFalse(
            all(
                self.adata.obs['time'].iloc[idx].is_monotonic_increasing
                for idx in td.shuffle_idxes
            )
        )

    def test_time_wrap_big(self):

        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            sequence_length=10,
            random_seed=1,
            wrap_times=True
        )

        self.assertFalse(
            all(
                self.adata.obs['time'].iloc[idx].is_monotonic_increasing
                for idx in td.shuffle_idxes
            )
        )

        self.assertEqual(
            len(td.shuffle_idxes[0]),
            10
        )

    def test_return_times_too(self):
        td = TimeDataset(
            self.adata.X,
            self.adata.obs['time'],
            0,
            4,
            t_step=1,
            return_times=True
        )

        dl = MultimodalDataLoader(td, batch_size=10)

        t = next(iter(dl))

        self.assertEqual(len(t), 2)
        self.assertTrue(isinstance(t, tuple))
        self.assertTrue(torch.is_tensor(t[0]))
        self.assertTrue(torch.is_tensor(t[1]))


class TestTimeDatasetSparse(TestTimeDataset):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T


class TestADBacked(unittest.TestCase):

    dataset = None

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
            shuffle=True
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
            shuffle=False
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
            shuffle=True
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


class TestADBackedSparse(TestADBacked):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)


class TestADBackedChunk(unittest.TestCase):

    dataset = None

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
            batch_size=10
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
            batch_size=50
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
            batch_size=10
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


class TestADBackedChunkSparse(TestADBackedChunk):

    def setUp(self) -> None:
        self.adata = ad.AnnData(
            sps.csr_matrix(X)
        )
        self.adata.obs['time'] = T
        self.adata.layers['tst'] = self.adata.X.copy()
        self.adata.write(self.filename)


class TestADBackedStratified(unittest.TestCase):

    dataset = None

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
            dataset.stratification_grouping
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

        dataloader = DataLoader(
            dataset,
            batch_size=10
        )

        lens = [d.shape[0] for d in dataloader]

        self.assertEqual(
            len(lens),
            6
        )
        self.assertListEqual(
            lens,
            [10] * 6
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
            batch_size=50
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


class TestADObsBacked(unittest.TestCase):

    dataset = None
    obs_dataset = None

    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.filename = os.path.join(cls.tempdir.name, "tests.h5ad")

    def setUp(self) -> None:
        x = X.copy()
        x[:, 0] = np.arange(x.shape[0])
        self.adata = ad.AnnData(
            x
        )
        self.adata.obs['time'] = T.astype(str)
        self.adata.obs['strat'] = np.tile(["A", "B", "C"], 34)[0:100]
        self.adata.write(self.filename)

    def tearDown(self) -> None:
        if self.dataset is not None:
            self.dataset.close()

        if self.obs_dataset is not None:
            self.obs_dataset.close()

        return super().tearDown()

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
        self.obs_dataset = obs_dataset

        self.assertEqual(
            obs_dataset._data_reference.shape,
            (100, 7)
        )

        obs_dataset._data_reference[:, 0] = torch.LongTensor(
            np.arange(100)
        )

        for i, j in zip(DataLoader(dataset), DataLoader(obs_dataset)):
            self.assertEqual(
                i[0][0], j[0][0]
            )
