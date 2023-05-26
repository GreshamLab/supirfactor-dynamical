import unittest

import anndata as ad
import numpy as np
import numpy.testing as npt
import pandas as pd

from torch.utils.data import DataLoader

from ._stubs import (
    X,
    T
)

from supirfactor_dynamical import TimeDataset


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

        for i in range(50):
            npt.assert_equal(
                td[i],
                self.adata.X[i, :]
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
