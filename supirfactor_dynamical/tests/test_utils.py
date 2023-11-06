import unittest
import torch
import numpy.testing as npt
import numpy as np
import pandas as pd
import anndata as ad

from supirfactor_dynamical._utils import (
    _calculate_erv,
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _aggregate_r2,
    _process_weights_to_tensor
)

from scipy.linalg import pinv

from ._stubs import (
    X,
    X_SP,
    A,
    X_tensor
)


class TestTensorUtils(unittest.TestCase):

    def test_array_to_tensor(self):

        x_t, (a, b) = _process_weights_to_tensor(
            X
        )

        torch.testing.assert_close(x_t, torch.transpose(X_tensor, 0, 1))
        self.assertIsNone(a)
        self.assertIsNone(b)

        x_t, (a, b) = _process_weights_to_tensor(
            X,
            transpose=False
        )

        torch.testing.assert_close(x_t, X_tensor)
        self.assertIsNone(a)
        self.assertIsNone(b)

    def test_dataframe_to_tensor(self):

        x = pd.DataFrame(X)

        x_t, _ = _process_weights_to_tensor(
            x
        )

        torch.testing.assert_close(x_t, torch.transpose(X_tensor, 0, 1))

        x_t, _ = _process_weights_to_tensor(
            x,
            transpose=False
        )

        torch.testing.assert_close(x_t, X_tensor)

    def test_sparse_to_tensor(self):

        x_t, (a, b) = _process_weights_to_tensor(
            X_SP
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            torch.transpose(X_tensor, 0, 1)
        )
        self.assertIsNone(a)
        self.assertIsNone(b)

        x_t, (a, b) = _process_weights_to_tensor(
            X_SP,
            transpose=False
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            X_tensor
        )
        self.assertIsNone(a)
        self.assertIsNone(b)

    def adata_to_tensor(self):

        adata = ad.AnnData(X)

        x_t, _ = _process_weights_to_tensor(
            adata
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            torch.transpose(X_tensor, 0, 1)
        )

        x_t, _ = _process_weights_to_tensor(
            adata,
            transpose=False
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            X_tensor
        )

    def adata_sparse_to_tensor(self):

        adata = ad.AnnData(X_SP)

        x_t, _ = _process_weights_to_tensor(
            adata
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            torch.transpose(X_tensor, 0, 1)
        )

        x_t, _ = _process_weights_to_tensor(
            adata,
            transpose=False
        )

        torch.testing.assert_close(
            x_t.to_dense(),
            X_tensor
        )


class TestMathUtils(unittest.TestCase):

    def test_erv(self):

        a_inv = pinv(A)

        latent = X_tensor @ A
        full = (X_tensor - latent @ a_inv) ** 2
        full = full.sum(axis=0)

        rss_manual = torch.zeros(4, 3)

        for i in range(3):
            latent_drop = torch.clone(latent)
            latent_drop[:, i] = 0
            rss_manual[:, i] = (X_tensor - latent_drop @ a_inv ** 2).sum(
                axis=0
            )

        erv = _calculate_erv(full, rss_manual)

        erv_expect = torch.ones(erv.shape)
        erv_expect[3, :] = 0.3823919

        npt.assert_almost_equal(
            erv_expect,
            erv
        )

    def test_rss(self):

        a_inv = pinv(A)

        npt.assert_almost_equal(
            np.sum(
                (X - X @ A @ a_inv) ** 2,
                axis=0
            ),
            _calculate_rss(
                X_tensor,
                X_tensor @ A @ a_inv
            ),
            decimal=5
        )

    def test_r2_ybar(self):

        a_inv = pinv(A)

        rss = _calculate_rss(
            X_tensor,
            X_tensor @ A @ a_inv
        )

        tss = _calculate_tss(
            X_tensor,
            ybar=True
        )

        r2 = _calculate_r2(rss, tss)

        _bad_r2 = ((X[:, 3]) ** 2).sum()
        _bad_r2 /= ((X[:, 3].mean() - X[:, 3]) ** 2).sum()

        npt.assert_almost_equal(
            [1., 1., 1., 1 - _bad_r2],
            r2,
            decimal=5
        )

        npt.assert_almost_equal(
            0.75,
            _aggregate_r2(r2)
        )

    def test_r2(self):

        a_inv = pinv(A)

        rss = _calculate_rss(
            X_tensor,
            X_tensor @ A @ a_inv
        )

        tss = _calculate_tss(
            X_tensor,
            ybar=False
        )

        r2 = _calculate_r2(rss, tss)

        npt.assert_almost_equal(
            [1., 1., 1., 0],
            r2,
            decimal=5
        )

        npt.assert_almost_equal(
            0.75,
            _aggregate_r2(r2)
        )
