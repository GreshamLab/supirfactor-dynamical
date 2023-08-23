import unittest
import torch
import numpy.testing as npt
import numpy as np

from supirfactor_dynamical._utils import (
    _calculate_erv,
    _calculate_rss,
    _calculate_tss,
    _calculate_r2,
    _aggregate_r2
)

from supirfactor_dynamical._utils._dropout import ConsistentDropout

from scipy.linalg import pinv

from ._stubs import (
    X,
    A,
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
            X_tensor
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


class TestDropout(unittest.TestCase):

    def test_repeated_dropouts(self):

        torch.manual_seed(10)
        t = torch.Tensor(np.arange(100))
        do = ConsistentDropout(0.5)

        self.assertIsNone(do._dropout)

        t_1 = do(t)

        self.assertIsNotNone(do._dropout)

        for _ in range(20):
            t_2 = do(t)
            torch.testing.assert_close(
                t_1,
                t_2
            )

        do.reset()

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                t_1,
                do(t)
            )
