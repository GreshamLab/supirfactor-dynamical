import unittest

import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader
from scipy.linalg import pinv

from supirfactor_dynamical import TFAutoencoder, TimeDataset

from ._stubs import (
    X,
    X_tensor,
    A,
    T
)


class TestTFAutoencoder(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = TFAutoencoder(A, use_prior_weights=True)
        self.ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

    def test_initialize(self):

        self.assertEqual(
            self.ae.k,
            3
        )

        self.assertEqual(
            self.ae.g,
            4
        )

    def test_latent_layer(self):

        self.ae.train()
        ll_train = self.ae.latent_layer(X_tensor)

        self.ae.eval()
        ll_eval = self.ae.latent_layer(X_tensor)

        npt.assert_almost_equal(
            ll_eval.numpy(),
            ll_train.numpy()
        )

        npt.assert_almost_equal(
            ll_eval.numpy(),
            X @ A
        )

    def test_model_call(self):

        self.ae.eval()
        Y_hat = self.ae(X_tensor)

        npt.assert_almost_equal(
            Y_hat.detach().numpy(),
            X @ A @ pinv(A),
            decimal=4
        )

        Y_expect = X.copy()
        Y_expect[:, 3] = 0.

        npt.assert_almost_equal(
            Y_hat.detach().numpy(),
            Y_expect,
            decimal=4
        )

    def test_model_call_dataloader(self):

        self.ae.eval()

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        for data in loader:
            Y_hat = self.ae(data)

            npt.assert_almost_equal(
                Y_hat.detach().numpy(),
                data.numpy() @ A @ pinv(A),
                decimal=4
            )

            Y_expect = data.numpy()
            Y_expect[:, 3] = 0.

            npt.assert_almost_equal(
                Y_hat.detach().numpy(),
                Y_expect,
                decimal=4
            )

    def test_train_loop(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        losses, vlosses = self.ae.train_model(
            loader,
            10
        )

        self.assertEqual(len(losses), 10)
        self.assertIsNone(vlosses)

        with torch.no_grad():
            in_weights = self.ae.encoder[0].weight.numpy()
            out_weights = self.ae.decoder[0].weight.numpy()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, A.T == 0)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_train_loop_with_validation(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        vloader = DataLoader(
            X_tensor,
            batch_size=2
        )

        losses, vlosses = self.ae.train_model(
            loader,
            10,
            validation_dataloader=vloader
        )

        self.assertEqual(len(losses), 10)
        self.assertEqual(len(vlosses), 10)

        with torch.no_grad():
            in_weights = self.ae.encoder[0].weight.numpy()
            out_weights = self.ae.decoder[0].weight.numpy()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, A.T == 0)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_erv(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        losses, vlosses = self.ae.train_model(
            loader,
            20
        )

        self.ae.eval()
        with torch.no_grad():
            in_weights = self.ae.encoder[0].weight.numpy()
            out_weights = self.ae.decoder[0].weight.numpy()

        h = X @ in_weights.T
        h[h < 0] = 0

        npt.assert_almost_equal(
            self.ae.latent_layer(X_tensor).numpy(),
            h,
            decimal=3
        )

        y = h @ out_weights.T
        y[y < 0] = 0

        yrss = (X - y) ** 2
        yrss = yrss.sum(axis=0)

        erv, rss, full_rss = self.ae.erv(loader, return_rss=True)

        npt.assert_almost_equal(
            full_rss,
            yrss,
            decimal=1
        )

        for i in range(3):
            h_partial = h.copy()
            h_partial[:, i] = 0

            y_partial = h_partial @ out_weights.T
            y_partial[y_partial < 0] = 0

            y_partial_rss = (X - y_partial) ** 2
            y_partial_rss = y_partial_rss.sum(axis=0)

            npt.assert_almost_equal(
                rss[:, i],
                y_partial_rss,
                decimal=1
            )

            with self.assertRaises(AssertionError):
                npt.assert_almost_equal(
                    yrss,
                    y_partial_rss,
                    decimal=1
                )

            erv_expect = 1 - yrss / y_partial_rss

            npt.assert_almost_equal(
                erv[:, i],
                erv_expect,
                decimal=1
            )

    def test_weights(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        losses, vlosses = self.ae.train_model(
            loader,
            20
        )

        erv = self.ae.erv(loader, return_rss=False)

        with torch.no_grad():
            out_weights = self.ae.decoder[0].weight.numpy()

        masked_weights = self.ae.pruned_model_weights(data_loader=loader)
        out_weights[erv <= 0] = 0.

        self.assertEqual(
            masked_weights.shape,
            A.shape
        )

        npt.assert_almost_equal(
            masked_weights,
            out_weights
        )

    def test_r2(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        self.ae.eval()

        r2 = self.ae._calculate_r2_score(
            loader
        )

        npt.assert_equal(
            r2.numpy() == 1.,
            np.array([True, True, True, False])
        )

    def test_r2_model(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        train_r2, val_r2 = self.ae.r2(
            loader,
            loader
        )

        npt.assert_almost_equal(
            train_r2,
            0.75
        )

        npt.assert_almost_equal(
            val_r2,
            0.75
        )


class TestTFAutoencoderOffset(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)

        self.static_dataloader = DataLoader(
            X_tensor,
            batch_size=1
        )

        self.dynamic_dataloader = DataLoader(
            TimeDataset(
                X,
                T,
                0,
                2,
                t_step=1
            ),
            batch_size=5
        )

        self.ae = TFAutoencoder(A, use_prior_weights=True, prediction_offset=0)
        self.ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

    def test_offset_zero(self):

        losses, vlosses = self.ae.train_model(
            self.static_dataloader,
            20
        )

        _ = self.ae.erv(self.static_dataloader)

    def test_offset_one(self):

        self.ae.prediction_offset = 1
        losses, vlosses = self.ae.train_model(
            self.dynamic_dataloader,
            20
        )

        _ = self.ae.erv(self.dynamic_dataloader)

    def test_predict(self):

        self.ae.prediction_offset = 1
        losses, vlosses = self.ae.train_model(
            self.dynamic_dataloader,
            20
        )

        predictions = self.ae.predict(self.static_dataloader, 10)

        self.assertEqual(
            predictions.shape,
            (100, 10, 4)
        )

        predictions = self.ae.predict(
            DataLoader(
                X_tensor,
                batch_size=5
            ),
            20
        )

        self.assertEqual(
            predictions.shape,
            (20, 5, 20, 4)
        )

        predictions = self.ae.predict(
            X_tensor[0:25, :],
            20
        )

        self.assertEqual(
            predictions.shape,
            (25, 20, 4)
        )

        predictions = self.ae.predict(
            X_tensor[0, :],
            20
        )

        self.assertEqual(
            predictions.shape,
            (20, 4)
        )
