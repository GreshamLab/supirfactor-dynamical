import unittest

import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader
from scipy.linalg import pinv

from supirfactor_dynamical import (
    TFRecurrentAutoencoder,
    TFRNNDecoder,
    TimeDataset
)

from ._stubs import (
    X,
    X_tensor,
    A,
    T
)


class TestTFRecurrentAutoencoder(unittest.TestCase):

    weight_stack = 1
    class_holder = TFRecurrentAutoencoder

    def setUp(self) -> None:

        torch.manual_seed(55)
        self.dyn_ae = self.class_holder(
            A,
            use_prior_weights=True,
            recurrency_mask=torch.zeros(
                (3 * self.weight_stack, 3)
            )
        )

        self.dyn_ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A.T), dtype=torch.float32)
        )

        self.time_data = TimeDataset(
            X,
            T,
            0,
            2,
            t_step=1
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )

        self.expect_mask = A.T == 0
        self.expect_mask = np.vstack(
            [self.expect_mask for _ in range(self.weight_stack)]
        )

    def test_initialize(self):

        self.assertEqual(
            self.dyn_ae.k,
            3
        )

        self.assertEqual(
            self.dyn_ae.g,
            4
        )

    def test_initialize_hidden_weights(self):

        with torch.no_grad():
            npt.assert_almost_equal(
                np.vstack(
                    [np.zeros((3, 3)) for _ in range(self.weight_stack)]
                ),
                self.dyn_ae.recurrent_weights.numpy()
            )

        diagonal_only = self.class_holder(
            A,
            use_prior_weights=True
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                np.vstack(
                    [np.eye(3, dtype=bool) for _ in range(self.weight_stack)]
                ),
                diagonal_only.recurrent_weights.numpy() != 0
            )

        no_mask = self.class_holder(
            A,
            use_prior_weights=True,
            recurrency_mask=False
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                np.vstack(
                    [
                        np.ones((3, 3), dtype=bool)
                        for _ in range(self.weight_stack)
                    ]
                ),
                no_mask.recurrent_weights.numpy() != 0
            )

    def test_latent_layer(self):

        X_tensor = list(self.time_dataloader)[0]

        print(X_tensor.shape)

        self.dyn_ae.train()
        ll_train = self.dyn_ae.latent_layer(X_tensor)

        self.dyn_ae.eval()
        ll_eval = self.dyn_ae.latent_layer(X_tensor)

        npt.assert_almost_equal(
            ll_eval.numpy(),
            ll_train.numpy()
        )

        npt.assert_almost_equal(
            ll_eval.numpy(),
            X_tensor.numpy() @ A
        )

        print(ll_eval.numpy().shape)

    def test_train_loop(self):

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            10
        )

        self.assertEqual(len(losses), 10)
        self.assertIsNone(vlosses)

        with torch.no_grad():
            in_weights = self.dyn_ae.encoder_weights.numpy()
            out_weights = self.dyn_ae.output_weights()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, self.expect_mask)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_train_loop_with_validation(self):

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            10,
            validation_dataloader=DataLoader(
                self.time_data,
                batch_size=1
            )
        )

        self.assertEqual(len(losses), 10)
        self.assertEqual(len(vlosses), 10)

        self.dyn_ae.eval()
        with torch.no_grad():
            in_weights = self.dyn_ae.encoder_weights.numpy()
            out_weights = self.dyn_ae.output_weights()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, self.expect_mask)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_erv(self):

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        with torch.no_grad():
            in_weights = self.dyn_ae.encoder_weights.numpy()
            out_weights = self.dyn_ae.output_weights()

        X_loader = DataLoader(self.time_data, batch_size=25)
        X_time = list(X_loader)[0]

        h = X_time.numpy() @ in_weights.T
        h[h < 0] = 0

        npt.assert_almost_equal(
            self.dyn_ae.latent_layer(X_time).numpy(),
            h,
            decimal=3
        )

        X_time = X_time.numpy()

        y = h @ out_weights.T
        y[y < 0] = 0

        yrss = (X_time - y) ** 2
        yrss = yrss.sum(axis=(0, 1))

        erv, rss, full_rss = self.dyn_ae.erv(X_loader, return_rss=True)

        npt.assert_almost_equal(
            full_rss,
            yrss,
            decimal=1
        )

        for i in range(3):
            h_partial = h.copy()
            h_partial[:, :, i] = 0

            y_partial = h_partial @ out_weights.T
            y_partial[y_partial < 0] = 0

            y_partial_rss = (X_time - y_partial) ** 2
            y_partial_rss = y_partial_rss.sum(axis=(0, 1))

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

    def test_r2_model(self):

        train_r2, val_r2 = self.dyn_ae.r2(
            self.time_dataloader,
            self.time_dataloader
        )

        npt.assert_almost_equal(
            train_r2,
            0.75
        )

        npt.assert_almost_equal(
            val_r2,
            0.75
        )

    def test_train_loop_offset(self):

        self.dyn_ae.prediction_offset = 1
        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            10
        )

        self.assertEqual(len(losses), 10)
        self.assertIsNone(vlosses)

        with torch.no_grad():
            in_weights = self.dyn_ae.encoder_weights.numpy()
            out_weights = self.dyn_ae.output_weights()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, self.expect_mask)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_train_loop_offset_predict(self):

        self.dyn_ae.prediction_offset = 1
        self.dyn_ae.L = 2

        base_data = torch.tensor(
            np.vstack([
                np.mean(X[T == 0, :], axis=0),
                np.mean(X[T == 1, :], axis=0)
            ]),
            dtype=torch.float32
        )

        self.dyn_ae.eval()

        with torch.no_grad():
            x_forward = self.dyn_ae(base_data).numpy()

        expect_predicts = np.repeat(x_forward[1, :].reshape(1, -1), 10, axis=0)

        predicts = self.dyn_ae.predict(
            base_data,
            10
        )

        npt.assert_almost_equal(
            expect_predicts,
            predicts,
            decimal=4
        )

    def test_r2_over_timemodel(self):

        train_r2, val_r2 = self.dyn_ae.r2_over_time(
            self.time_dataloader,
            self.time_dataloader
        )

        self.assertEqual(
            len(train_r2),
            1
        )

        npt.assert_almost_equal(
            train_r2[0],
            0.75
        )

        self.assertEqual(
            len(val_r2),
            1
        )

        npt.assert_almost_equal(
            val_r2[0],
            0.75
        )

    def test_r2_over_timemodel_len2(self):

        time_data = TimeDataset(
            X,
            T,
            0,
            4,
            t_step=1,
            sequence_length=2
        )

        time_dataloader = DataLoader(
            time_data,
            batch_size=1
        )

        train_r2, val_r2 = self.dyn_ae.r2_over_time(
            time_dataloader,
            time_dataloader
        )

        self.assertEqual(
            len(train_r2),
            3
        )

        npt.assert_almost_equal(
            train_r2[0],
            0.75
        )

        self.assertEqual(
            len(val_r2),
            3
        )

        npt.assert_almost_equal(
            val_r2[0],
            0.75
        )


class TestTFRecurrentDecoder(TestTFRecurrentAutoencoder):

    class_holder = TFRNNDecoder

    def setUp(self) -> None:

        torch.manual_seed(55)
        self.dyn_ae = self.class_holder(
            A,
            use_prior_weights=True
        )

        self.dyn_ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A.T), dtype=torch.float32)
        )

        self.time_data = TimeDataset(
            X,
            T,
            0,
            2,
            t_step=1
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )

        self.expect_mask = A.T == 0

    def test_predict(self):

        self.dyn_ae.prediction_offset = 1
        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            self.time_dataloader,
            10
        )

        self.assertEqual(
            predictions.shape,
            (25, 10, 4)
        )

    def test_predict_tensor(self):

        self.dyn_ae.prediction_offset = 1
        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            torch.unsqueeze(X_tensor[0:25, :], 1),
            20
        )

        self.assertEqual(
            predictions.shape,
            (25, 20, 4)
        )

        predictions = self.dyn_ae.predict(
            X_tensor[0:, :],
            20
        )

        self.assertEqual(
            predictions.shape,
            (20, 4)
        )

    @unittest.SkipTest
    def test_initialize_hidden_weights(self):
        pass

    @unittest.SkipTest
    def test_train_loop_offset_predict():
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel():
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel_len2():
        pass

    @unittest.SkipTest
    def test_r2_model():
        pass

    @unittest.SkipTest
    def test_erv():
        pass

    @unittest.SkipTest
    def test_latent_layer():
        pass


class TestTFRecurrentDecoderShuffler(TestTFRecurrentDecoder):

    def setUp(self) -> None:
        super().setUp()

        self.time_data = TimeDataset(
            X,
            T,
            0,
            3,
            t_step=1,
            shuffle_time_vector=[0, 2],
            sequence_length=[2, 3]
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )
