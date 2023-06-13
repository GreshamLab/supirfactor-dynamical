import unittest

import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader
from scipy.linalg import pinv

from supirfactor_dynamical import (
    TFRNNDecoder,
    TimeDataset
)

from ._stubs import (
    X,
    X_tensor,
    A,
    T
)

TEST_SHORT = torch.rand((3, 2, 4))
TEST_MEDIUM = torch.rand((3, 20, 4))
TEST_LONG = torch.rand((3, 50, 4))


class TestTFRecurrentDecoder(unittest.TestCase):

    weight_stack = 1
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

        self._reset_dynae_intermediate()

        self.time_data = TimeDataset(
            X,
            T,
            0,
            3,
            t_step=1
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )

        self.expect_mask = A.T == 0

    def _reset_dynae_intermediate(self):

        # Make the hidden layer a passthrough
        # By setting I for input weights and zeros for recurrent weights
        self.dyn_ae._intermediate.weight_ih_l0 = torch.nn.parameter.Parameter(
            torch.vstack(
                [
                    torch.eye(3, dtype=torch.float32)
                    for _ in range(self.weight_stack)
                ]
            )
        )

        self.dyn_ae._intermediate.weight_hh_l0 = torch.nn.parameter.Parameter(
            torch.zeros((3 * self.weight_stack, 3), dtype=torch.float32)
        )

    def test_data_slice_offset(self):

        self.dyn_ae.set_time_parameters(
            output_t_plus_one=True
        )

        self.assertEqual(
            self.dyn_ae.input_data(TEST_SHORT).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.dyn_ae.output_data(TEST_SHORT).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.dyn_ae.input_data(TEST_MEDIUM).shape,
            (3, 19, 4)
        )

        self.assertEqual(
            self.dyn_ae.output_data(TEST_MEDIUM).shape,
            (3, 19, 4)
        )

    def test_data_slice_offset_plusone(self):

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=1
        )

        with self.assertRaises(ValueError):
            self.dyn_ae.input_data(TEST_SHORT).shape

        with self.assertRaises(ValueError):
            self.dyn_ae.output_data(TEST_SHORT).shape

        self.assertEqual(
            self.dyn_ae.input_data(TEST_MEDIUM).shape,
            (3, 18, 4)
        )

        self.assertEqual(
            self.dyn_ae.output_data(TEST_MEDIUM).shape,
            (3, 19, 4)
        )

    def test_data_slice_offset_noleak(self):

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=10,
            loss_offset=10
        )

        self.assertEqual(
            self.dyn_ae.input_data(TEST_MEDIUM).shape,
            (3, 9, 4)
        )

        self.assertEqual(
            self.dyn_ae.output_data(TEST_MEDIUM).shape,
            (3, 9, 4)
        )

    def test_data_slice_offset_big(self):

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=25,
            loss_offset=20
        )

        with self.assertRaises(ValueError):
            self.dyn_ae.input_data(TEST_MEDIUM).shape

        with self.assertRaises(ValueError):
            self.dyn_ae.output_data(TEST_MEDIUM).shape

        self.assertEqual(
            self.dyn_ae.input_data(TEST_LONG).shape,
            (3, 24, 4)
        )

        self.assertEqual(
            self.dyn_ae(
                self.dyn_ae.input_data(TEST_LONG),
                n_time_steps=25
            ).shape,
            (3, 49, 4)
        )

        self.assertEqual(
            self.dyn_ae._slice_data_and_forward(TEST_LONG).shape,
            (3, 29, 4)
        )

        self.assertEqual(
            self.dyn_ae.output_data(TEST_LONG).shape,
            (3, 29, 4)
        )

    def test_erv_offset_big(self):

        dl = DataLoader(
            TEST_LONG,
            batch_size=1
        )

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=25,
            loss_offset=20
        )

        self.dyn_ae.train_model(dl, epochs=10)
        self.dyn_ae.eval()

        self.assertEqual(
            self.dyn_ae.latent_layer(dl).shape,
            (3, 50, 3)
        )

        self.assertEqual(
            self.dyn_ae.latent_layer(
                self.dyn_ae._slice_data_and_forward(TEST_LONG)
            ).shape,
            (3, 29, 3)
        )

        self.assertEqual(
            self.dyn_ae.latent_layer(
                self.dyn_ae.output_data(TEST_LONG, no_loss_offset=True)
            ).shape,
            (3, 49, 3)
        )

        _ = self.dyn_ae.erv(dl)

    def test_predict(self):

        self.dyn_ae.set_time_parameters(
            output_t_plus_one=True
        )

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            self.time_dataloader,
            n_time_steps=10
        )

        self.assertEqual(
            predictions.shape,
            (25, 13, 4)
        )

    def test_predict_loss_offset(self):

        self.time_data = TimeDataset(
            X,
            T,
            0,
            4,
            t_step=1
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            self.time_dataloader,
            n_time_steps=10
        )

        self.assertEqual(
            predictions.shape,
            (25, 14, 4)
        )

    def test_predict_two_loss_offset(self):

        self.time_data = TimeDataset(
            X,
            T,
            0,
            4,
            t_step=1
        )

        self.time_dataloader = DataLoader(
            self.time_data,
            batch_size=1
        )

        self.dyn_ae.set_time_parameters(
            n_additional_predictions=2,
            loss_offset=1
        )

        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            self.time_dataloader,
            n_time_steps=10
        )

        self.assertEqual(
            predictions.shape,
            (25, 14, 4)
        )

    def test_predict_tensor(self):

        self.dyn_ae.output_t_plus_one = True
        losses, vlosses = self.dyn_ae.train_model(
            self.time_dataloader,
            20
        )

        predictions = self.dyn_ae.predict(
            torch.unsqueeze(X_tensor[0:25, :], 1),
            n_time_steps=20
        )

        self.assertEqual(
            predictions.shape,
            (25, 21, 4)
        )

        predictions = self.dyn_ae.predict(
            X_tensor[[0, 25, 50, 75], :],
            20
        )

        self.assertEqual(
            predictions.shape,
            (24, 4)
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

        self._reset_dynae_intermediate()
        self.dyn_ae.eval()

        with torch.no_grad():
            in_weights = self.dyn_ae.encoder_weights.numpy()
            out_weights = self.dyn_ae.output_weights()

        self.time_data.sequence_length = 2
        self.time_data.shuffle_idxes = self.time_data._get_shuffle_indexes(
            with_replacement=self.time_data.with_replacement,
            n=self.time_data.n
        )

        X_loader = DataLoader(self.time_data, batch_size=25)
        X_time = list(X_loader)[0]

        h = X_time.numpy() @ in_weights.T
        h[h < 0] = 0

        self.dyn_ae(X_time)

        npt.assert_almost_equal(
            self.dyn_ae.latent_layer(X_time).numpy(),
            h,
            decimal=3
        )

        try:
            hiddens = self.dyn_ae.hidden_final.detach().numpy()[0, :, :]
        except AttributeError:
            hiddens = self.dyn_ae.hidden_final[0].detach().numpy()[0, :, :]

        npt.assert_almost_equal(
            hiddens,
            h[:, 1, :],
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

        self.dyn_ae.output_t_plus_one = True
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

        self.dyn_ae.output_t_plus_one = True

        base_data = torch.tensor(
            np.vstack([
                np.mean(X[T == 0, :], axis=0),
                np.mean(X[T == 1, :], axis=0)
            ]),
            dtype=torch.float32
        )

        print(base_data.shape)

        self.dyn_ae.eval()

        with torch.no_grad():
            x_forward = self.dyn_ae(base_data).numpy()

        expect_predicts = np.repeat(x_forward[1, :].reshape(1, -1), 11, axis=0)

        predicts = self.dyn_ae.forward(
            base_data,
            n_time_steps=10
        ).detach().numpy()

        npt.assert_almost_equal(
            expect_predicts,
            predicts[1:, :],
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

    @unittest.SkipTest
    def test_r2_over_timemodel(self):
        pass

    @unittest.SkipTest
    def test_predict(self):
        pass
