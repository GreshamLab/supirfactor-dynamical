import unittest

import numpy as np
import numpy.testing as npt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from scipy.linalg import pinv

from supirfactor_dynamical.models.ae_model import Autoencoder
from supirfactor_dynamical import (
    TFAutoencoder,
    TFMetaAutoencoder,
    TimeDataset,
    TFMultilayerAutoencoder
)

from ._stubs import (
    X,
    X_tensor,
    A,
    T
)

TEST_SHORT = torch.rand((3, 2, 4))
TEST_MEDIUM = torch.rand((3, 10, 4))
TEST_LONG = torch.rand((3, 50, 4))


class TestTFAutoencoderNoPrior(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = TFAutoencoder(A.shape)

    def test_initialize(self):

        self.assertEqual(
            self.ae.k,
            3
        )

        self.assertEqual(
            self.ae.g,
            4
        )


class TestTFAutoencoder(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = TFAutoencoder(A, use_prior_weights=True)
        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
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

        self.ae.train_model(
            loader,
            10
        )

        self.assertEqual(len(self.ae.training_loss), 10)
        self.assertEqual(len(self.ae.validation_loss), 0)

        with torch.no_grad():
            in_weights = self.ae.encoder_weights.numpy()
            out_weights = self.ae.decoder_weights.numpy()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, A.T == 0)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_forward_tfa(self):

        data = next(iter(DataLoader(
            X_tensor,
            batch_size=2
        )))

        p, tfa = self.ae(
            data,
            return_tfa=True
        )

        torch.testing.assert_close(
            tfa,
            self.ae.drop_encoder(p)
        )

    def test_train_loop_with_validation(self):

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        vloader = DataLoader(
            X_tensor,
            batch_size=2
        )

        self.ae.train_model(
            loader,
            10,
            validation_dataloader=vloader
        )

        self.assertEqual(len(self.ae.training_loss), 10)
        self.assertEqual(len(self.ae.validation_loss), 10)

        with torch.no_grad():
            in_weights = self.ae.encoder_weights.numpy()
            out_weights = self.ae.decoder_weights.numpy()

        expected_nz = np.ones_like(out_weights, dtype=bool)
        expected_nz[3, :] = False

        npt.assert_equal(in_weights == 0, A.T == 0)
        npt.assert_equal(out_weights != 0, expected_nz)

    def test_erv(self):

        def _forward_from_latent(x):
            with torch.no_grad():
                if isinstance(self.ae.intermediate_weights, list):
                    for w in self.ae.intermediate_weights:
                        x = x @ w.numpy().T
                        x[x < 0] = 0
                elif self.ae.intermediate_weights is not None:
                    x = x @ self.ae.intermediate_weights.numpy().T
                    x[x < 0] = 0

                if hasattr(self.ae, "_decoder") and len(self.ae._decoder) > 3:
                    x = x @ self.ae._decoder[0].weight.numpy().T
                    x[x < 0] = 0
                    x = x @ self.ae._decoder[3].weight.numpy().T
                    x[x < 0] = 0
                else:
                    x = x @ self.ae.decoder_weights.numpy().T
                    x[x < 0] = 0

            return x

        loader = DataLoader(
            X_tensor,
            batch_size=2
        )

        self.ae.train_model(
            loader,
            20
        )

        self.ae.eval()
        with torch.no_grad():
            in_weights = self.ae.encoder_weights.numpy()

            h = X @ in_weights.T
            h[h < 0] = 0

            npt.assert_almost_equal(
                self.ae.latent_layer(X_tensor).numpy(),
                h,
                decimal=3
            )

            y = _forward_from_latent(h)

        yrss = (X - y) ** 2
        yrss = yrss.sum(axis=0)

        erv, rss, full_rss = self.ae.erv(loader, return_rss=True)

        npt.assert_almost_equal(
            full_rss,
            yrss,
            decimal=1
        )

        for i in range(3):
            h_partial = self.ae.latent_layer(X_tensor).numpy()
            h_partial[:, i] = 0

            y_partial = _forward_from_latent(h_partial)

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

    def test_data_slice_offset(self):

        self.ae.set_time_parameters(
            output_t_plus_one=True
        )

        self.assertEqual(
            self.ae.input_data(TEST_SHORT).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_SHORT).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.input_data(TEST_MEDIUM).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_MEDIUM).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.input_data(TEST_LONG).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_LONG).shape,
            (3, 1, 4)
        )

    def test_data_slice_offset_plusone(self):

        self.ae.set_time_parameters(
            n_additional_predictions=1,
            output_t_plus_one=True
        )

        with self.assertRaises(ValueError):
            self.ae.output_data(TEST_SHORT).shape

        self.assertEqual(
            self.ae.input_data(TEST_MEDIUM).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_MEDIUM).shape,
            (3, 2, 4)
        )

        self.assertEqual(
            self.ae.input_data(TEST_LONG).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_LONG).shape,
            (3, 2, 4)
        )

    def test_data_slice_offset_big(self):

        self.ae.set_time_parameters(
            n_additional_predictions=25,
            loss_offset=20,
            output_t_plus_one=True
        )

        with self.assertRaises(ValueError):
            self.ae.output_data(TEST_MEDIUM).shape

        self.assertEqual(
            self.ae.input_data(TEST_LONG).shape,
            (3, 1, 4)
        )

        self.assertEqual(
            self.ae(
                self.ae.input_data(TEST_LONG),
                n_time_steps=25
            ).shape,
            (3, 26, 4)
        )

        self.assertEqual(
            self.ae._slice_data_and_forward(TEST_LONG).shape,
            (3, 6, 4)
        )

        self.assertEqual(
            self.ae.output_data(TEST_LONG).shape,
            (3, 6, 4)
        )


class TestTFDropouts(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)

    def test_no_dropout(self):
        self.ae = TFAutoencoder(A, use_prior_weights=True)
        self.ae.set_drop_tfs(None)

        npt.assert_almost_equal(
            (X_tensor @ A).numpy(),
            self.ae.latent_layer(X_tensor).numpy()
        )

    def test_no_labels(self):
        self.ae = TFAutoencoder(A, use_prior_weights=True)

        with self.assertRaises(RuntimeError):
            self.ae.set_drop_tfs("BAD")

    def test_good_dropouts(self):
        self.ae = TFAutoencoder(
            pd.DataFrame(A),
            use_prior_weights=True
        )

        self.ae.set_drop_tfs(None)

        npt.assert_almost_equal(
            (X_tensor @ A).numpy(),
            self.ae.latent_layer(X_tensor).numpy()
        )

        self.ae.set_drop_tfs(0)

        ll = (X_tensor @ A).numpy()
        ll[:, 0] = 0.
        npt.assert_almost_equal(
            ll,
            self.ae.latent_layer(X_tensor).numpy()
        )

        self.ae.set_drop_tfs([0, 1])

        ll = (X_tensor @ A).numpy()
        ll[:, [0, 1]] = 0.
        npt.assert_almost_equal(
            ll,
            self.ae.latent_layer(X_tensor).numpy()
        )

    def test_missing_dropout_warnings(self):

        self.ae = TFAutoencoder(
            pd.DataFrame(A),
            use_prior_weights=True
        )

        self.ae.set_drop_tfs(None)

        with self.assertWarns(RuntimeWarning):
            self.ae.set_drop_tfs("A", raise_error=False)

        with self.assertWarns(RuntimeWarning):
            self.ae.set_drop_tfs("C", raise_error=False)

        with self.assertWarns(RuntimeWarning):
            self.ae.set_drop_tfs(["A", "c"], raise_error=False)

        with self.assertWarns(RuntimeWarning):
            self.ae.set_drop_tfs(["A", 0], raise_error=False)

        with self.assertWarns(RuntimeWarning):
            self.ae.set_drop_tfs(["A", 0, 1, 2], raise_error=False)

        self.ae.set_drop_tfs([0, 1, 2])


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
                4,
                t_step=1
            ),
            batch_size=5
        )

        self.ae = TFAutoencoder(A, use_prior_weights=True)
        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

    def test_offset_zero(self):

        self.ae.set_time_parameters(
            n_additional_predictions=0,
            loss_offset=0
        )

        self.ae.train_model(
            self.static_dataloader,
            20
        )

        _ = self.ae.erv(self.static_dataloader)

    def test_offset_one(self):

        self.ae.set_time_parameters(
            output_t_plus_one=True,
            n_additional_predictions=0,
            loss_offset=0
        )

        self.ae.train_model(
            self.dynamic_dataloader,
            20
        )
        _ = self.ae.erv(self.dynamic_dataloader)

    def test_offset_long(self):

        self.ae.set_time_parameters(
            output_t_plus_one=True,
            n_additional_predictions=2,
            loss_offset=0
        )

        self.ae.train_model(
            self.dynamic_dataloader,
            20
        )

        _ = self.ae.erv(self.dynamic_dataloader)

    def test_predict(self):

        self.ae.set_time_parameters(
            output_t_plus_one=True,
            n_additional_predictions=0,
            loss_offset=0
        )

        self.ae.train_model(
            self.dynamic_dataloader,
            20
        )

        predictions = self.ae.predict(
            self.static_dataloader,
            n_time_steps=10
        )

        self.assertEqual(
            predictions.shape,
            (100, 11, 4)
        )

        predictions = self.ae.predict(
            DataLoader(
                X_tensor,
                batch_size=5
            ),
            n_time_steps=20
        )

        self.assertEqual(
            predictions.shape,
            (100, 21, 4)
        )

        predictions = self.ae.predict(
            X_tensor[0:25, :],
            n_time_steps=20
        )

        self.assertEqual(
            predictions.shape,
            (25, 21, 4)
        )

        predictions = self.ae.predict(
            X_tensor[0, :],
            n_time_steps=20
        )

        self.assertEqual(
            predictions.shape,
            (1, 21, 4)
        )


class TestTFMetaAutoencoder(TestTFAutoencoder):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = TFMetaAutoencoder(A, use_prior_weights=True)
        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )
        self.ae._intermediate[0].weight = torch.nn.parameter.Parameter(
            torch.eye(3, dtype=torch.float32)
        )


class TestTFMultilayerAutoencoder(TestTFAutoencoder):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = TFMultilayerAutoencoder(
            prior_network=A,
            use_prior_weights=True,
            intermediate_dropout_rate=0.0,
            intermediate_sizes=(3, 3, 3),
            decoder_sizes=(3, )
        )
        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.eye(3, dtype=torch.float32)
        )
        self.ae._decoder[3].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        for i in range(3):
            self.ae._intermediate[3*i].weight = torch.nn.parameter.Parameter(
                torch.eye(3, dtype=torch.float32)
            )
        print(self.ae)


class TestAutoencoder(TestTFAutoencoder):

    def setUp(self) -> None:
        torch.manual_seed(55)
        self.ae = Autoencoder(
            n_genes=A.shape[0],
            n_hidden_layers=2,
            hidden_layer_width=A.shape[1]
        )
        self.ae.encoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(A.T, dtype=torch.float32)
        )
        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )
        self.ae._intermediate[0].weight = torch.nn.parameter.Parameter(
            torch.eye(3, dtype=torch.float32)
        )
        self.ae._intermediate[0].weight.requires_grad = False

    def test_module_construction(self):

        self.assertEqual(
            len(self.ae.encoder),
            2
        )

        self.assertEqual(
            len(self.ae._intermediate),
            3
        )

        self.assertEqual(
            len(self.ae._decoder),
            2
        )

    @unittest.skip
    def test_train_loop(self):
        pass

    @unittest.skip
    def test_train_loop_with_validation(self):
        pass


class TestTFMetaAutoencoderOffset(TestTFAutoencoderOffset):

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
                4,
                t_step=1
            ),
            batch_size=5
        )

        self.ae = TFMetaAutoencoder(
            A,
            use_prior_weights=True
        )

        self.ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )
        self.ae._intermediate[0].weight = torch.nn.parameter.Parameter(
            torch.eye(3, dtype=torch.float32)
        )


class TestTFAutoencoderCUDA(TestTFAutoencoder):

    def setUp(self) -> None:
        super().setUp()
        self.ae.device = 'cuda' if torch.cuda.is_available() else 'cpu'
