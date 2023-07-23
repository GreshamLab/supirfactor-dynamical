import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    get_model
)

from supirfactor_dynamical.train import (
    model_training
)

from supirfactor_dynamical.models.biophysical_model import (
    SupirFactorBiophysical
)

from ._stubs import (
    A,
    T,
    XV_tensor,
    XTV_tensor
)


class TestDynamicalModel(unittest.TestCase):

    decay_model = None

    def setUp(self) -> None:
        super().setUp()

        self.count_data = DataLoader(
            TimeDataset(
                XV_tensor[..., 0],
                T,
                0,
                4,
                1,
                sequence_length=3
            ),
            batch_size=25
        )

        self.velocity_data = DataLoader(
            TimeDataset(
                XV_tensor,
                T,
                0,
                4,
                1,
                sequence_length=3
            ),
            batch_size=25
        )

        self.ordered_data = torch.stack(
            (
                torch.Tensor(
                    (1 + np.arange(0, 20) / 10).reshape(
                        2, 10
                    ).T.reshape(
                        1, 10, 2
                    )
                ),
                torch.full((1, 10, 2), 0.1)
            ),
            dim=-1
        )

    def test_construction(self):

        self.count_model = get_model('rnn')(
            A
        )

        self.count_model.train_model(self.count_data, 50)

        self.dynamical_model = SupirFactorBiophysical(
            A,
            trained_count_model=self.count_model,
            decay_model=self.decay_model
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        with torch.no_grad():
            for data in self.velocity_data:

                predicts = self.dynamical_model(data[..., 0])
                predict_pos, predict_neg = self.dynamical_model(
                    data[..., 0],
                    return_submodels=True
                )

                if self.decay_model is None:
                    self.assertGreaterEqual(predict_pos.min(), 0)
                    self.assertGreaterEqual(0, predict_neg.max())

                    npt.assert_almost_equal(
                        (predict_pos + predict_neg).numpy(),
                        predicts.numpy()
                    )
                else:
                    self.assertIsNone(predict_neg)
                    npt.assert_almost_equal(
                        predict_pos.numpy(),
                        predicts.numpy()
                    )

    def test_training_offset(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model
        )

        self.dynamical_model.set_time_parameters(
            output_t_plus_one=True
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)
        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is None:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )
        else:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_predict(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model
        )

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is None:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )
        else:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_scale(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model
        )

        self.dynamical_model.set_scaling(
            velocity_scaling=np.ones(4),
            count_scaling=np.ones(4)
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is None:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )
        else:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_constant_decay_predict(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model,
            time_dependent_decay=False
        )

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is None:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )
        else:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_constant_decay(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model,
            time_dependent_decay=False
        )

        self.dynamical_model.set_scaling(
            velocity_scaling=np.ones(4),
            count_scaling=np.ones(4)
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is None:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )
        else:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_train_loop(self):

        bioph_obj, res, erv = model_training(
            self.velocity_data,
            pd.DataFrame(A),
            50,
            validation_dataloader=self.velocity_data,
            decay_model=self.decay_model,
            prediction_length=1,
            prediction_loss_offset=1,
            model_type='biophysical',
            return_erv=True
        )

    def test_erv_passthrough(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model
        )

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        def _test_erv(x, output_data_loader):

            for _x, _y in zip(x, output_data_loader):
                self.assertEqual(_x.shape, _y.shape)

        self.dynamical_model._transcription_model.erv = _test_erv
        _ = self.dynamical_model.erv(self.velocity_data)

    def test_forward_counts(self):

        dynamical_model = SupirFactorBiophysical(
            np.ones((2, 1), dtype=np.float32) / 10,
            use_prior_weights=True,
            transcription_model=get_model('static'),
            input_dropout_rate=0.0,
            decay_model=self.decay_model
        )

        dynamical_model._transcription_model.decoder[0].weight = torch.nn.Parameter(
            torch.ones_like(dynamical_model._transcription_model.decoder[0].weight) / 6
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                dynamical_model.input_data(self.ordered_data).numpy(),
                dynamical_model.counts(
                    dynamical_model.input_data(self.ordered_data)
                ).numpy()
            )

        self.assertEqual(
            dynamical_model.counts(
                    dynamical_model.input_data(self.ordered_data[:, [0], ...]),
                    n_time_steps=9
            ).shape,
            (1, 10, 2)
        )

    def test_forward_steps(self):

        class BioTestClass(SupirFactorBiophysical):

            def forward_model(self, x, **kwargs):

                return torch.full_like(x, 0.1)

        dynamical_model = BioTestClass(
            np.ones((2, 1), dtype=np.float32) / 10,
            use_prior_weights=True,
            transcription_model=get_model('static'),
            input_dropout_rate=0.0,
            decay_model=self.decay_model
        )

        with torch.no_grad():
            v = dynamical_model(
                self.ordered_data[..., 0]
            )

            npt.assert_almost_equal(
                v.numpy(),
                self.ordered_data[..., 1].numpy()
            )

            c = dynamical_model(
                self.ordered_data[:, [0], :, 0],
                return_counts=True,
                n_time_steps=9
            )

            npt.assert_almost_equal(
                c.numpy(),
                self.ordered_data[..., 0].numpy(),
                decimal=6
            )


class TestDynamicalModelNoDecay(TestDynamicalModel):

    decay_model = False
