import unittest
import tempfile
import os

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

from supirfactor_dynamical.models import (
    SupirFactorBiophysical,
    DecayModule
)

from ._stubs import (
    A,
    T,
    XV_tensor,
    XTV_tensor,
    XTVD_tensor
)

temp = tempfile.TemporaryDirectory(prefix='pytest')
temp_name = temp.name
temp_file_name = os.path.join(temp.name, "static.h5")


class TestDynamicalModel(unittest.TestCase):

    decay_model = None
    optimize_decay_too = False
    decay_weight=None

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

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model,
            joint_optimize_decay_model=self.optimize_decay_too,
            decay_loss_weight=self.decay_weight
        )

        if self.dynamical_model._decay_model is not None:
            self.dynamical_model._decay_model.optimizer = self.dynamical_model._decay_model.process_optimizer(None)

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

                if self.decay_model is False:
                    self.assertIsNone(predict_neg)
                    npt.assert_almost_equal(
                        predict_pos.numpy(),
                        predicts.numpy()
                    )
                else:

                    self.assertGreaterEqual(predict_pos.min(), 0)
                    self.assertGreaterEqual(0, predict_neg.max())

                    npt.assert_almost_equal(
                        (predict_pos + predict_neg).numpy(),
                        predicts.numpy()
                    )

    def test_training_offset(self):

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)
        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is False:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_predict(self):

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

        if self.decay_model is False:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
                x.detach().numpy()
            )

    def test_training_scale(self):

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

        if self.decay_model is False:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
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

        if self.decay_model is False:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
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

        if self.decay_model is False:
            self.assertIsNone(xn)
            npt.assert_almost_equal(
                xp.detach().numpy(),
                x.detach().numpy()
            )

        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            npt.assert_almost_equal(
                xn.detach().numpy() + xp.detach().numpy(),
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

        dynamical_model = SupirFactorBiophysical(
            np.ones((2, 1), dtype=np.float32) / 10,
            use_prior_weights=True,
            transcription_model=get_model('static'),
            input_dropout_rate=0.0,
            decay_model=self.decay_model
        )

        def forward_model(x, **kwargs):
            return torch.full_like(x, 0.1)

        dynamical_model.forward_model = forward_model

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

    def test_forward_steps_scaled(self):

        dynamical_model = SupirFactorBiophysical(
            np.ones((2, 1), dtype=np.float32) / 10,
            use_prior_weights=True,
            transcription_model=get_model('static'),
            input_dropout_rate=0.0,
            decay_model=self.decay_model
        )

        dynamical_model.set_scaling(
            count_scaling=[1, 1],
            velocity_scaling=[0.1, 0.1]
        )

        def forward_model(x, **kwargs):
            return torch.full_like(x, 1)

        dynamical_model.forward_model = forward_model

        with torch.no_grad():
            v = dynamical_model(
                self.ordered_data[..., 0]
            )

            npt.assert_almost_equal(
                v.numpy() * 0.1,
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

    def test_input_output(self):

        testy = torch.Tensor(
            np.stack((
                np.arange(100).reshape(5, 10, 2),
                np.arange(100).reshape(5, 10, 2) * -1,
                ),
                -1
            )
        )

        self.assertFalse(self.dynamical_model._offset_data)

        torch.testing.assert_close(
            self.dynamical_model.input_data(testy),
            testy[..., 0]
        )

        torch.testing.assert_close(
            self.dynamical_model.output_data(testy),
            testy[..., 1]
        )

        self.dynamical_model.set_time_parameters(
            loss_offset=5
        )

        self.assertEqual(
            self.dynamical_model._get_data_offsets(
                self.dynamical_model.input_data(testy)
            ),
            (10, 5)
        )

        torch.testing.assert_close(
            self.dynamical_model.input_data(testy),
            testy[..., 0]
        )

        torch.testing.assert_close(
            self.dynamical_model.output_data(testy),
            testy[:, 5:, :, 1]
        )

        self.dynamical_model.set_time_parameters(
            loss_offset=0,
            n_additional_predictions=5
        )

        torch.testing.assert_close(
            self.dynamical_model.input_data(testy),
            testy[:, :5, :, 0]
        )

        torch.testing.assert_close(
            self.dynamical_model.output_data(testy),
            testy[..., 1]
        )

        self.dynamical_model.set_time_parameters(
            loss_offset=3,
            n_additional_predictions=3
        )

        torch.testing.assert_close(
            self.dynamical_model.input_data(testy),
            testy[:, :7, :, 0]
        )

        torch.testing.assert_close(
            self.dynamical_model.output_data(testy),
            testy[:, 3:, :, 1]
        )

    def test_joint_loss(self):

        opt = self.dynamical_model.process_optimizer(None)

        self.dynamical_model.eval()
        x = self.dynamical_model.output_data(XTVD_tensor)
        x_bar = self.dynamical_model(
            self.dynamical_model.input_data(XTVD_tensor)
        )
        x_mse = torch.nn.MSELoss()(
            x,
            x_bar
        ).item()

        loss = self.dynamical_model._training_step(
            XTVD_tensor,
            opt,
            torch.nn.MSELoss()
        )

        if not self.dynamical_model.joint_optimize_decay_model:
            model_loss = loss

        else:
            self.assertEqual(len(loss), 3)
            model_loss = loss[1]

            npt.assert_almost_equal(
                loss[0],
                x_mse + loss[2],
                decimal=5
            )

        npt.assert_almost_equal(
            model_loss,
            x_mse,
            decimal=5
        )

    def test_loss_df(self):

        self.dynamical_model.joint_optimize_decay_model = True
        self.dynamical_model.train_model(self.velocity_data, 10)

        if self.dynamical_model._decay_model is None:
            self.assertEqual(
                self.dynamical_model.training_loss_df.shape,
                (1, 11)
            )

            self.assertEqual(
                self.dynamical_model.training_loss_df.iloc[0, 0],
                self.dynamical_model.type_name
            )

        else:
            self.assertEqual(
                self.dynamical_model.training_loss_df.shape,
                (3, 11)
            )

            self.assertEqual(
                self.dynamical_model.training_loss_df.iloc[:, 0].tolist(),
                self.dynamical_model._loss_type_names
            )


class TestDynamicalModelNoDecay(TestDynamicalModel):

    decay_model = False


class TestDynamicalModelTuneDecay(TestDynamicalModel):

    def setUp(self) -> None:
        # Pretrain a decay module a bit for tuned testing later
        self.decay_model = DecayModule(4, 2)

        self.decay_model.train_model(
            [XTVD_tensor],
            10
        )

        return super().setUp()

    @unittest.skip
    def test_forward_counts():
        pass


class TestDynamicalModelJointDecay(TestDynamicalModel):

    optimize_decay_too = True
    decay_weight = 10