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
    decay_weight = None

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
            pd.DataFrame(A),
            decay_model=self.decay_model,
            joint_optimize_decay_model=self.optimize_decay_too,
            decay_loss_weight=self.decay_weight
        )

        dm = self.dynamical_model._decay_model
        if dm:
            dm.optimizer = dm.process_optimizer(None)

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
        x = self.dynamical_model.output_data(XTVD_tensor, counts=True)
        x_bar = self.dynamical_model(
            self.dynamical_model.input_data(XTVD_tensor),
            return_counts=True
        )
        x_mse = torch.nn.MSELoss()(
            x,
            x_bar
        ).item()

        v = self.dynamical_model.output_data(XTVD_tensor)
        v_bar = self.dynamical_model(
            self.dynamical_model.input_data(XTVD_tensor)
        )
        v_mse = torch.nn.MSELoss()(
            v,
            v_bar
        ).item()

        loss = self.dynamical_model._training_step(
            XTVD_tensor,
            opt,
            torch.nn.MSELoss()
        )

        npt.assert_almost_equal(
            loss[0],
            v_mse,
            decimal=5
        )

        npt.assert_almost_equal(
            loss[1],
            x_mse,
            decimal=5
        )

    def test_loss_df(self):

        self.dynamical_model.joint_optimize_decay_model = True
        self.dynamical_model.train_model(self.velocity_data, 10)

        self.assertEqual(
            self.dynamical_model.training_loss_df.shape,
            (3, 11)
        )

        self.assertEqual(
            self.dynamical_model.training_loss_df.iloc[:, 0].tolist(),
            self.dynamical_model._loss_type_names
        )

    def test_optimizer(self):

        if self.optimize_decay_too:
            _correct_n = 4
        elif self.dynamical_model._decay_model is not None:
            _correct_n = 9
        else:
            _correct_n = 4

        def _optimizer_correct(
            train_x,
            optimizer,
            loss_function
        ):
            self.assertEqual(
                len(optimizer.param_groups[0]['params']),
                _correct_n
            )

            return (1, 1, 1)

        self.dynamical_model._training_step = _optimizer_correct
        self.dynamical_model.train_model(self.velocity_data, 10)

    @unittest.skip
    def test_perturbation_prediction(self):

        self.dynamical_model.train_model(self.velocity_data, 10)

        for d in self.velocity_data:
            break

        predicts = self.dynamical_model.predict_perturbation(
            d[..., [0]],
            n_time_steps=5,
            perturbation=1
        )

        self.assertEqual(
            predicts.shape,
            (25, 8, 4)
        )

        with torch.no_grad():
            bad_predicts = self.dynamical_model(
                d[..., [0]],
                x_decay=torch.zeros(25, 8, 4),
                n_time_steps=5,
                return_submodels=True
            )

        torch.testing.assert_close(
            bad_predicts[1],
            torch.zeros_like(bad_predicts[1])
        )

        self.assertEqual(
            bad_predicts[1].shape,
            (25, 8, 4)
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


class TestDynamicalModelJointDecayScale(TestDynamicalModelJointDecay):

    decay_weight = 10
