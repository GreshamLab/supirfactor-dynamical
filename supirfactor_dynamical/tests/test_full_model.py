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

from supirfactor_dynamical.training import (
    model_training
)

from supirfactor_dynamical.models import (
    SupirFactorBiophysical,
    DecayModule
)

from supirfactor_dynamical._utils import _get_data_offsets

from ._stubs import (
    A,
    T,
    XV_tensor,
    XTV_tensor,
    XVD_tensor,
    XTVD_tensor
)

temp = tempfile.TemporaryDirectory(prefix='pytest')
temp_name = temp.name
temp_file_name = os.path.join(temp.name, "static.h5")


class TestDynamicalModel(unittest.TestCase):

    decay_model = None
    decay_delay = None
    decay_k = 20

    optimize_too = False

    def setUp(self) -> None:
        super().setUp()

        self.count_data = DataLoader(
            TimeDataset(
                XV_tensor[..., 0],
                T,
                0,
                3,
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
            decay_epoch_delay=self.decay_delay,
            decay_k=self.decay_k,
            separately_optimize_decay_model=self.optimize_too
        )

        if self.dynamical_model._decay_model is not None:
            _opt = self.dynamical_model._decay_model.process_optimizer(None)
        else:
            _opt = None

        self.opt = (
            self.dynamical_model.process_optimizer(None),
            self.dynamical_model._transcription_model.process_optimizer(None),
            _opt
        )

    def test_construction(self):

        self.dynamical_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay_model
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        with torch.no_grad():
            for data in self.velocity_data:

                predicts = self.dynamical_model(data[..., 0])[0]
                predict_pos, predict_neg = self.dynamical_model(
                    data[..., 0],
                    return_submodels=True
                )[0]

                if self.decay_model is False:
                    self.assertIsNone(
                        predict_neg
                    )
                else:
                    self.assertGreaterEqual(predict_pos.min(), 0)
                    self.assertGreaterEqual(0, predict_neg.max())

                    torch.testing.assert_close(
                        (predict_pos + predict_neg),
                        predicts
                    )

    def test_training_offset(self):

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])[0]
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)
        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )[0]

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is False:
            torch.testing.assert_close(
                xp.detach(),
                x.detach()
            )
        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            torch.testing.assert_close(
                xn.detach() + xp.detach(),
                x.detach()
            )

    def test_predict_wrapper(self):

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        predicts = self.dynamical_model.predict(
            self.count_data,
            return_submodels=True,
            n_time_steps=2
        )

        self.assertEqual(
            predicts[0][0].shape,
            (25, 5, 4)
        )

        if self.dynamical_model._decay_model is not None:
            self.assertEqual(
                predicts[0][1].shape,
                (25, 5, 4)
            )

        self.assertEqual(
            predicts[1].shape,
            (25, 5, 4)
        )

        if self.dynamical_model._decay_model is not None:
            self.assertEqual(
                predicts[2].shape,
                (25, 5, 4)
            )

    def test_training_predict(self):

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])[0]
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )[0]

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is False:
            torch.testing.assert_close(
                xp.detach(),
                x.detach()
            )
        else:
            self.assertTrue(np.all(xp.detach().numpy() >= 0))
            self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)
            self.assertTrue(np.all(xn.detach().numpy() <= 0))

            torch.testing.assert_close(
                xn.detach() + xp.detach(),
                x.detach()
            )

    def test_training_scale(self):

        self.dynamical_model.set_scaling(
            velocity_scaling=np.ones(4),
            count_scaling=np.ones(4)
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])[0]
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(
            XTV_tensor[..., 0],
            return_submodels=True
        )[0]

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)

        if self.decay_model is False:
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
                self.assertEqual(
                    _x[..., 0].shape,
                    _y.shape
                )

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

        def forward_model(x, **kwargs):
            return torch.full_like(x, 0.1), None, None

        dynamical_model.forward_model = forward_model

        with torch.no_grad():
            npt.assert_almost_equal(
                dynamical_model.input_data(self.ordered_data).numpy() + 0.1,
                dynamical_model(
                    dynamical_model.input_data(self.ordered_data)
                )[1].numpy()
            )

        self.assertEqual(
            dynamical_model(
                dynamical_model.input_data(self.ordered_data[:, [0], ...]),
                n_time_steps=9
            )[1].shape,
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
            return torch.full_like(x, 0.1), None, None

        dynamical_model.forward_model = forward_model

        with torch.no_grad():
            v = dynamical_model(
                self.ordered_data[..., 0]
            )[0]

            npt.assert_almost_equal(
                v.numpy(),
                self.ordered_data[..., 1].numpy()
            )

            c = dynamical_model(
                self.ordered_data[:, [0], :, 0],
                n_time_steps=9
            )[1]

            c_expect = self.ordered_data[..., 0].numpy()
            c_expect += self.ordered_data[..., 1].numpy()

            npt.assert_almost_equal(
                c.numpy(),
                c_expect,
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
            return torch.full_like(x, 1), None, None

        dynamical_model.forward_model = forward_model

        with torch.no_grad():
            v = dynamical_model(
                self.ordered_data[..., 0]
            )[0]

            npt.assert_almost_equal(
                v.numpy() * 0.1,
                self.ordered_data[..., 1].numpy()
            )

            c = dynamical_model(
                self.ordered_data[:, [0], :, 0],
                n_time_steps=9
            )[1]

            c_expect = self.ordered_data[..., 0].numpy()
            c_expect += self.ordered_data[..., 1].numpy()

            npt.assert_almost_equal(
                c.numpy(),
                c_expect,
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

        L = testy.shape[1]

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
            _get_data_offsets(
                L,
                loss_offset=self.dynamical_model.loss_offset
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

    def test_loss(self):

        self.dynamical_model.eval()

        v = self.dynamical_model.output_data(XTVD_tensor)
        v_bar = self.dynamical_model(
            self.dynamical_model.input_data(XTVD_tensor)
        )[0]
        v_mse = torch.nn.MSELoss()(
            v,
            v_bar
        ).item()

        self.assertAlmostEqual(
            v_mse,
            torch.nn.MSELoss()(
                self.dynamical_model(
                    self.dynamical_model.input_data(XTVD_tensor)
                )[0],
                self.dynamical_model.output_data(XTVD_tensor)
            ).item(),
            places=5
        )

    def test_joint_loss(self):

        self.dynamical_model.eval()

        v = self.dynamical_model.output_data(XTVD_tensor)
        v_bar = self.dynamical_model(
            self.dynamical_model.input_data(XTVD_tensor)
        )[0]
        v_mse = torch.nn.MSELoss()(
            v,
            v_bar
        ).item()

        loss = self.dynamical_model._training_step(
            0,
            XTVD_tensor,
            self.opt,
            torch.nn.MSELoss()
        )

        npt.assert_almost_equal(
            loss[0],
            v_mse,
            decimal=5
        )

    def test_joint_loss_offsets(self):

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.eval()

        v = self.dynamical_model.output_data(XTVD_tensor)
        v_bar = self.dynamical_model.output_data(
            self.dynamical_model(
                self.dynamical_model.input_data(XTVD_tensor),
                n_time_steps=1
            )[0],
            keep_all_dims=True,
            offset_only=True
        )
        v_mse = torch.nn.MSELoss()(
            v,
            v_bar
        ).item()

        loss = self.dynamical_model._training_step(
            0,
            XTVD_tensor,
            self.opt,
            torch.nn.MSELoss()
        )

        npt.assert_almost_equal(
            loss[0],
            v_mse,
            decimal=1
        )

    def test_loss_df(self):

        self.dynamical_model.train_model(
            self.velocity_data,
            10,
            self.velocity_data
        )

        self.assertEqual(
            self.dynamical_model.training_loss_df.shape,
            (2, 11)
        )

        self.assertEqual(
            self.dynamical_model.training_loss_df.iloc[:, 0].tolist(),
            self.dynamical_model._loss_type_names
        )

        self.assertEqual(
            self.dynamical_model.validation_loss_df.shape,
            (2, 11)
        )

        self.assertEqual(
            self.dynamical_model.validation_loss_df.iloc[:, 0].tolist(),
            self.dynamical_model._loss_type_names
        )

    def test_optimizer(self):

        def _optimizer_correct(
            num_epoch,
            train_x,
            optimizer,
            loss_function,
            target_x=None
        ):

            self.assertEqual(
                len(optimizer[0].param_groups[0]['params']),
                4
            )

            if self.dynamical_model._decay_model is None:
                self.assertFalse(optimizer[1])
            else:
                self.assertEqual(
                    len(optimizer[1].param_groups[0]['params']),
                    5
                )

            return 1

        self.dynamical_model._training_step = _optimizer_correct
        self.dynamical_model.train_model(self.velocity_data, 10)


class TestDynamicalModelNoDecay(TestDynamicalModel):

    decay_model = False


class TestDynamicalModelBigDecay(TestDynamicalModel):

    decay_k = 50


class TestDynamicalModelTuneDecay(TestDynamicalModel):

    def setUp(self) -> None:
        # Pretrain a decay module a bit for tuned testing later
        self.decay_model = DecayModule(4, 2)

        self.decay_model.train_model(
            [XTVD_tensor],
            10
        )

        super().setUp()

    @unittest.skip
    def test_joint_loss_offsets():
        pass


class TestDynamicalModelTuneDecayDelay(TestDynamicalModelTuneDecay):

    decay_delay = 5


class TestDynamicalModelOptimizeDecay(TestDynamicalModelTuneDecay):

    optimize_too = True

    def setUp(self) -> None:
        super().setUp()
        self.velocity_data = DataLoader(
            TimeDataset(
                XVD_tensor,
                T,
                0,
                4,
                1,
                sequence_length=3
            ),
            batch_size=25
        )


class TestDynamicalModelOptimizeDecayDelay(TestDynamicalModelOptimizeDecay):

    decay_delay = 5
