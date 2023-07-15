import unittest

import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    get_model
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

    def test_construction(self):

        self.count_model = get_model('rnn')(
            A
        )

        self.count_model.train_model(self.count_data, 50)

        self.dynamical_model = SupirFactorBiophysical(
            A,
            trained_count_model=self.count_model

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

                self.assertGreaterEqual(predict_pos.min(), 0)
                self.assertGreaterEqual(0, predict_neg.max())

                npt.assert_almost_equal(
                    (predict_pos + predict_neg).numpy(),
                    predicts.numpy()
                )

    def test_training_offset(self):

        self.dynamical_model = SupirFactorBiophysical(
            A
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
        self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)

        self.assertTrue(np.all(xn.detach().numpy() <= 0))
        self.assertTrue(np.all(xp.detach().numpy() >= 0))

        npt.assert_almost_equal(
            xn.detach().numpy() + xp.detach().numpy(),
            x.detach().numpy()
        )

    def test_training_predict(self):

        self.dynamical_model = SupirFactorBiophysical(
            A
        )

        self.dynamical_model.set_time_parameters(
            n_additional_predictions=1,
            loss_offset=1
        )

        self.dynamical_model.train_model(self.velocity_data, 50)
        self.dynamical_model.eval()

        x = self.dynamical_model(XTV_tensor[..., 0])
        self.assertEqual(x.shape, XTV_tensor[..., 0].shape)

        (xp, xn) = self.dynamical_model(XTV_tensor[..., 0], return_submodels=True)

        self.assertEqual(xp.shape, XTV_tensor[..., 0].shape)
        self.assertEqual(xn.shape, XTV_tensor[..., 0].shape)

        self.assertTrue(np.all(xn.detach().numpy() <= 0))
        self.assertTrue(np.all(xp.detach().numpy() >= 0))

        npt.assert_almost_equal(
            xn.detach().numpy() + xp.detach().numpy(),
            x.detach().numpy()
        )
