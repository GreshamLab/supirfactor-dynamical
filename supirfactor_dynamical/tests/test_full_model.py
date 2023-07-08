import unittest

import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    get_model
)

from supirfactor_dynamical.models.biophysical_model import (
    SupirFactorDynamical
)

from ._stubs import (
    A,
    T,
    XV_tensor
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

        self.dynamical_model = SupirFactorDynamical(
            self.count_model,
            A
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
