import unittest

import numpy as np

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    SupirFactorBiophysical
)

from supirfactor_dynamical.models.decay_model import (
    DecayModule,
    DecayModuleSimple
)

from ._stubs import (
    A,
    T,
    XV_tensor
)


class TestDecayModule(unittest.TestCase):

    expected_decay_size = (25, 3, 4)

    def setUp(self) -> None:

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

        self.decay = DecayModule(4, 2)

        return super().setUp()

    def test_training(self):

        for d in self.velocity_data:

            self.decay.train_model(
                [torch.stack((
                    d[..., 0],
                    torch.nn.ReLU()(d[..., 1] * -1) * -1
                    ),
                    dim=-1
                )],
                10
            )

            self.decay.eval()

            with torch.no_grad():

                dout, dc_out = self.decay(
                    d[..., 0],
                    return_decay_constants=True
                )

                dout = dout.numpy()
                dc_out = dc_out.numpy()

                self.assertEqual(
                    dout.shape,
                    (25, 3, 4)
                )

                self.assertEqual(
                    dc_out.shape,
                    self.expected_decay_size
                )

                self.assertTrue(
                    np.all(dout <= 0)
                )

                self.assertTrue(
                    np.all(dc_out <= 0)
                )

        self.assertEqual(len(self.decay.training_loss), 10)

    def test_training_in_context(self):

        full_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay
        )

        full_model.train_model(self.velocity_data, 20)

    def test_training_in_context_predict(self):

        full_model = SupirFactorBiophysical(
            A,
            decay_model=self.decay
        )

        full_model.set_time_parameters(
            n_additional_predictions=1
        )

        full_model.train_model(self.velocity_data, 20)


class TestDecaySimple(TestDecayModule):

    expected_decay_size = (4, )

    def setUp(self):
        super().setUp()

        self.decay = DecayModuleSimple(
            np.full(4, 0.1)
        )
