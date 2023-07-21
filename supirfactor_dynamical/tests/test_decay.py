import unittest

import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset
)

from supirfactor_dynamical.models.decay_model import (
    DecayModule
)

from ._stubs import (
    T,
    XV_tensor
)


class TestDecayModule(unittest.TestCase):

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

        return super().setUp()

    def test_training(self):

        decay = DecayModule(4, 2)

        for d in self.velocity_data:

            decay.train_model(
                [torch.stack((
                    d[..., 0],
                    torch.nn.ReLU()(d[..., 1] * -1) * -1
                    ),
                    dim=-1
                )],
                10
            )

            decay.eval()

            with torch.no_grad():

                dout = decay(d[..., 0]).numpy()
                dc_out = decay(d[..., 0], return_decay_constants=True).numpy()

                self.assertEqual(
                    dout.shape,
                    (25, 3, 4)
                )

                self.assertEqual(
                    dc_out.shape,
                    (3, 4)
                )

                self.assertTrue(
                    np.all(dout <= 0)
                )

                self.assertTrue(
                    np.all(dc_out <= 0)
                )

        self.assertEqual(len(decay.training_loss), 10)

    def test_training_static(self):

        decay = DecayModule(4, 2, time_dependent_decay=False)

        for d in self.velocity_data:

            decay.train_model(
                [torch.stack((
                    d[..., 0],
                    torch.nn.ReLU()(d[..., 1] * -1) * -1
                    ),
                    dim=-1
                )],
                10
            )

            decay.eval()

            with torch.no_grad():

                dout = decay(d[..., 0]).numpy()
                dc_out = decay(d[..., 0], return_decay_constants=True).numpy()

                self.assertEqual(
                    dout.shape,
                    (25, 3, 4)
                )

                self.assertEqual(
                    dc_out.shape,
                    (4, )
                )

                self.assertTrue(
                    np.all(dout <= 0)
                )

                self.assertTrue(
                    np.all(dc_out <= 0)
                )

        self.assertEqual(len(decay.training_loss), 10)
