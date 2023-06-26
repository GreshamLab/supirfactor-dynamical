import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset
)

from supirfactor_dynamical.models import (
    get_model
)

from ._stubs import (
    X,
    X_tensor,
    A,
    T
)

V = np.diff(X, axis=0, prepend=np.zeros((1, 4), dtype=np.float32))
V_tensor = torch.tensor(V)


class TestVelocity(unittest.TestCase):

    model = 'static_meta'

    def test_constructor(self):

        normal_model = get_model(self.model)

        self.assertFalse(
            hasattr(normal_model, 'initalize_decay_module')
        )

        self.assertFalse(normal_model._velocity_model)
        self.assertFalse(normal_model._decay_model)

        velo_model = get_model(self.model, velocity=True)

        self.assertFalse(
            hasattr(velo_model, 'initalize_decay_module')
        )

        self.assertTrue(velo_model._velocity_model)
        self.assertFalse(velo_model._decay_model)

        decay_model = get_model(self.model, velocity=True, decay=True)

        self.assertTrue(
            hasattr(decay_model, 'initalize_decay_module')
        )

        self.assertTrue(decay_model._velocity_model)
        self.assertTrue(decay_model._decay_model)

    def test_velocity_data(self):

        data = DataLoader(
            TimeDataset(
                torch.stack((X_tensor, V_tensor), dim=2),
                T,
                0,
                4,
                1,
                sequence_length=3
            ),
            batch_size=25
        )

        normal_model = get_model(self.model)(
            A,
            use_prior_weights=True
        )

        velo_model = get_model(self.model, velocity=True)(
            A,
            use_prior_weights=True
        )

        for d in data:
            self.assertEqual(
                d.shape,
                (25, 3, 4, 2)
            )

            self.assertEqual(
                normal_model.input_data(d).shape,
                (25, 3, 4, 2)
            )

            self.assertEqual(
                normal_model.output_data(d).shape,
                (25, 3, 4, 2)
            )

            self.assertEqual(
                velo_model.input_data(d).shape,
                (25, 3, 4)
            )

            self.assertEqual(
                velo_model.output_data(d).shape,
                (25, 3, 4)
            )

            npt.assert_almost_equal(
                velo_model.input_data(
                    torch.stack((X_tensor, V_tensor), dim=2)
                ).numpy(),
                X
            )

            npt.assert_almost_equal(
                velo_model.output_data(
                    torch.stack((X_tensor, V_tensor), dim=2)
                ).numpy(),
                V
            )

    def test_velocity_data_offsets(self):

        data = DataLoader(
            TimeDataset(
                torch.stack((X_tensor, V_tensor), dim=2),
                T,
                0,
                4,
                1,
                sequence_length=3
            ),
            batch_size=25
        )

        normal_model = get_model(self.model)(
            A,
            use_prior_weights=True
        ).set_time_parameters(
            output_t_plus_one=True
        )

        velo_model = get_model(self.model, velocity=True)(
            A,
            use_prior_weights=True
        ).set_time_parameters(
            output_t_plus_one=True
        )

        if self.model in ['static', 'static_meta']:
            n=1
        else:
            n=2

        for d in data:
            self.assertEqual(
                d.shape,
                (25, 3, 4, 2)
            )

            self.assertEqual(
                normal_model.input_data(d).shape,
                (25, n, 4, 2)
            )

            self.assertEqual(
                normal_model.output_data(d).shape,
                (25, n, 4, 2)
            )

            self.assertEqual(
                velo_model.input_data(d).shape,
                (25, n, 4)
            )

            self.assertEqual(
                velo_model.output_data(d).shape,
                (25, n, 4)
            )


class TestVelocityRNN(TestVelocity):

    model = 'rnn'
