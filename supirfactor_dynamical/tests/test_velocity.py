import unittest

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
    T,
    V,
    V_tensor
)


class TestVelocity(unittest.TestCase):

    model = 'static_meta'

    def setUp(self) -> None:
        super().setUp()

        self.data = DataLoader(
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

    def test_constructor(self):

        normal_model = get_model(self.model)
        self.assertFalse(normal_model._velocity_model)

        velo_model = get_model(self.model, velocity=True)

        self.assertTrue(velo_model._velocity_model)

    def test_velocity_data(self):

        normal_model = get_model(self.model)(
            A,
            use_prior_weights=True
        )

        velo_model = get_model(self.model, velocity=True)(
            A,
            use_prior_weights=True
        )

        for d in self.data:
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
            n = 1
        else:
            n = 2

        for d in self.data:
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

    def test_scaling(self):

        v_scale = torch.Tensor([1, 1.5, 2, 2.5])
        c_scale = torch.Tensor([2, 1, 0.5, 1])

        c = torch.div(X_tensor, c_scale[None, :])
        v = torch.div(V_tensor, v_scale[None, :])

        model = get_model(self.model, velocity=True)(
            A,
            use_prior_weights=True
        )

        model.set_scaling(
            count_scaling=c_scale
        )

        with torch.no_grad():

            npt.assert_almost_equal(
                np.diag(c_scale),
                model.scaler.numpy()
            )
            npt.assert_almost_equal(
                model.scale_count_to_velocity(c).numpy(),
                X_tensor.numpy()
            )

        model.set_scaling(
            count_scaling=None,
            velocity_scaling=v_scale
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                np.diag(v_scale),
                model.inv_scaler.numpy()
            )

            npt.assert_almost_equal(
                model.scale_velocity_to_count(v).numpy(),
                V_tensor.numpy()
            )

        model.set_scaling(
            count_scaling=c_scale,
            velocity_scaling=v_scale
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                np.diag(v_scale / c_scale),
                model.inv_scaler.numpy()
            )

            npt.assert_almost_equal(
                np.diag(c_scale / v_scale),
                model.scaler.numpy()
            )
