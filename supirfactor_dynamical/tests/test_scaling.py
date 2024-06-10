import unittest

import numpy.testing as npt

import torch

from supirfactor_dynamical.models._model_mixins import (
    _ScalingMixin
)

from ._stubs import (
    X_tensor,
    V_tensor
)


v_scale = torch.Tensor([1, 1.5, 2, 2.5])
c_scale = torch.Tensor([2, 1, 0.5, 1])

c = torch.div(X_tensor, c_scale[None, :])
v = torch.div(V_tensor, v_scale[None, :])

r = torch.rand(10, 4)
rc = torch.div(r, c_scale[None, :])
rv = torch.div(r, v_scale[None, :])


class TestScalerMixin(unittest.TestCase):

    def setUp(self):
        self.model = _ScalingMixin()

    def test_count_scaling(self):

        self.model.set_scaling(
            count_scaling=c_scale
        )

        with torch.no_grad():

            npt.assert_almost_equal(
                c_scale,
                self.model.count_scaler.numpy()
            )
            npt.assert_almost_equal(
                self.model.scale_count_to_velocity(c).numpy(),
                X_tensor.numpy()
            )
            npt.assert_almost_equal(
                self.model.unscale_counts(c).numpy(),
                X_tensor.numpy()
            )
            npt.assert_almost_equal(
                self.model.rescale_counts(X_tensor).numpy(),
                c.numpy()
            )

    def test_velocity_scaling(self):

        self.model.set_scaling(
            count_scaling=None,
            velocity_scaling=v_scale
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                v_scale,
                self.model.velocity_to_count_scaler.numpy()
            )

            npt.assert_almost_equal(
                self.model.scale_velocity_to_count(v).numpy(),
                V_tensor.numpy()
            )
            npt.assert_almost_equal(
                self.model.unscale_velocity(v).numpy(),
                V_tensor.numpy()
            )
            npt.assert_almost_equal(
                self.model.rescale_velocity(V_tensor).numpy(),
                v.numpy()
            )

    def test_both_scaling(self):
        self.model.set_scaling(
            count_scaling=c_scale,
            velocity_scaling=v_scale
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                v_scale / c_scale,
                self.model.velocity_to_count_scaler.numpy()
            )

            npt.assert_almost_equal(
                c_scale / v_scale,
                self.model.count_to_velocity_scaler.numpy()
            )

            torch.testing.assert_close(
                self.model.scale_velocity_to_count(rv),
                rc
            )

            torch.testing.assert_close(
                self.model.scale_count_to_velocity(rc),
                rv
            )
