import unittest
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    predict_perturbation,
    SupirFactorBiophysical
)

from supirfactor_dynamical.perturbation.predict import (
    _get_perturbed_velocities
)

from ._stubs import (
    A,
    T,
    XV_tensor,
    XTV_tensor
)


class TestPerturbBiophysical(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        torch.manual_seed(100)

        cls.count_data = DataLoader(
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

        cls.velocity_data = DataLoader(
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

        cls.ordered_data = torch.stack(
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

        cls.dynamical_model = SupirFactorBiophysical(
            pd.DataFrame(A),
            decay_model=None,
            decay_k=4
        )

        cls.dynamical_model.train_model(
            cls.velocity_data,
            10
        ).eval()

    def setUp(self):
        super().setUp()

        self.dynamical_model.set_drop_tfs(None)

    def test_x_decay(self):

        in_data = self.dynamical_model.input_data(XTV_tensor)

        _, vdecay = self.dynamical_model._decay_model(
            in_data,
            return_decay_constants=True
        )

        vdecay_2 = self.dynamical_model(
            in_data,
            0,
            return_decays=True
        )

        torch.testing.assert_close(
            vdecay,
            vdecay_2
        )

    def test_x_velocity(self):

        self.dynamical_model.eval()

        in_data = self.dynamical_model.input_data(XTV_tensor)

        velo, dr = self.dynamical_model(
            in_data,
            return_velocities=True,
            return_decays=True,
            return_submodels=True,
            unmodified_counts=True
        )

        pos_velocity, neg_velocity = _get_perturbed_velocities(
            self.dynamical_model,
            in_data,
            dr,
            return_submodels=True
        )

        torch.testing.assert_close(
            torch.multiply(
                in_data,
                dr[None, ...]
            ),
            velo[1]
        )

        torch.testing.assert_close(
            neg_velocity,
            velo[1]
        )

        torch.testing.assert_close(
            pos_velocity,
            velo[0]
        )

    def test_no_perturb_no_predict(self):

        in_data = self.dynamical_model.input_data(XTV_tensor)

        model_fit = self.dynamical_model(
            in_data,
            n_time_steps=0,
            return_velocities=True,
            return_submodels=True,
            return_counts=True
        )

        predicts = predict_perturbation(
            self.dynamical_model,
            in_data,
            perturbation=None,
            return_submodels=True,
            n_time_steps=0,
            unmodified_counts=False
        )

        self.assertEqual(
            predicts[0][0].shape,
            (25, 4, 4)
        )

        torch.testing.assert_close(
            model_fit[0][0],
            predicts[0][0]
        )

        torch.testing.assert_close(
            model_fit[0][1],
            predicts[0][1]
        )

        torch.testing.assert_close(
            model_fit[1],
            predicts[1]
        )

    def test_no_perturb_prediction(self):

        in_data = self.dynamical_model.input_data(XTV_tensor)

        model_fit = self.dynamical_model(
            in_data,
            n_time_steps=5,
            return_velocities=True,
            return_counts=True,
            return_submodels=True
        )

        predicts = predict_perturbation(
            self.dynamical_model,
            in_data,
            perturbation=None,
            n_time_steps=5,
            return_submodels=True,
            unmodified_counts=False
        )

        self.assertEqual(
            predicts[0][0].shape,
            (25, 9, 4)
        )

        torch.testing.assert_close(
            model_fit[0][0],
            predicts[0][0]
        )

        torch.testing.assert_close(
            model_fit[0][1],
            predicts[0][1]
        )

        torch.testing.assert_close(
            model_fit[1],
            predicts[1]
        )

    def test_perturb_prediction(self):

        in_data = self.dynamical_model.input_data(XTV_tensor)

        model_fit = self.dynamical_model(
            in_data,
            n_time_steps=5,
            return_velocities=True,
            return_counts=True,
            return_submodels=True,
            return_decays=True,
            unmodified_counts=True
        )

        predicts = predict_perturbation(
            self.dynamical_model,
            in_data,
            perturbation=1,
            n_time_steps=5,
            return_submodels=True,
            unmodified_counts=True
        )

        self.assertEqual(
            predicts[0][0].shape,
            (25, 9, 4)
        )

        torch.testing.assert_close(
            predicts[2],
            model_fit[2]
        )

        torch.testing.assert_close(
            predicts[0][1][:, 0:4, :],
            model_fit[0][1][:, 0:4, :]
        )

        torch.testing.assert_close(
            predicts[1][:, 0:4, :],
            model_fit[1][:, 0:4, :]
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                predicts[0][1][:, 4:, :],
                model_fit[0][1][:, 4:, :]
            )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                predicts[0][0][:, 0:4, :],
                model_fit[0][0][:, 0:4, :]
            )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                predicts[1][:, 4:, :],
                model_fit[1][:, 4:, :]
            )

    def test_perturbation_prediction_badtf(self):

        with self.assertRaises(RuntimeError):
            predict_perturbation(
                self.dynamical_model,
                self.dynamical_model.input_data(XTV_tensor),
                perturbation="Q",
                n_time_steps=5
            )
