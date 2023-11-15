import unittest

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset
)

from supirfactor_dynamical.models import (
    TFAutoencoder
)

from supirfactor_dynamical.models._model_mixins import (
    _MultiencoderModuleMixin
)

from ._stubs import (
    A,
    T,
    X_tensor
)


class TestMultiEncoderMixin(unittest.TestCase):

    def setUp(self) -> None:

        class _multimodel(
            _MultiencoderModuleMixin,
            TFAutoencoder
        ):
            pass

        self.model = _multimodel(A)

        self.count_data = DataLoader(
            TimeDataset(
                X_tensor,
                T,
                0,
                3,
                1,
                sequence_length=3
            ),
            batch_size=25
        )

        return super().setUp()

    def test_one_encoder(self):

        self.model.train_model(self.count_data, 10)

        x = self.model(X_tensor)
        self.assertEqual(x.shape, X_tensor.shape)

        self.assertEqual(
            len(self.model._training_loss),
            10
        )

    def test_two_encoders(self):

        self.model.add_encoder(
            'lots_o_layers',
            torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 3),
                torch.nn.Tanh(),
                torch.nn.Linear(3, 3)
            )
        )

        self.model.train_model(self.count_data, 10)

        x = self.model(X_tensor)
        self.assertEqual(x.shape, X_tensor.shape)

        self.assertEqual(
            len(self.model._training_loss),
            10
        )

        self.assertEqual(
            self.model.training_loss_df.shape,
            (2, 11)
        )

    def test_param_freeze(self):

        self.model.add_encoder(
            'lots_o_layers',
            torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 3),
                torch.nn.Tanh(),
                torch.nn.Linear(3, 3)
            )
        )

        self.assertEqual(
            self.model.active_encoder, 'linear'
        )

        self.assertEqual(
            len(self.model.encoder), 2
        )

        self.model.select_encoder('lots_o_layers')

        self.assertEqual(
            self.model.active_encoder, 'lots_o_layers'
        )

        self.assertEqual(
            len(self.model.encoder), 5
        )

        self.model.train_model(self.count_data, 10)

        x = self.model(X_tensor)
        self.assertEqual(x.shape, X_tensor.shape)

        self.assertEqual(
            len(self.model._training_loss),
            10
        )

        self.assertEqual(
            self.model.training_loss_df.shape,
            (2, 11)
        )
