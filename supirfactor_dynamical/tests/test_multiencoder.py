import unittest

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset
)

from supirfactor_dynamical.models import (
    get_model
)

from ._stubs import (
    A,
    T,
    X_tensor
)


class TestMultiEncoderMixin(unittest.TestCase):

    model = 'static'

    def setUp(self) -> None:

        self.model = get_model(
            self.model,
            multisubmodel=True
        )(A, use_prior_weights=True)

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

        self.model.add_submodel(
            'lots_o_layers',
            torch.nn.Sequential(
                torch.nn.Linear(4, 3),
                torch.nn.ReLU(),
                torch.nn.Linear(3, 3),
                torch.nn.Tanh(),
                torch.nn.Linear(3, 3)
            )
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

        self.model.train_model(self.count_data, 10)

        x = self.model(X_tensor)
        self.assertEqual(x.shape, X_tensor.shape)

        self.assertEqual(
            len(self.model._training_loss),
            10
        )

        self.assertEqual(
            self.model.training_loss_df.shape,
            (1, 11)
        )

    def test_param_swap(self):

        self.assertEqual(
            self.model.active_encoder, 'default_encoder'
        )

        self.assertEqual(
            len(self.model.encoder), 2
        )

        self.assertEqual(
            id(self.model._module_bag['default_encoder']),
            id(self.model.encoder)
        )

        self.model.select_submodel('lots_o_layers')

        self.assertEqual(
            id(self.model._module_bag['lots_o_layers']),
            id(self.model.encoder)
        )

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
            (1, 11)
        )

        torch.testing.assert_close(
            torch.tensor(A.T),
            self.model.module_bag['default_encoder'][0].weight
        )

    def test_param_freeze(self):

        self.model.freeze_submodel('encoder')

        for p in self.model.encoder.parameters():
            self.assertFalse(p.requires_grad)

        torch.testing.assert_close(
            torch.tensor(A.T),
            self.model.encoder[0].weight
        )

        self.model.train_model(self.count_data, 10)

        torch.testing.assert_close(
            torch.tensor(A.T),
            self.model.encoder[0].weight
        )

        decoder_weight = torch.clone(self.model._decoder[0].weight.detach())

        self.model.freeze_submodel('encoder', unfreeze=True)

        for p in self.model.encoder.parameters():
            self.assertTrue(p.requires_grad)

        self.model.freeze_submodel('decoder')

        self.model.train_model(self.count_data, 20)

        torch.testing.assert_close(
            decoder_weight,
            self.model._decoder[0].weight
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                torch.tensor(A.T),
                self.model.encoder[0].weight
            )

    def test_param_train(self):

        for p in self.model.encoder.parameters():
            self.assertTrue(p.requires_grad)

        torch.testing.assert_close(
            torch.tensor(A.T),
            self.model.encoder[0].weight
        )

        if len(self.model._intermediate) > 0:
            intermediate_weight = torch.clone(
                self.model._intermediate[0].weight.detach()
            )

        decoder_weight = torch.clone(self.model._decoder[0].weight.detach())
        self.model.train_model(self.count_data, 10)

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                torch.tensor(A.T),
                self.model.encoder[0].weight
            )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                decoder_weight,
                self.model._decoder[0].weight
            )

        if len(self.model._intermediate) > 0:
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(
                    intermediate_weight,
                    self.model._intermediate[0].weight
                )
