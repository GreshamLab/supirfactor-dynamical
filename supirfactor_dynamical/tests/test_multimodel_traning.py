import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    get_model
)

from supirfactor_dynamical.training import (
    train_embedding_submodels,
    train_decoder_submodels
)

from ._stubs import (
    X,
    A,
    T
)


class _SetupMixin:

    def setUp(self) -> None:
        torch.manual_seed(55)

        self.static_data = TimeDataset(
            X,
            T,
            0,
            1
        )

        self.static_dataloader = DataLoader(
            self.static_data,
            batch_size=2,
            drop_last=True
        )

        self.dynamic_data = TimeDataset(
            X,
            T,
            0,
            4,
            t_step=1
        )

        self.dynamic_dataloader = DataLoader(
            self.dynamic_data,
            batch_size=2,
            drop_last=True
        )

        self.prior = pd.DataFrame(
            A,
            index=['A', 'B', 'C', 'D'],
            columns=['A', 'B', 'C']
        )


class TestEmbeddingModelTraining(_SetupMixin, unittest.TestCase):

    def test_wrong_model(self):

        model = get_model('static_meta', velocity=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        with self.assertRaises(RuntimeError):
            train_embedding_submodels(
                model,
                self.static_dataloader,
                10,
                optimizer={"lr": 1e-5, "weight_decay": 0.},
                validation_dataloader=self.static_dataloader
            )

    def test_no_swap_training(self):

        model = get_model('static_meta', multisubmodel=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        mw = model.encoder[0].weight.detach()

        train_embedding_submodels(
            model,
            self.static_dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            validation_dataloader=self.static_dataloader
        )

        torch.testing.assert_close(
            model.encoder[0].weight.detach(),
            mw
        )

        npt.assert_almost_equal(
            model.training_loss,
            np.zeros_like(model.training_loss)
        )

    def test_swap_training(self):

        model = get_model('static_meta', multisubmodel=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        model.add_submodel(
            'extra',
            torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Softplus(),
                torch.nn.Linear(4, 3)
            )
        )

        train_embedding_submodels(
            model,
            self.static_dataloader,
            10,
            training_encoder='extra',
            validation_dataloader=self.static_dataloader
        )

        self.assertTrue(
            np.all(np.diff(model.training_loss) <= 0)
        )

        self.assertEqual(
            model.active_encoder,
            'extra'
        )

        model.select_submodel(
            'default_encoder',
            'encoder'
        )

        torch.testing.assert_close(
            model.encoder[0].weight.detach(),
            torch.tensor(A).transpose(0, 1)
        )


class TestDecoderModelTraining(_SetupMixin, unittest.TestCase):

    def test_wrong_model(self):

        model = get_model('static_meta', velocity=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        with self.assertRaises(RuntimeError):
            train_decoder_submodels(
                model,
                self.static_dataloader,
                10,
                decoder_models=('default_decoder', 'bad_decoder'),
                optimizer={"lr": 1e-5, "weight_decay": 0.}
            )

    def test_no_swaps_no_freeze(self):

        model = get_model('static_meta', multisubmodel=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        train_decoder_submodels(
            model,
            self.static_dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            validation_dataloader=self.static_dataloader
        )

        self.assertTrue(
            np.all(np.diff(model.training_loss) <= 0)
        )

        with self.assertRaises(AssertionError):
            torch.testing.assert_close(
                model.encoder[0].weight.detach(),
                torch.tensor(A).transpose(0, 1)
            )

    def test_no_swaps_freeze(self):

        model = get_model('static_meta', multisubmodel=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        train_decoder_submodels(
            model,
            self.static_dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            freeze_embeddings=True,
            validation_dataloader=self.static_dataloader
        )

        self.assertTrue(
            np.all(np.diff(model.training_loss) <= 0)
        )

        torch.testing.assert_close(
            model.encoder[0].weight.detach(),
            torch.tensor(A).transpose(0, 1)
        )
