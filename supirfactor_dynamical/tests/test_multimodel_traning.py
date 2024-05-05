import unittest

import pandas as pd
import numpy as np
import numpy.testing as npt

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    TimeDataset,
    get_model,
    process_results_to_dataframes
)
from supirfactor_dynamical.datasets import (
    StackIterableDataset
)

from supirfactor_dynamical.training import (
    train_embedding_submodels,
    train_decoder_submodels
)

from supirfactor_dynamical.perturbation import decoder_loss_transfer


from ._stubs import (
    X,
    A,
    T
)

from supirfactor_dynamical.models.modules import (
    basic_classifier
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

        with self.assertRaises(AttributeError):
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

        res_df, loss_df, _ = process_results_to_dataframes(
            model,
            None,
            model_type='test'
        )

        self.assertEqual(loss_df.shape, (2, 13))
        self.assertEqual(res_df.shape, (1, 3))


class TestDecoderModelTraining(_SetupMixin, unittest.TestCase):

    def test_wrong_model(self):

        model = get_model('static_meta', velocity=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        with self.assertRaises(AttributeError):
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

    def test_swaps_no_freeze(self):

        model = get_model('static_meta', multisubmodel=True)(
            self.prior,
            input_dropout_rate=0.0,
            use_prior_weights=True
        )

        model.add_submodel(
            'test_decoder',
            torch.nn.Sequential(
                torch.nn.Linear(A.shape[1], A.shape[0], bias=False)
            )
        )

        train_decoder_submodels(
            model,
            self.static_dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            decoder_models=('default_decoder', 'test_decoder'),
            validation_dataloader=self.static_dataloader
        )

        res_df, loss_df, _ = process_results_to_dataframes(
            model,
            None,
            model_type=('default_decoder', 'test_decoder')
        )

        self.assertEqual(loss_df.shape, (4, 12))
        self.assertEqual(res_df.shape, (1, 3))


class TestDecoderClassifiers(unittest.TestCase):

    def setUp(self) -> None:

        super().setUp()

        self.dataset = StackIterableDataset(
            torch.Tensor(X),
            torch.LongTensor(T)
        )

        self.dataloader = DataLoader(self.dataset)

        return super().setUp()

    def test_joint_classifier(self):

        model = get_model('static_meta', multisubmodel=True)(
            (4, 3),
            input_dropout_rate=0.0
        )

        model.add_submodel(
            'classifier',
            basic_classifier(3, 4, 4)
        )

        train_decoder_submodels(
            model,
            self.dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            decoder_models=('default_decoder', 'classifier'),
            loss_function=(torch.nn.MSELoss(), torch.nn.CrossEntropyLoss()),
            validation_dataloader=self.dataloader
        )

        res_df, loss_df, _ = process_results_to_dataframes(
            model,
            None,
            model_type=('default_decoder', 'classifier')
        )

        self.assertEqual(loss_df.shape, (4, 12))
        self.assertEqual(res_df.shape, (1, 3))


class TestLossTransfer(unittest.TestCase):

    def setUp(self) -> None:

        super().setUp()

        self.dataset = StackIterableDataset(
            torch.Tensor(X),
            torch.LongTensor(T)
        )

        self.dataloader = DataLoader(self.dataset)

        return super().setUp()

    def test_classifier_to_expr(self):

        model = get_model('static_meta', multisubmodel=True)(
            (4, 3),
            input_dropout_rate=0.0
        )

        model.add_submodel(
            'classifier',
            basic_classifier(3, 4, 4)
        )

        train_decoder_submodels(
            model,
            self.dataloader,
            10,
            optimizer={"lr": 1e-5, "weight_decay": 0.},
            decoder_models=('default_decoder', 'classifier'),
            loss_function=(torch.nn.MSELoss(), torch.nn.CrossEntropyLoss()),
            validation_dataloader=self.dataloader
        )

        _embeds, _predicts = decoder_loss_transfer(
            model,
            torch.Tensor(X),
            torch.LongTensor(T),
            'classifier',
            'default_decoder',
            loss_function=torch.nn.CrossEntropyLoss()
        )

        self.assertEqual(_predicts.shape, X.shape)
        self.assertEqual(_embeds.shape, (X.shape[0], 3))
