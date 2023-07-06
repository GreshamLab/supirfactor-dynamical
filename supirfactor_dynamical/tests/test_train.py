import unittest

import pandas as pd

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    joint_model_training,
    TimeDataset,
    get_model
)

from supirfactor_dynamical.train import pretrain_and_tune_dynamic_model

from supirfactor_dynamical.models import _CLASS_DICT

from ._stubs import (
    X,
    A,
    T,
    XV_tensor
)


class TestCoupledTraining(unittest.TestCase):

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

    def test_training(self):

        results = joint_model_training(
            self.static_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            gold_standard=self.prior
        )

        self.assertEqual(len(results), 4)
        self.assertIsInstance(
            results[0],
            _CLASS_DICT['static']
        )
        self.assertIsInstance(
            results[2],
            _CLASS_DICT['rnn']
        )

    def test_lstm_training(self):

        results = joint_model_training(
            self.static_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            gold_standard=self.prior,
            dynamic_model_type='lstm'
        )

        self.assertEqual(len(results), 4)
        self.assertIsInstance(
            results[0],
            _CLASS_DICT['static']
        )
        self.assertIsInstance(
            results[2],
            _CLASS_DICT['lstm']
        )

    def test_gru_training(self):

        results = joint_model_training(
            self.static_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            gold_standard=self.prior,
            dynamic_model_type='gru'
        )

        self.assertEqual(len(results), 4)
        self.assertIsInstance(
            results[0],
            _CLASS_DICT['static']
        )
        self.assertIsInstance(
            results[2],
            _CLASS_DICT['gru']
        )

    def test_weird_options_training(self):

        results = joint_model_training(
            self.static_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            gold_standard=self.prior,
            input_dropout_rate=0.8,
            hidden_dropout_rate=0.2,
            static_model_type='static_meta'
        )

        self.assertIsInstance(
            results[0],
            _CLASS_DICT['static_meta']
        )
        self.assertIsInstance(
            results[2],
            _CLASS_DICT['rnn']
        )

        self.assertEqual(
            results[0].input_dropout_rate,
            0.8
        )

        self.assertEqual(
            results[2].input_dropout_rate,
            0.8
        )

        self.assertEqual(
            results[0].hidden_dropout_rate,
            0.2
        )

        self.assertEqual(
            results[2].hidden_dropout_rate,
            0.2
        )

    def test_pretrain_tune_model(self):

        results = pretrain_and_tune_dynamic_model(
            self.dynamic_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            prediction_length=1,
            prediction_loss_offset=1,
            gold_standard=self.prior,
        )

        self.assertIsInstance(
            results[0],
            _CLASS_DICT['rnn']
        )

        self.assertEqual(len(results), 3)

    def test_pretrain_tune_tuple_args(self):

        results = pretrain_and_tune_dynamic_model(
            self.dynamic_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            prediction_length=1,
            prediction_loss_offset=1,
            gold_standard=self.prior,
            hidden_dropout_rate=(0.0, 0.5),
            input_dropout_rate=(0.0, 0.25)
        )


class TestVelocityTraining(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(55)

        self.static_data = TimeDataset(
            XV_tensor,
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
            XV_tensor,
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

    def test_velocity_training(self):

        results = joint_model_training(
            self.static_dataloader,
            self.dynamic_dataloader,
            self.prior,
            10,
            gold_standard=self.prior,
            dynamic_model_type=get_model('rnn', velocity=True),
            static_model_type=get_model('static_meta', velocity=True)
        )

        self.assertEqual(len(results), 4)
        self.assertIsInstance(
            results[0],
            _CLASS_DICT['static_meta']
        )
        self.assertIsInstance(
            results[2],
            _CLASS_DICT['rnn']
        )
