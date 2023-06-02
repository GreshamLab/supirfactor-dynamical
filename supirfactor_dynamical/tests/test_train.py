import unittest

import pandas as pd

import torch
from torch.utils.data import DataLoader

from supirfactor_dynamical import (
    joint_model_training,
    TimeDataset
)

from ._stubs import (
    X,
    A,
    T
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
