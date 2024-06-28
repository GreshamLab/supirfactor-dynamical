import unittest
import pandas as pd
import torch
import tempfile
import os
from torch.utils.data import DataLoader

from supirfactor_dynamical import train_simple_model, read
from supirfactor_dynamical.postprocessing.results import (
    add_classification_metrics_to_dataframe
)
from supirfactor_dynamical.datasets import StackIterableDataset
from supirfactor_dynamical.models.simple_models import (
    LogisticRegressionTorch
)

from ._stubs import (
    X_tensor,
    T
)


class TestLogisticClass(unittest.TestCase):

    device = 'cpu'

    def setUp(self) -> None:
        torch.manual_seed(55)

        self.data = DataLoader(
            StackIterableDataset(
                X_tensor[0:50, :],
                torch.torch.tensor(T[0:50], dtype=torch.long)
            ),
            batch_size=5
        )

    def test_create_model(self):

        model = LogisticRegressionTorch(
            (4, 4),
            bias=True
        )

        self.assertEqual(model.prior_network, (4, 4))
        self.assertEqual(len(model.classifier), 3)

        self.assertEqual(
            model(X_tensor).shape,
            (100, 4)
        )

    def test_train_model(self):

        model = LogisticRegressionTorch(
            (4, 2),
            bias=True
        )

        train_simple_model(
            model,
            self.data,
            10,
            device=self.device,
            validation_dataloader=self.data,
            loss_function=torch.nn.CrossEntropyLoss(),
            separate_output_data=True
        )

        model.to('cpu')

        df = add_classification_metrics_to_dataframe(
            pd.DataFrame(['t', 'v']),
            model,
            self.data,
            self.data,
            column_prefix='simple',
            add_class_counts=True
        )

        self.assertEqual(df.shape, (2, 11))
        self.assertEqual(model.training_loss_df.shape, (1, 11))
        self.assertEqual(model.validation_loss_df.shape, (1, 11))

    def test_serialize(self):

        model = LogisticRegressionTorch(
            (4, 2),
            bias=True
        )

        with tempfile.TemporaryDirectory() as tempdir:
            _model_file = os.path.join(tempdir, 'model.h5')
            model.save(_model_file)
            model_load = read(_model_file)

            torch.testing.assert_close(
                model.classifier[1].weight,
                model_load.classifier[1].weight
            )
            torch.testing.assert_close(
                model.classifier[1].bias,
                model_load.classifier[1].bias
            )


class TestLogisticClassCUDA(TestLogisticClass):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
