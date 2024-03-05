import unittest
import numpy as np
import pandas as pd
import torch

from ._stubs import (
    X,
    PEAKS,
    G_TO_PEAK_PRIOR,
    PEAK_TO_TF_PRIOR
)

from torch.utils.data import DataLoader

from torch.utils.data import (
    StackDataset
)

from supirfactor_dynamical.models.chromatin_model import (
    ChromatinModule,
    ChromatinAwareModel
)

from supirfactor_dynamical.postprocessing.results import (
    process_results_to_dataframes,
    add_classification_metrics_to_dataframe
)


class TestChromatinTraining(unittest.TestCase):

    def setUp(self) -> None:
        self.peaks = PEAKS.copy()
        self.X = X.copy()

        self.data = StackDataset(
            torch.Tensor(self.X),
            torch.Tensor(self.peaks)
        )

        self.dataloader = DataLoader(
            self.data,
            batch_size=2
        )

    def test_train(self):

        model = ChromatinModule(
            n_genes=4,
            n_peaks=25,
            hidden_layer_width=10
        )

        model.train_model(
            self.dataloader,
            10
        )

        self.assertEqual(
            len(model.training_loss),
            10
        )

    def test_classification_results(self):

        model = ChromatinModule(
            n_genes=4,
            n_peaks=25,
            hidden_layer_width=10
        )

        model.train_model(
            self.dataloader,
            10
        )

        results, losses, _ = process_results_to_dataframes(
            model,
            None,
            model_type='chromatin',
            leader_columns=["Name"],
            leader_values=["Value"]
        )

        self.assertEqual(
            results.shape,
            (1, 4)
        )

        results = add_classification_metrics_to_dataframe(
            results,
            model,
            self.dataloader
        )

        self.assertEqual(
            results.shape,
            (1, 5)
        )


class TestChromatinAwareModel(unittest.TestCase):

    def setUp(self) -> None:
        self.peaks = PEAKS.copy()
        self.X = X.copy()

        self.dataloader = DataLoader(
            self.X,
            batch_size=5,
            drop_last=True
        )

    def test_train(self):

        model = ChromatinAwareModel(
            pd.DataFrame(
                G_TO_PEAK_PRIOR,
                index=['A', 'B', 'C', 'D'],
                columns=[chr(65 + x) for x in range(25)]
            ),
            pd.DataFrame(
                PEAK_TO_TF_PRIOR,
                index=[chr(65 + x) for x in range(25)],
                columns=['TF1', 'TF2', 'TF3']
            )
        )

        model.train_model(
            self.dataloader,
            10
        )

        self.assertEqual(
            len(model.training_loss),
            10
        )

        _erv, _rss, _full = model.erv(
            self.dataloader,
            return_rss=True,
            as_data_frame=True
        )

        print(_erv)

        self.assertEqual(_erv.shape, (4, 3))
        self.assertEqual(_rss.shape, (4, 3))
        self.assertEqual(_full.shape, (4, 1))

    def test_train_but_not_chromatin(self):

        with self.assertRaises(RuntimeError):
            model = ChromatinAwareModel(
                G_TO_PEAK_PRIOR,
                PEAK_TO_TF_PRIOR,
                train_chromatin_model=False
            )

        c_module = ChromatinModule(
            n_genes=4,
            n_peaks=25,
            hidden_layer_width=10
        )

        model = ChromatinAwareModel(
            G_TO_PEAK_PRIOR,
            PEAK_TO_TF_PRIOR,
            chromatin_model=c_module,
            train_chromatin_model=False
        )

        model.train_model(
            self.dataloader,
            10
        )

        self.assertEqual(
            len(model.training_loss),
            10
        )

        x = next(iter(self.dataloader))

        peaks = model.chromatin_encoder(x)
        torch.testing.assert_close(
            peaks,
            (model.chromatin_model(x) > 0.5).float()
        )
