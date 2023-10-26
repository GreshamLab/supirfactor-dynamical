import unittest
import numpy as np

from ._stubs import (
    X,
    X_SP,
    PEAKS,
    PEAKS_SP,
    G_TO_PEAK_PRIOR,
    PEAK_TO_TF_PRIOR
)

from torch.utils.data import DataLoader

from supirfactor_dynamical._utils.chromatin_dataset import (
    ChromatinDataset,
    ChromatinDataLoader
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

        self.data = ChromatinDataset(
            self.X,
            self.peaks
        )

        self.dataloader = ChromatinDataLoader(
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
            (1, 9)
        )

        self.assertEqual(
            results.iloc[0, 7],
            np.sum(PEAKS == 0) / PEAKS.size
        )

        self.assertEqual(
            results.iloc[0, 8],
            np.sum(PEAKS == 1) / PEAKS.size
        )


class TestChromatinTrainingSparse(TestChromatinTraining):

    def setUp(self) -> None:
        self.peaks = PEAKS_SP.copy()
        self.X = X_SP.copy()

        self.data = ChromatinDataset(
            self.X,
            self.peaks
        )

        self.dataloader = ChromatinDataLoader(
            self.data,
            batch_size=2
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
            G_TO_PEAK_PRIOR,
            PEAK_TO_TF_PRIOR
        )

        model.train_model(
            self.dataloader,
            10
        )

        self.assertEqual(
            len(model.training_loss),
            10
        )

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
