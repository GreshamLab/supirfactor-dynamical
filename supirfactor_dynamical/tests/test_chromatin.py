import unittest

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
            4,
            25,
            k=10
        )

        model.train_model(
            self.dataloader,
            10
        )

        self.assertEqual(
            len(model.training_loss),
            10
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
