import unittest

from ._stubs import (
    X,
    X_SP,
    PEAKS,
    PEAKS_SP
)

from supirfactor_dynamical._utils.chromatin_dataset import (
    ChromatinDataset,
    ChromatinDataLoader
)

from supirfactor_dynamical.models.chromatin_model import (
    ChromatinModule
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