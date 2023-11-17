import unittest

from ._stubs import (
    X,
    X_SP,
    PEAKS,
    PEAKS_SP
)

from supirfactor_dynamical.datasets import (
    ChromatinDataset,
    ChromatinDataLoader
)


class TestChromatinDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.peaks = PEAKS.copy()
        self.X = X.copy()

    def test_init(self):

        data = ChromatinDataset(
            self.X,
            self.peaks
        )

        self.assertEqual(
            len(data),
            100
        )

        dataloader = ChromatinDataLoader(
            data,
            batch_size=2
        )

        ld = [d for d in dataloader]

        self.assertEqual(
            len(ld),
            50
        )

        self.assertEqual(
            ld[0][0].shape,
            (2, 4)
        )

        self.assertEqual(
            ld[0][1].shape,
            (2, 25)
        )

    def test_misaligned(self):

        with self.assertRaises(ValueError):
            ChromatinDataset(
                self.X,
                self.peaks.T
            )


class TestChromatinDatasetSparse(TestChromatinDataset):

    def setUp(self) -> None:
        self.peaks = PEAKS_SP.copy()
        self.X = X_SP.copy()
