
import torch

from .time_dataset import TimeDataset


class StackIterableDataset(
    torch.utils.data.IterableDataset
):

    datasets = None

    def __init__(self, *args):

        super().__init__()
        self.datasets = args

    def __iter__(self):
        for d in zip(*self.datasets):
            yield d


from .anndata_backed_dataset import (
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified,
    H5ADDatasetObsStratified
)
