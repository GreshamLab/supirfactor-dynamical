
import torch
import copy
from torch.utils.data import DataLoader

from .time_dataset import TimeDataset, TimeDatasetIter


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


class DataLoaderStack:

    num_loaders = None
    epoch = None

    _loaders = None
    _iterators = None

    def __init__(
        self,
        dataset,
        num_loaders,
        dataset_init_fn=None,
        **kwargs
    ):

        self.num_loaders = num_loaders
        self.epoch = -1
        self._loaders = []
        self._iterators = []

        for i in range(num_loaders):

            _dataset = copy.deepcopy(dataset)
            if dataset_init_fn is not None:
                dataset_init_fn(_dataset, i)

            self._loaders.append(
                DataLoader(
                    _dataset,
                    **kwargs
                )
            )

        for loader in self._loaders:
            self._iterators.append(
                iter(loader)
            )

    def __iter__(self):

        self.epoch = self.epoch + 1
        _position = self.epoch % self.num_loaders
        yield from self._iterators[_position]
        self._iterators[_position] = iter(self._loaders[_position])


from .anndata_backed_dataset import (
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified,
    H5ADDatasetObsStratified
)

from .stratified_file_dataset import (
    StratifiedFilesDataset
)
