
import torch
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


from .anndata_backed_dataset import (
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified,
    H5ADDatasetObsStratified
)

from .stratified_file_dataset import (
    StratifiedFilesDataset
)


def stack_dataloaders(loaders):

    if loaders is None:
        return None

    # If the loader is a DataLoader or a Tensor
    # yield from it, making this essentially a
    # noop
    elif (
        isinstance(loaders, DataLoader) or
        torch.is_tensor(loaders)
    ):
        yield from loaders

    # Otherwise treat it as an iterable of generators
    else:
        for loader in loaders:
            # IF this is a tensor, yield it
            # because it's probably a list of tensors
            # for training
            if torch.is_tensor(loader):
                yield loader

            # Otherwise yield from it because it's
            # probably a DataLoader
            else:
                yield from loader


def _shuffle_time_data(dl):
    if dl is None:
        return None

    try:
        dl.dataset.shuffle()
    except AttributeError:
        pass

    if (
        not isinstance(dl, DataLoader) and
        not torch.is_tensor(dl)
    ):
        for d in dl:
            _shuffle_time_data(d)
