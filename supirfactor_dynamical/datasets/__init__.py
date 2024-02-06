from .time_dataset import TimeDataset

from .multimodal_dataset import (
    MultimodalDataset,
    MultimodalDataLoader
)

from .anndata_backed_dataset import (
    H5ADDataset,
    H5ADDatasetIterable,
    H5ADDatasetStratified
)

class ChromatinDataset(MultimodalDataset):
    pass

class ChromatinDataLoader(MultimodalDataLoader):
    pass
